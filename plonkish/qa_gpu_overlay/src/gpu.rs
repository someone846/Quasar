use plonkish_backend::{
    pcs::multilinear::quasar::{QACodewordColumns, QACodewordRows, QAParams},
    util::{arithmetic::Field, new_fields::Mersenne127},
};
use rayon::prelude::*;
use std::{
    ffi::{c_char, c_int, c_void, CStr},
    mem::{align_of, size_of, ManuallyDrop, MaybeUninit},
    ptr::NonNull,
    sync::{
        atomic::{AtomicU64, Ordering},
        Mutex,
    },
    time::{Duration, Instant},
};

#[cfg(not(target_endian = "little"))]
compile_error!("the direct Mersenne127 Montgomery-limb CUDA path requires a little-endian host");

/// Exact C/CUDA view of `Mersenne127([u64; 2])` after the backend type has
/// been marked `#[repr(transparent)]`.
///
/// Do not add `align(16)` here: `[u64; 2]` has alignment 8, and the Rust and
/// CUDA ABI views must agree exactly.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct M127Mont {
    lo: u64,
    hi: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct RawGpuTiming {
    host_to_device_ms: f32,
    device_input_copy_ms: f32,
    first_wht_ms: f32,
    scaling_ms: f32,
    second_wht_ms: f32,
    assemble_ms: f32,
    device_to_host_ms: f32,
    total_cuda_ms: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct RawGpuMerkleTiming {
    column_hash_ms: f32,
    digest_device_to_host_ms: f32,
    total_cuda_ms: f32,
}

#[derive(Clone, Debug, Default)]
pub struct GpuQaTiming {
    pub host_to_device_ms: f64,
    pub device_input_copy_ms: f64,
    pub first_wht_ms: f64,
    pub scaling_ms: f64,
    pub second_wht_ms: f64,
    pub assemble_ms: f64,
    pub device_to_host_ms: f64,
    pub column_hash_ms: f64,
    pub digest_device_to_host_ms: f64,
    pub host_leaf_decode: Duration,
    pub cpu_upper_merkle: Duration,
    pub total_cuda_ms: f64,
    pub ffi_wall: Duration,
    pub total_wall: Duration,
}

#[derive(Clone, Debug, Default)]
pub struct GpuQaDeviceOutputSetupTiming {
    pub device_allocation: Duration,
    pub digest_allocation_and_prefault: Duration,
    pub pin_registration: Duration,
    pub total: Duration,
}

#[derive(Clone, Debug, Default)]
pub struct GpuQaOutputSetupTiming {
    pub allocation: Duration,
    pub prefault_and_initialize: Duration,
    pub pin_registration: Duration,
    pub total: Duration,
}

#[derive(Clone, Debug, Default)]
pub struct GpuQaInputSetupTiming {
    pub pin_registration: Duration,
}

impl GpuQaTiming {
    fn add_raw(&mut self, raw: RawGpuTiming) {
        self.host_to_device_ms += raw.host_to_device_ms as f64;
        self.device_input_copy_ms += raw.device_input_copy_ms as f64;
        self.first_wht_ms += raw.first_wht_ms as f64;
        self.scaling_ms += raw.scaling_ms as f64;
        self.second_wht_ms += raw.second_wht_ms as f64;
        self.assemble_ms += raw.assemble_ms as f64;
        self.device_to_host_ms += raw.device_to_host_ms as f64;
        self.total_cuda_ms += raw.total_cuda_ms as f64;
    }

    fn add_merkle_raw(&mut self, raw: RawGpuMerkleTiming) {
        self.column_hash_ms += raw.column_hash_ms as f64;
        self.digest_device_to_host_ms += raw.digest_device_to_host_ms as f64;
        self.device_to_host_ms += raw.digest_device_to_host_ms as f64;
        self.total_cuda_ms += raw.total_cuda_ms as f64;
    }
}

extern "C" {
    fn qa_gpu_create_m127_mont(
        coefficients: *const M127Mont,
        row_len: usize,
        inverse_rate: u32,
        max_batch_rows: u32,
    ) -> *mut c_void;
    fn qa_gpu_encode_m127_mont(
        context: *mut c_void,
        messages: *const M127Mont,
        rows: u32,
        codewords: *mut M127Mont,
        timing: *mut RawGpuTiming,
    ) -> c_int;
    fn qa_gpu_destroy_m127_mont(context: *mut c_void);
    fn qa_gpu_device_name_m127_mont(
        context: *mut c_void,
        output: *mut c_char,
        capacity: usize,
    ) -> c_int;
    fn qa_gpu_host_register_m127_mont(values: *mut M127Mont, elements: usize) -> c_int;
    fn qa_gpu_host_unregister_m127_mont(values: *mut M127Mont) -> c_int;
    fn qa_gpu_host_register_bytes(values: *mut c_void, bytes: usize) -> c_int;
    fn qa_gpu_host_unregister_bytes(values: *mut c_void) -> c_int;
    fn qa_gpu_create_device_commitment_m127_mont(
        rows: usize,
        columns: usize,
    ) -> *mut c_void;
    fn qa_gpu_destroy_device_commitment_m127_mont(commitment: *mut c_void);
    fn qa_gpu_encode_store_m127_mont(
        context: *mut c_void,
        messages: *const M127Mont,
        rows: u32,
        commitment: *mut c_void,
        row_offset: usize,
        timing: *mut RawGpuTiming,
    ) -> c_int;
    fn qa_gpu_hash_columns_blake2b256_m127_mont(
        commitment: *mut c_void,
        host_digests: *mut u8,
        timing: *mut RawGpuMerkleTiming,
    ) -> c_int;
    fn qa_gpu_read_column_m127_mont(
        commitment: *mut c_void,
        column: usize,
        host_output: *mut M127Mont,
    ) -> c_int;
    fn qa_gpu_last_error_m127_mont() -> *const c_char;
}

fn last_error() -> String {
    unsafe {
        let ptr = qa_gpu_last_error_m127_mont();
        if ptr.is_null() {
            "unknown CUDA error".to_owned()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }
}

/// Checks the assumptions that make the zero-conversion FFI path sound.
///
/// `Mersenne127` must be declared as
/// `#[repr(transparent)] pub struct Mersenne127([u64; 2]);` in
/// `plonkish_backend/src/util/new_fields.rs`. The derive stores elements in
/// Montgomery form with R = 2^128 mod (2^127 - 1) = 2, so raw ONE is [2, 0].
fn validate_montgomery_layout() -> Result<(), String> {
    if size_of::<Mersenne127>() != size_of::<M127Mont>() {
        return Err(format!(
            "Mersenne127 size is {}, expected {} bytes for two Montgomery limbs",
            size_of::<Mersenne127>(),
            size_of::<M127Mont>()
        ));
    }
    if align_of::<Mersenne127>() != align_of::<M127Mont>() {
        return Err(format!(
            "Mersenne127 alignment is {}, expected {}; add #[repr(transparent)] to its definition",
            align_of::<Mersenne127>(),
            align_of::<M127Mont>()
        ));
    }

    // This read is layout-safe under the required repr(transparent)
    // declaration. It checks both limb order and the Montgomery radix.
    let zero = unsafe { *(&Mersenne127::ZERO as *const Mersenne127).cast::<M127Mont>() };
    let one = unsafe { *(&Mersenne127::ONE as *const Mersenne127).cast::<M127Mont>() };
    if zero != (M127Mont { lo: 0, hi: 0 }) {
        return Err(format!("unexpected raw Montgomery ZERO: {zero:?}"));
    }
    if one != (M127Mont { lo: 2, hi: 0 }) {
        return Err(format!(
            "unexpected raw Montgomery ONE: {one:?}; expected R = 2 for p = 2^127 - 1"
        ));
    }
    Ok(())
}

#[inline]
fn mont_ptr(values: &[Mersenne127]) -> *const M127Mont {
    values.as_ptr().cast::<M127Mont>()
}

/// Converts a fully initialized `Vec<MaybeUninit<T>>` without moving or
/// touching any element.
unsafe fn assume_init_vec<T>(values: Vec<MaybeUninit<T>>) -> Vec<T> {
    let mut values = ManuallyDrop::new(values);
    Vec::from_raw_parts(
        values.as_mut_ptr().cast::<T>(),
        values.len(),
        values.capacity(),
    )
}

/// Reusable, pre-faulted, CUDA-registered host storage for complete QA
/// codewords. The allocation never moves while it is registered, so each
/// batch can copy directly into its final row-major location.
pub struct GpuQaOutput {
    values: Vec<Mersenne127>,
    rows: usize,
    row_len: usize,
    codeword_len: usize,
    registered: bool,
    setup_timing: GpuQaOutputSetupTiming,
}

/// Complete encoded QA matrix retained in device memory. Only 32-byte leaf
/// digests and Fiat--Shamir-selected columns cross back to the host.
pub struct GpuQaDeviceOutput {
    commitment: NonNull<c_void>,
    rows: usize,
    columns: usize,
    leaf_digests: Vec<u8>,
    leaf_digests_registered: bool,
    query_scratch: Mutex<Vec<Mersenne127>>,
    query_scratch_registered: bool,
    setup_timing: GpuQaDeviceOutputSetupTiming,
    query_transfer_ns: AtomicU64,
    query_transfer_count: AtomicU64,
}

unsafe impl Send for GpuQaDeviceOutput {}
unsafe impl Sync for GpuQaDeviceOutput {}

/// Reusable CUDA-registered host storage for QA messages. The caller fills
/// the vector before registration, so every page is already resident. Keeping
/// the allocation registered across all encoder calls removes CUDA's
/// per-transfer pageable-memory staging from the H2D path.
pub struct GpuQaInput {
    values: Vec<Mersenne127>,
    registered: bool,
    setup_timing: GpuQaInputSetupTiming,
}

impl GpuQaInput {
    fn new(mut values: Vec<Mersenne127>) -> Result<Self, String> {
        if values.is_empty() {
            return Ok(Self {
                values,
                registered: false,
                setup_timing: GpuQaInputSetupTiming::default(),
            });
        }

        let register_start = Instant::now();
        let status = unsafe {
            qa_gpu_host_register_m127_mont(
                values.as_mut_ptr().cast::<M127Mont>(),
                values.len(),
            )
        };
        let pin_registration = register_start.elapsed();
        if status != 0 {
            return Err(format!(
                "failed to register the reusable host input as pinned memory: {}",
                last_error()
            ));
        }

        Ok(Self {
            values,
            registered: true,
            setup_timing: GpuQaInputSetupTiming { pin_registration },
        })
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn as_slice(&self) -> &[Mersenne127] {
        &self.values
    }

    pub fn setup_timing(&self) -> &GpuQaInputSetupTiming {
        &self.setup_timing
    }
}

impl Drop for GpuQaInput {
    fn drop(&mut self) {
        if self.registered {
            let _ = unsafe {
                qa_gpu_host_unregister_m127_mont(self.values.as_mut_ptr().cast::<M127Mont>())
            };
            self.registered = false;
        }
    }
}

impl GpuQaOutput {
    fn new(rows: usize, row_len: usize, inverse_rate: usize) -> Result<Self, String> {
        let codeword_len = row_len
            .checked_mul(inverse_rate)
            .ok_or_else(|| "GPU codeword row length overflows usize".to_owned())?;
        let elements = rows
            .checked_mul(codeword_len)
            .ok_or_else(|| "GPU output length overflows usize".to_owned())?;
        let total_start = Instant::now();
        if elements == 0 {
            return Ok(Self {
                values: Vec::new(),
                rows,
                row_len,
                codeword_len,
                registered: false,
                setup_timing: GpuQaOutputSetupTiming::default(),
            });
        }

        // Reserve virtual address space first, then initialize in parallel.
        // Writing every byte is deliberate: it commits all physical pages
        // before cudaHostRegister and keeps page faults out of the D2H timer.
        let allocation_start = Instant::now();
        let mut uninitialized = Vec::<MaybeUninit<Mersenne127>>::with_capacity(elements);
        unsafe { uninitialized.set_len(elements) };
        let allocation = allocation_start.elapsed();

        let prefault_start = Instant::now();
        uninitialized
            .par_chunks_mut(1 << 18)
            .for_each(|chunk| unsafe {
                // Mersenne127::ZERO is represented by two zero Montgomery
                // limbs, which validate_montgomery_layout checks at startup.
                std::ptr::write_bytes(chunk.as_mut_ptr(), 0, chunk.len());
            });
        let prefault_and_initialize = prefault_start.elapsed();
        let mut values = unsafe { assume_init_vec(uninitialized) };

        let register_start = Instant::now();
        let status = unsafe {
            qa_gpu_host_register_m127_mont(
                values.as_mut_ptr().cast::<M127Mont>(),
                values.len(),
            )
        };
        let pin_registration = register_start.elapsed();
        if status != 0 {
            return Err(format!(
                "failed to register the reusable host output as pinned memory: {}",
                last_error()
            ));
        }

        Ok(Self {
            values,
            rows,
            row_len,
            codeword_len,
            registered: true,
            setup_timing: GpuQaOutputSetupTiming {
                allocation,
                prefault_and_initialize,
                pin_registration,
                total: total_start.elapsed(),
            },
        })
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn as_slice(&self) -> &[Mersenne127] {
        &self.values
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn row_len(&self) -> usize {
        self.row_len
    }

    pub fn codeword_len(&self) -> usize {
        self.codeword_len
    }

    pub fn setup_timing(&self) -> &GpuQaOutputSetupTiming {
        &self.setup_timing
    }

    fn as_mut_mont_ptr(&mut self) -> *mut M127Mont {
        self.values.as_mut_ptr().cast::<M127Mont>()
    }
}

impl QACodewordRows<Mersenne127> for GpuQaOutput {
    fn num_rows(&self) -> usize {
        self.rows
    }

    fn num_cols(&self) -> usize {
        self.codeword_len
    }

    fn row(&self, row_index: usize) -> &[Mersenne127] {
        assert!(row_index < self.rows, "GPU QA row index out of bounds");
        let start = row_index * self.codeword_len;
        &self.values[start..start + self.codeword_len]
    }
}

impl QACodewordColumns<Mersenne127> for GpuQaOutput {
    fn row_count(&self) -> usize {
        self.rows
    }

    fn column_count(&self) -> usize {
        self.codeword_len
    }

    fn read_column(&self, column_index: usize) -> Result<Vec<Mersenne127>, String> {
        if column_index >= self.codeword_len {
            return Err("GPU QA host column index is out of bounds".to_owned());
        }
        Ok((0..self.rows)
            .map(|row| self.values[row * self.codeword_len + column_index])
            .collect())
    }
}

impl Drop for GpuQaOutput {
    fn drop(&mut self) {
        if self.registered {
            // All encoder calls synchronize before returning, so no D2H can
            // still be using the allocation here.
            let _ = unsafe {
                qa_gpu_host_unregister_m127_mont(self.values.as_mut_ptr().cast::<M127Mont>())
            };
            self.registered = false;
        }
    }
}

impl GpuQaDeviceOutput {
    fn new(rows: usize, columns: usize) -> Result<Self, String> {
        let total_start = Instant::now();
        let device_start = Instant::now();
        let commitment = unsafe {
            qa_gpu_create_device_commitment_m127_mont(rows, columns)
        };
        let device_allocation = device_start.elapsed();
        let commitment = NonNull::new(commitment).ok_or_else(last_error)?;

        let host_start = Instant::now();
        let digest_bytes = columns
            .checked_mul(32)
            .ok_or_else(|| "GPU leaf digest length overflows usize".to_owned())?;
        let mut leaf_digests = vec![0u8; digest_bytes];
        let mut query_scratch = vec![Mersenne127::ZERO; rows];
        let digest_allocation_and_prefault = host_start.elapsed();

        let pin_start = Instant::now();
        let leaf_status = unsafe {
            qa_gpu_host_register_bytes(
                leaf_digests.as_mut_ptr().cast::<c_void>(),
                leaf_digests.len(),
            )
        };
        if leaf_status != 0 {
            unsafe { qa_gpu_destroy_device_commitment_m127_mont(commitment.as_ptr()) };
            return Err(format!(
                "failed to pin GPU leaf-digest output: {}",
                last_error()
            ));
        }
        let query_status = unsafe {
            qa_gpu_host_register_m127_mont(
                query_scratch.as_mut_ptr().cast::<M127Mont>(),
                query_scratch.len(),
            )
        };
        if query_status != 0 {
            let _ = unsafe {
                qa_gpu_host_unregister_bytes(leaf_digests.as_mut_ptr().cast::<c_void>())
            };
            unsafe { qa_gpu_destroy_device_commitment_m127_mont(commitment.as_ptr()) };
            return Err(format!(
                "failed to pin queried-column output: {}",
                last_error()
            ));
        }
        let pin_registration = pin_start.elapsed();

        Ok(Self {
            commitment,
            rows,
            columns,
            leaf_digests,
            leaf_digests_registered: true,
            query_scratch: Mutex::new(query_scratch),
            query_scratch_registered: true,
            setup_timing: GpuQaDeviceOutputSetupTiming {
                device_allocation,
                digest_allocation_and_prefault,
                pin_registration,
                total: total_start.elapsed(),
            },
            query_transfer_ns: AtomicU64::new(0),
            query_transfer_count: AtomicU64::new(0),
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn leaf_digest_bytes(&self) -> &[u8] {
        &self.leaf_digests
    }

    pub fn setup_timing(&self) -> &GpuQaDeviceOutputSetupTiming {
        &self.setup_timing
    }

    pub fn query_transfer_stats(&self) -> (u64, Duration) {
        (
            self.query_transfer_count.swap(0, Ordering::Relaxed),
            Duration::from_nanos(self.query_transfer_ns.swap(0, Ordering::Relaxed)),
        )
    }

    fn raw_handle(&self) -> *mut c_void {
        self.commitment.as_ptr()
    }
}

impl QACodewordColumns<Mersenne127> for GpuQaDeviceOutput {
    fn row_count(&self) -> usize {
        self.rows
    }

    fn column_count(&self) -> usize {
        self.columns
    }

    fn read_column(&self, column_index: usize) -> Result<Vec<Mersenne127>, String> {
        if column_index >= self.columns {
            return Err("GPU QA column index is out of bounds".to_owned());
        }
        let mut scratch = self
            .query_scratch
            .lock()
            .map_err(|_| "GPU queried-column scratch lock is poisoned".to_owned())?;
        let start = Instant::now();
        let status = unsafe {
            qa_gpu_read_column_m127_mont(
                self.raw_handle(),
                column_index,
                scratch.as_mut_ptr().cast::<M127Mont>(),
            )
        };
        let elapsed = start.elapsed();
        if status != 0 {
            return Err(last_error());
        }
        self.query_transfer_ns.fetch_add(
            elapsed.as_nanos().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
        self.query_transfer_count.fetch_add(1, Ordering::Relaxed);
        Ok(scratch.clone())
    }
}

impl Drop for GpuQaDeviceOutput {
    fn drop(&mut self) {
        if self.leaf_digests_registered {
            let _ = unsafe {
                qa_gpu_host_unregister_bytes(self.leaf_digests.as_mut_ptr().cast::<c_void>())
            };
            self.leaf_digests_registered = false;
        }
        if self.query_scratch_registered {
            let scratch = self
                .query_scratch
                .get_mut()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let _ = unsafe {
                qa_gpu_host_unregister_m127_mont(scratch.as_mut_ptr().cast::<M127Mont>())
            };
            self.query_scratch_registered = false;
        }
        unsafe { qa_gpu_destroy_device_commitment_m127_mont(self.commitment.as_ptr()) };
    }
}

pub struct GpuQaEncoder {
    context: NonNull<c_void>,
    row_len: usize,
    inverse_rate: usize,
    max_batch_rows: usize,
}

unsafe impl Send for GpuQaEncoder {}

impl GpuQaEncoder {
    /// Uploads the public QA coefficient vectors in their existing Montgomery
    /// representation and allocates reusable device buffers.
    pub fn new(
        params: &QAParams<Mersenne127>,
        max_batch_rows: usize,
    ) -> Result<Self, String> {
        validate_montgomery_layout()?;
        if params.inverse_rate < 2 || !params.inverse_rate.is_power_of_two() {
            return Err("inverse_rate must be a power of two and at least two".to_owned());
        }
        if max_batch_rows == 0 || max_batch_rows > u32::MAX as usize {
            return Err("max_batch_rows is outside the CUDA ABI range".to_owned());
        }
        let row_len = params.e.first().map_or(0, Vec::len);
        if row_len == 0
            || !row_len.is_power_of_two()
            || params.e.iter().any(|e| e.len() != row_len)
        {
            return Err(
                "all QA coefficient vectors must have the same power-of-two length".to_owned(),
            );
        }

        // This is only a contiguous byte-for-byte copy. There is no
        // to_repr/from_repr or Montgomery/canonical conversion.
        let montgomery_coefficients = params
            .e
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<Mersenne127>>();
        let context = unsafe {
            qa_gpu_create_m127_mont(
                mont_ptr(&montgomery_coefficients),
                row_len,
                params.inverse_rate as u32,
                max_batch_rows as u32,
            )
        };
        let context = NonNull::new(context).ok_or_else(last_error)?;
        Ok(Self {
            context,
            row_len,
            inverse_rate: params.inverse_rate,
            max_batch_rows,
        })
    }

    pub fn device_name(&self) -> Result<String, String> {
        let mut bytes = vec![0i8; 256];
        let result = unsafe {
            qa_gpu_device_name_m127_mont(
                self.context.as_ptr(),
                bytes.as_mut_ptr(),
                bytes.len(),
            )
        };
        if result != 0 {
            return Err(last_error());
        }
        Ok(unsafe { CStr::from_ptr(bytes.as_ptr()) }
            .to_string_lossy()
            .into_owned())
    }

    /// Allocates, pre-faults, and pins the complete host output once. Reuse
    /// the returned buffer across repetitions or commitments of this shape.
    pub fn allocate_output(&self, rows: usize) -> Result<GpuQaOutput, String> {
        GpuQaOutput::new(rows, self.row_len, self.inverse_rate)
    }

    /// Allocates a reusable complete codeword in device memory plus pinned
    /// host buffers for leaf digests and queried columns.
    pub fn allocate_device_output(
        &self,
        rows: usize,
    ) -> Result<GpuQaDeviceOutput, String> {
        GpuQaDeviceOutput::new(rows, self.row_len * self.inverse_rate)
    }

    /// Registers an already initialized message vector once and retains its
    /// stable allocation for all subsequent H2D transfers.
    pub fn register_input(&self, messages: Vec<Mersenne127>) -> Result<GpuQaInput, String> {
        if messages.is_empty() || messages.len() % self.row_len != 0 {
            return Err("message length is not a non-zero multiple of the QA row length".to_owned());
        }
        GpuQaInput::new(messages)
    }

    /// Encodes row-major messages by transferring the backend's raw
    /// Montgomery limbs directly between reusable pinned host buffers. No
    /// field representation conversion or per-call host allocation occurs.
    pub fn encode_rows_into(
        &mut self,
        messages: &GpuQaInput,
        result: &mut GpuQaOutput,
    ) -> Result<GpuQaTiming, String> {
        let messages = messages.as_slice();
        if messages.len() % self.row_len != 0 {
            return Err("message length is not a multiple of the QA row length".to_owned());
        }
        let rows = messages.len() / self.row_len;
        if result.rows != rows
            || result.row_len != self.row_len
            || result.codeword_len != self.inverse_rate * self.row_len
        {
            return Err("GPU output shape does not match the encoder input".to_owned());
        }
        if rows == 0 {
            if !result.is_empty() {
                return Err("non-empty GPU output supplied for zero message rows".to_owned());
            }
            return Ok(GpuQaTiming::default());
        }
        let output_elements = rows
            .checked_mul(self.inverse_rate)
            .and_then(|value| value.checked_mul(self.row_len))
            .ok_or_else(|| "GPU output length overflows usize".to_owned())?;
        if result.len() != output_elements {
            return Err(format!(
                "GPU output has {} elements, expected {output_elements}",
                result.len()
            ));
        }

        let total_start = Instant::now();
        let mut timing = GpuQaTiming::default();

        for row_start in (0..rows).step_by(self.max_batch_rows) {
            let batch_rows = (rows - row_start).min(self.max_batch_rows);
            let input =
                &messages[row_start * self.row_len..(row_start + batch_rows) * self.row_len];
            let output_offset = row_start * self.inverse_rate * self.row_len;
            let output_ptr = unsafe { result.as_mut_mont_ptr().add(output_offset) };

            let mut raw = MaybeUninit::<RawGpuTiming>::zeroed();
            let start = Instant::now();
            let status = unsafe {
                qa_gpu_encode_m127_mont(
                    self.context.as_ptr(),
                    mont_ptr(input),
                    batch_rows as u32,
                    output_ptr,
                    raw.as_mut_ptr(),
                )
            };
            timing.ffi_wall += start.elapsed();
            if status != 0 {
                return Err(last_error());
            }
            timing.add_raw(unsafe { raw.assume_init() });
        }

        timing.total_wall = total_start.elapsed();
        Ok(timing)
    }

    /// Encodes all rows into persistent device storage, hashes every encoded
    /// column with the backend-compatible BLAKE2b-256 on the GPU, and copies
    /// only the leaf digests back.
    pub fn encode_rows_to_device(
        &mut self,
        messages: &GpuQaInput,
        result: &mut GpuQaDeviceOutput,
    ) -> Result<GpuQaTiming, String> {
        let messages = messages.as_slice();
        if messages.len() % self.row_len != 0 {
            return Err("message length is not a multiple of the QA row length".to_owned());
        }
        let rows = messages.len() / self.row_len;
        if rows == 0
            || result.rows != rows
            || result.columns != self.inverse_rate * self.row_len
        {
            return Err("GPU device output shape does not match the encoder input".to_owned());
        }

        let total_start = Instant::now();
        let mut timing = GpuQaTiming::default();
        for row_start in (0..rows).step_by(self.max_batch_rows) {
            let batch_rows = (rows - row_start).min(self.max_batch_rows);
            let input =
                &messages[row_start * self.row_len..(row_start + batch_rows) * self.row_len];
            let mut raw = MaybeUninit::<RawGpuTiming>::zeroed();
            let start = Instant::now();
            let status = unsafe {
                qa_gpu_encode_store_m127_mont(
                    self.context.as_ptr(),
                    mont_ptr(input),
                    batch_rows as u32,
                    result.raw_handle(),
                    row_start,
                    raw.as_mut_ptr(),
                )
            };
            timing.ffi_wall += start.elapsed();
            if status != 0 {
                return Err(last_error());
            }
            timing.add_raw(unsafe { raw.assume_init() });
        }

        let mut raw_merkle = MaybeUninit::<RawGpuMerkleTiming>::zeroed();
        let start = Instant::now();
        let status = unsafe {
            qa_gpu_hash_columns_blake2b256_m127_mont(
                result.raw_handle(),
                result.leaf_digests.as_mut_ptr(),
                raw_merkle.as_mut_ptr(),
            )
        };
        timing.ffi_wall += start.elapsed();
        if status != 0 {
            return Err(last_error());
        }
        timing.add_merkle_raw(unsafe { raw_merkle.assume_init() });
        timing.total_wall = total_start.elapsed();
        Ok(timing)
    }
}

impl Drop for GpuQaEncoder {
    fn drop(&mut self) {
        unsafe { qa_gpu_destroy_m127_mont(self.context.as_ptr()) }
    }
}
