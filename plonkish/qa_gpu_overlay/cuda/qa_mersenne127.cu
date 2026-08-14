#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>

namespace {

constexpr uint64_t MASK63 = 0x7fffffffffffffffULL;
constexpr unsigned TILE = 1024;
constexpr unsigned THREADS = 256;

// Raw little-endian Montgomery limbs of
// plonkish_backend::util::new_fields::Mersenne127.
// p = 2^127 - 1 and R = 2^128 mod p = 2, so x is stored as xR mod p.
struct M127Mont {
    uint64_t lo;
    uint64_t hi;
};

static_assert(sizeof(M127Mont) == 16, "M127Mont must contain exactly two u64 limbs");
static_assert(alignof(M127Mont) == alignof(uint64_t), "Rust/CUDA limb alignment mismatch");

struct RawGpuTiming {
    float host_to_device_ms;
    float device_input_copy_ms;
    float first_wht_ms;
    float scaling_ms;
    float second_wht_ms;
    float assemble_ms;
    float device_to_host_ms;
    float total_cuda_ms;
};

struct RawGpuMerkleTiming {
    float column_hash_ms;
    float digest_device_to_host_ms;
    float total_cuda_ms;
};

thread_local char last_error[1024] = "";

void set_error(const char* where, cudaError_t error) {
    std::snprintf(last_error, sizeof(last_error), "%s: %s", where, cudaGetErrorString(error));
}

void set_error_text(const char* text) {
    std::snprintf(last_error, sizeof(last_error), "%s", text);
}

#define CUDA_TRY(expression)                       \
    do {                                           \
        cudaError_t status_ = (expression);        \
        if (status_ != cudaSuccess) {              \
            set_error(#expression, status_);       \
            return false;                          \
        }                                          \
    } while (false)

__device__ __forceinline__ M127Mont canonicalize_sum(uint64_t lo, uint64_t hi) {
    const uint64_t top = hi >> 63;
    hi &= MASK63;
    const uint64_t old_lo = lo;
    lo += top;
    hi += (lo < old_lo);
    if (lo == UINT64_MAX && hi == MASK63) return M127Mont{0, 0};
    return M127Mont{lo, hi};
}

__device__ __forceinline__ M127Mont add_mod(M127Mont a, M127Mont b) {
    const uint64_t lo = a.lo + b.lo;
    const uint64_t carry = lo < a.lo;
    return canonicalize_sum(lo, a.hi + b.hi + carry);
}

__device__ __forceinline__ M127Mont sub_mod(M127Mont a, M127Mont b) {
    uint64_t lo = a.lo - b.lo;
    const uint64_t borrow0 = a.lo < b.lo;
    const uint64_t hi_sub = b.hi + borrow0;
    const uint64_t borrow1 = (a.hi < hi_sub) || (borrow0 && hi_sub == 0);
    uint64_t hi = a.hi - hi_sub;
    if (borrow1) {
        const uint64_t old_lo = lo;
        lo += UINT64_MAX;
        hi += MASK63 + (lo < old_lo);
    }
    return M127Mont{lo, hi};
}

__device__ __forceinline__ uint64_t add_limb(uint64_t& target, uint64_t value) {
    const uint64_t old = target;
    target += value;
    return target < old;
}

// Ordinary multiplication modulo p of the 127-bit integers in a and b.
__device__ __forceinline__ M127Mont mul_mod_p(M127Mont a, M127Mont b) {
    const uint64_t p00_lo = a.lo * b.lo;
    const uint64_t p00_hi = __umul64hi(a.lo, b.lo);
    const uint64_t p01_lo = a.lo * b.hi;
    const uint64_t p01_hi = __umul64hi(a.lo, b.hi);
    const uint64_t p10_lo = a.hi * b.lo;
    const uint64_t p10_hi = __umul64hi(a.hi, b.lo);
    const uint64_t p11_lo = a.hi * b.hi;
    uint64_t r3 = __umul64hi(a.hi, b.hi);

    const uint64_t r0 = p00_lo;
    uint64_t r1 = p00_hi;
    const uint64_t carry01 = add_limb(r1, p01_lo);
    const uint64_t carry10 = add_limb(r1, p10_lo);

    uint64_t r2 = p01_hi;
    r3 += add_limb(r2, p10_hi);
    r3 += add_limb(r2, p11_lo);
    r3 += add_limb(r2, carry01);
    r3 += add_limb(r2, carry10);

    // product = low + 2^127 high == low + high (mod p).
    const M127Mont low{r0, r1 & MASK63};
    const M127Mont high{(r1 >> 63) | (r2 << 1), (r2 >> 63) | (r3 << 1)};
    return add_mod(low, high);
}

__device__ __forceinline__ M127Mont divide_by_two_mod_p(M127Mont value) {
    // 2^{-1} modulo 2^127-1 is a cyclic right rotation of 127 bits.
    return M127Mont{
        (value.lo >> 1) | (value.hi << 63),
        (value.hi >> 1) | ((value.lo & 1) << 62),
    };
}

__device__ __forceinline__ M127Mont montgomery_mul(M127Mont a, M127Mont b) {
    // (xR)(yR)R^{-1} = xyR, with R = 2.
    return divide_by_two_mod_p(mul_mod_p(a, b));
}

__global__ void wht_initial(M127Mont* data, size_t n, size_t vectors) {
    extern __shared__ M127Mont shared[];
    const size_t tiles_per_vector = n / TILE;
    const size_t vector = blockIdx.x / tiles_per_vector;
    const size_t tile = blockIdx.x % tiles_per_vector;
    if (vector >= vectors) return;
    const size_t base = vector * n + tile * TILE;
    for (unsigned i = threadIdx.x; i < TILE; i += blockDim.x) shared[i] = data[base + i];
    __syncthreads();
    for (unsigned step = 1; step < TILE; step <<= 1) {
        for (unsigned butterfly = threadIdx.x; butterfly < TILE / 2; butterfly += blockDim.x) {
            const unsigned group = butterfly / step;
            const unsigned j = butterfly - group * step;
            const unsigned left = group * (2 * step) + j;
            const unsigned right = left + step;
            const M127Mont u = shared[left];
            const M127Mont v = shared[right];
            shared[left] = add_mod(u, v);
            shared[right] = sub_mod(u, v);
        }
        __syncthreads();
    }
    for (unsigned i = threadIdx.x; i < TILE; i += blockDim.x) data[base + i] = shared[i];
}

__global__ void wht_initial_small(M127Mont* data, size_t n, size_t vectors) {
    extern __shared__ M127Mont shared[];
    const size_t vector = blockIdx.x;
    if (vector >= vectors) return;
    const size_t base = vector * n;
    for (size_t i = threadIdx.x; i < n; i += blockDim.x) shared[i] = data[base + i];
    __syncthreads();
    for (size_t step = 1; step < n; step <<= 1) {
        for (size_t butterfly = threadIdx.x; butterfly < n / 2; butterfly += blockDim.x) {
            const size_t group = butterfly / step;
            const size_t j = butterfly - group * step;
            const size_t left = group * (2 * step) + j;
            const size_t right = left + step;
            const M127Mont u = shared[left];
            const M127Mont v = shared[right];
            shared[left] = add_mod(u, v);
            shared[right] = sub_mod(u, v);
        }
        __syncthreads();
    }
    for (size_t i = threadIdx.x; i < n; i += blockDim.x) data[base + i] = shared[i];
}

__global__ void wht_layer(M127Mont* data, size_t n, size_t vectors, size_t step) {
    const size_t global = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t butterflies_per_vector = n / 2;
    const size_t total = vectors * butterflies_per_vector;
    if (global >= total) return;
    const size_t vector = global / butterflies_per_vector;
    const size_t butterfly = global - vector * butterflies_per_vector;
    const size_t group = butterfly / step;
    const size_t j = butterfly - group * step;
    const size_t left = vector * n + group * (2 * step) + j;
    const size_t right = left + step;
    const M127Mont u = data[left];
    const M127Mont v = data[right];
    data[left] = add_mod(u, v);
    data[right] = sub_mod(u, v);
}

__global__ void scale_blocks(
    const M127Mont* middle,
    const M127Mont* coefficients,
    M127Mont* parity,
    size_t n,
    size_t rows,
    size_t parity_blocks) {
    const size_t global = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = rows * parity_blocks * n;
    if (global >= total) return;
    const size_t j = global % n;
    const size_t vector = global / n;
    const size_t block = vector % parity_blocks;
    const size_t row = vector / parity_blocks;
    parity[global] = montgomery_mul(middle[row * n + j], coefficients[block * n + j]);
}

__global__ void assemble_codeword(
    const M127Mont* messages,
    const M127Mont* parity,
    M127Mont* codewords,
    size_t n,
    size_t rows,
    size_t inverse_rate) {
    const size_t global = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = rows * inverse_rate * n;
    if (global >= total) return;
    const size_t j = global % n;
    const size_t vector = global / n;
    const size_t block = vector % inverse_rate;
    const size_t row = vector / inverse_rate;
    codewords[global] = block == 0
        ? messages[row * n + j]
        : parity[(row * (inverse_rate - 1) + block - 1) * n + j];
}

// BLAKE2b-256, specialized for one column of canonical little-endian
// Mersenne127 field representations.
//
// Despite its name, plonkish_backend::util::hash::Blake2s wraps
// blake2b_simd::State with a 32-byte output.  The CPU Merkle path therefore
// uses BLAKE2b-256, not RustCrypto's BLAKE2s-256.  Rust's
// update_field_element feeds PrimeField::to_repr() bytes, so Montgomery limbs
// must also be divided by R=2 before hashing.
__device__ __constant__ uint8_t BLAKE2B_SIGMA[12][16] = {
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15},
    {14,10, 4, 8, 9,15,13, 6, 1,12, 0, 2,11, 7, 5, 3},
    {11, 8,12, 0, 5, 2,15,13,10,14, 3, 6, 7, 1, 9, 4},
    { 7, 9, 3, 1,13,12,11,14, 2, 6, 5,10, 4, 0,15, 8},
    { 9, 0, 5, 7, 2, 4,10,15,14, 1,11,12, 6, 8, 3,13},
    { 2,12, 6,10, 0,11, 8, 3, 4,13, 7, 5,15,14, 1, 9},
    {12, 5, 1,15,14,13, 4,10, 0, 7, 6, 3, 9, 2, 8,11},
    {13,11, 7,14,12, 1, 3, 9, 5, 0,15, 4, 8, 6, 2,10},
    { 6,15,14, 9,11, 3, 0, 8,12, 2,13, 7, 1, 4,10, 5},
    {10, 2, 8, 4, 7, 6, 1, 5,15,11, 9,14, 3,12,13, 0},
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15},
    {14,10, 4, 8, 9,15,13, 6, 1,12, 0, 2,11, 7, 5, 3},
};

__device__ __forceinline__ uint64_t rotr64(uint64_t x, unsigned n) {
    return (x >> n) | (x << (64 - n));
}

__device__ __forceinline__ void blake2b_g(
    uint64_t& a, uint64_t& b, uint64_t& c, uint64_t& d,
    uint64_t x, uint64_t y) {
    a = a + b + x;
    d = rotr64(d ^ a, 32);
    c += d;
    b = rotr64(b ^ c, 24);
    a = a + b + y;
    d = rotr64(d ^ a, 16);
    c += d;
    b = rotr64(b ^ c, 63);
}

__device__ __forceinline__ void blake2b_compress(
    uint64_t h[8], const uint64_t m[16], uint64_t bytes, bool last) {
    const uint64_t iv[8] = {
        0x6A09E667F3BCC908ULL, 0xBB67AE8584CAA73BULL,
        0x3C6EF372FE94F82BULL, 0xA54FF53A5F1D36F1ULL,
        0x510E527FADE682D1ULL, 0x9B05688C2B3E6C1FULL,
        0x1F83D9ABFB41BD6BULL, 0x5BE0CD19137E2179ULL,
    };
    uint64_t v[16];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        v[i] = h[i];
        v[i + 8] = iv[i];
    }
    v[12] ^= bytes;
    if (last) v[14] ^= 0xffffffffffffffffULL;

#pragma unroll
    for (int round = 0; round < 12; ++round) {
        const uint8_t* s = BLAKE2B_SIGMA[round];
        blake2b_g(v[0], v[4], v[ 8], v[12], m[s[ 0]], m[s[ 1]]);
        blake2b_g(v[1], v[5], v[ 9], v[13], m[s[ 2]], m[s[ 3]]);
        blake2b_g(v[2], v[6], v[10], v[14], m[s[ 4]], m[s[ 5]]);
        blake2b_g(v[3], v[7], v[11], v[15], m[s[ 6]], m[s[ 7]]);
        blake2b_g(v[0], v[5], v[10], v[15], m[s[ 8]], m[s[ 9]]);
        blake2b_g(v[1], v[6], v[11], v[12], m[s[10]], m[s[11]]);
        blake2b_g(v[2], v[7], v[ 8], v[13], m[s[12]], m[s[13]]);
        blake2b_g(v[3], v[4], v[ 9], v[14], m[s[14]], m[s[15]]);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) h[i] ^= v[i] ^ v[i + 8];
}

__global__ void hash_columns_blake2b256(
    const M127Mont* codewords,
    uint8_t* digests,
    size_t rows,
    size_t columns) {
    const size_t column = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (column >= columns) return;

    const uint64_t iv[8] = {
        0x6A09E667F3BCC908ULL, 0xBB67AE8584CAA73BULL,
        0x3C6EF372FE94F82BULL, 0xA54FF53A5F1D36F1ULL,
        0x510E527FADE682D1ULL, 0x9B05688C2B3E6C1FULL,
        0x1F83D9ABFB41BD6BULL, 0x5BE0CD19137E2179ULL,
    };
    uint64_t h[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) h[i] = iv[i];
    h[0] ^= 0x01010020ULL;  // fanout=1, depth=1, digest length=32

    const size_t total_bytes = rows * sizeof(M127Mont);
    const size_t message_blocks = (total_bytes + 127) / 128;
    for (size_t message_block = 0; message_block < message_blocks; ++message_block) {
        uint64_t m[16] = {};
        const size_t first_row = message_block * 8;
#pragma unroll
        for (int lane = 0; lane < 8; ++lane) {
            const size_t row = first_row + size_t(lane);
            if (row < rows) {
                const M127Mont canonical = divide_by_two_mod_p(codewords[row * columns + column]);
                m[2 * lane    ] = canonical.lo;
                m[2 * lane + 1] = canonical.hi;
            }
        }
        const size_t processed = (message_block + 1) * 128;
        const bool last = processed >= total_bytes;
        blake2b_compress(h, m, uint64_t(last ? total_bytes : processed), last);
    }

    uint64_t* output = reinterpret_cast<uint64_t*>(digests + column * 32);
#pragma unroll
    for (int i = 0; i < 4; ++i) output[i] = h[i];
}

__global__ void gather_column(
    const M127Mont* codewords,
    M127Mont* output,
    size_t rows,
    size_t columns,
    size_t column) {
    const size_t row = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row < rows) output[row] = codewords[row * columns + column];
}

struct Context {
    size_t row_len = 0;
    size_t inverse_rate = 0;
    size_t max_batch_rows = 0;
    M127Mont* coefficients = nullptr;
    M127Mont* messages = nullptr;
    M127Mont* middle = nullptr;
    M127Mont* parity = nullptr;
    M127Mont* codewords = nullptr;
    cudaEvent_t events[8]{};
    char device_name[256]{};
};

struct DeviceCommitment {
    size_t rows = 0;
    size_t columns = 0;
    M127Mont* codewords = nullptr;
    uint8_t* leaf_digests = nullptr;
    M127Mont* query_column = nullptr;
    cudaEvent_t events[3]{};
};

bool event_elapsed(float* result, cudaEvent_t begin, cudaEvent_t end) {
    CUDA_TRY(cudaEventElapsedTime(result, begin, end));
    return true;
}

bool launch_wht(M127Mont* data, size_t n, size_t vectors) {
    if (n <= TILE) {
        wht_initial_small<<<vectors, THREADS, n * sizeof(M127Mont)>>>(data, n, vectors);
        CUDA_TRY(cudaGetLastError());
        return true;
    }
    const size_t tiles = vectors * (n / TILE);
    wht_initial<<<tiles, THREADS, TILE * sizeof(M127Mont)>>>(data, n, vectors);
    CUDA_TRY(cudaGetLastError());
    for (size_t step = TILE; step < n; step <<= 1) {
        const size_t butterflies = vectors * n / 2;
        const size_t blocks = (butterflies + THREADS - 1) / THREADS;
        wht_layer<<<blocks, THREADS>>>(data, n, vectors, step);
        CUDA_TRY(cudaGetLastError());
    }
    return true;
}

void free_context(Context* context) {
    if (!context) return;
    for (auto& event : context->events) if (event) cudaEventDestroy(event);
    if (context->coefficients) cudaFree(context->coefficients);
    if (context->messages) cudaFree(context->messages);
    if (context->middle) cudaFree(context->middle);
    if (context->parity) cudaFree(context->parity);
    if (context->codewords) cudaFree(context->codewords);
    delete context;
}

void free_device_commitment(DeviceCommitment* commitment) {
    if (!commitment) return;
    for (auto& event : commitment->events) if (event) cudaEventDestroy(event);
    if (commitment->codewords) cudaFree(commitment->codewords);
    if (commitment->leaf_digests) cudaFree(commitment->leaf_digests);
    if (commitment->query_column) cudaFree(commitment->query_column);
    delete commitment;
}

}  // namespace

extern "C" const char* qa_gpu_last_error_m127_mont() {
    return last_error;
}

extern "C" void* qa_gpu_create_m127_mont(
    const M127Mont* coefficients,
    size_t row_len,
    uint32_t inverse_rate,
    uint32_t max_batch_rows) {
    last_error[0] = '\0';
    if (!coefficients || row_len == 0 || (row_len & (row_len - 1)) != 0 ||
        inverse_rate < 2 || max_batch_rows == 0) {
        set_error_text("invalid QA GPU context parameters");
        return nullptr;
    }
    Context* context = new (std::nothrow) Context();
    if (!context) {
        set_error_text("failed to allocate CUDA QA context");
        return nullptr;
    }
    context->row_len = row_len;
    context->inverse_rate = inverse_rate;
    context->max_batch_rows = max_batch_rows;

    int device = 0;
    cudaDeviceProp properties{};
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
        set_error_text("failed to query the active CUDA device");
        free_context(context);
        return nullptr;
    }
    std::snprintf(context->device_name, sizeof(context->device_name), "%s", properties.name);
    if (properties.sharedMemPerBlock < std::min(row_len, size_t(TILE)) * sizeof(M127Mont)) {
        set_error_text("CUDA device has insufficient shared memory for the WHT tile");
        free_context(context);
        return nullptr;
    }

    const size_t coefficient_elements = size_t(inverse_rate - 1) * row_len;
    const size_t message_elements = size_t(max_batch_rows) * row_len;
    const size_t parity_elements = message_elements * size_t(inverse_rate - 1);
    const size_t codeword_elements = message_elements * size_t(inverse_rate);

    cudaError_t status = cudaMalloc(
        reinterpret_cast<void**>(&context->coefficients), coefficient_elements * sizeof(M127Mont));
    if (status == cudaSuccess) status = cudaMalloc(
        reinterpret_cast<void**>(&context->messages), message_elements * sizeof(M127Mont));
    if (status == cudaSuccess) status = cudaMalloc(
        reinterpret_cast<void**>(&context->middle), message_elements * sizeof(M127Mont));
    if (status == cudaSuccess) status = cudaMalloc(
        reinterpret_cast<void**>(&context->parity), parity_elements * sizeof(M127Mont));
    if (status == cudaSuccess) status = cudaMalloc(
        reinterpret_cast<void**>(&context->codewords), codeword_elements * sizeof(M127Mont));
    if (status == cudaSuccess) {
        status = cudaMemcpy(
            context->coefficients,
            coefficients,
            coefficient_elements * sizeof(M127Mont),
            cudaMemcpyHostToDevice);
    }
    for (auto& event : context->events) {
        if (status == cudaSuccess) status = cudaEventCreate(&event);
    }
    if (status != cudaSuccess) {
        set_error("CUDA context allocation", status);
        free_context(context);
        return nullptr;
    }
    return context;
}

extern "C" int qa_gpu_device_name_m127_mont(void* raw, char* output, size_t capacity) {
    if (!raw || !output || capacity == 0) {
        set_error_text("invalid device-name output buffer");
        return 1;
    }
    const Context* context = static_cast<const Context*>(raw);
    std::snprintf(output, capacity, "%s", context->device_name);
    return 0;
}

extern "C" int qa_gpu_host_register_m127_mont(M127Mont* values, size_t elements) {
    last_error[0] = '\0';
    if (!values || elements == 0 || elements > SIZE_MAX / sizeof(M127Mont)) {
        set_error_text("invalid reusable host buffer");
        return 1;
    }
    const cudaError_t status = cudaHostRegister(
        values, elements * sizeof(M127Mont), cudaHostRegisterDefault);
    if (status != cudaSuccess) {
        set_error("cudaHostRegister(reusable host buffer)", status);
        return 1;
    }
    return 0;
}

extern "C" int qa_gpu_host_unregister_m127_mont(M127Mont* values) {
    last_error[0] = '\0';
    if (!values) {
        set_error_text("null reusable host buffer");
        return 1;
    }
    const cudaError_t status = cudaHostUnregister(values);
    if (status != cudaSuccess) {
        set_error("cudaHostUnregister(reusable host buffer)", status);
        return 1;
    }
    return 0;
}

extern "C" int qa_gpu_host_register_bytes(void* values, size_t bytes) {
    last_error[0] = '\0';
    if (!values || bytes == 0) {
        set_error_text("invalid reusable byte buffer");
        return 1;
    }
    const cudaError_t status = cudaHostRegister(values, bytes, cudaHostRegisterDefault);
    if (status != cudaSuccess) {
        set_error("cudaHostRegister(reusable byte buffer)", status);
        return 1;
    }
    return 0;
}

extern "C" int qa_gpu_host_unregister_bytes(void* values) {
    last_error[0] = '\0';
    if (!values) {
        set_error_text("null reusable byte buffer");
        return 1;
    }
    const cudaError_t status = cudaHostUnregister(values);
    if (status != cudaSuccess) {
        set_error("cudaHostUnregister(reusable byte buffer)", status);
        return 1;
    }
    return 0;
}

extern "C" void* qa_gpu_create_device_commitment_m127_mont(
    size_t rows,
    size_t columns) {
    last_error[0] = '\0';
    if (rows == 0 || columns == 0 ||
        rows > SIZE_MAX / columns || rows * columns > SIZE_MAX / sizeof(M127Mont) ||
        columns > SIZE_MAX / 32) {
        set_error_text("invalid device commitment dimensions");
        return nullptr;
    }
    DeviceCommitment* commitment = new (std::nothrow) DeviceCommitment();
    if (!commitment) {
        set_error_text("failed to allocate device commitment handle");
        return nullptr;
    }
    commitment->rows = rows;
    commitment->columns = columns;
    cudaError_t status = cudaMalloc(
        reinterpret_cast<void**>(&commitment->codewords),
        rows * columns * sizeof(M127Mont));
    if (status == cudaSuccess) status = cudaMalloc(
        reinterpret_cast<void**>(&commitment->leaf_digests), columns * 32);
    if (status == cudaSuccess) status = cudaMalloc(
        reinterpret_cast<void**>(&commitment->query_column), rows * sizeof(M127Mont));
    for (auto& event : commitment->events) {
        if (status == cudaSuccess) status = cudaEventCreate(&event);
    }
    if (status != cudaSuccess) {
        set_error("device commitment allocation", status);
        free_device_commitment(commitment);
        return nullptr;
    }
    return commitment;
}

extern "C" void qa_gpu_destroy_device_commitment_m127_mont(void* raw) {
    free_device_commitment(static_cast<DeviceCommitment*>(raw));
}

extern "C" int qa_gpu_encode_store_m127_mont(
    void* raw_context,
    const M127Mont* messages,
    uint32_t rows,
    void* raw_commitment,
    size_t row_offset,
    RawGpuTiming* timing) {
    last_error[0] = '\0';
    if (!raw_context || !messages || !raw_commitment || !timing) {
        set_error_text("null pointer passed to qa_gpu_encode_store_m127_mont");
        return 1;
    }
    Context* context = static_cast<Context*>(raw_context);
    DeviceCommitment* commitment = static_cast<DeviceCommitment*>(raw_commitment);
    if (rows == 0 || rows > context->max_batch_rows ||
        row_offset > commitment->rows || size_t(rows) > commitment->rows - row_offset ||
        commitment->columns != context->inverse_rate * context->row_len) {
        set_error_text("device commitment batch shape mismatch");
        return 1;
    }
    const size_t n = context->row_len;
    const size_t c = context->inverse_rate;
    const size_t message_elements = size_t(rows) * n;
    const size_t parity_elements = message_elements * (c - 1);
    const size_t codeword_elements = message_elements * c;
    std::memset(timing, 0, sizeof(*timing));

    cudaEventRecord(context->events[0]);
    if (cudaMemcpy(context->messages, messages, message_elements * sizeof(M127Mont),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        set_error_text("message host-to-device copy failed");
        return 1;
    }
    cudaEventRecord(context->events[1]);
    if (cudaMemcpy(context->middle, context->messages, message_elements * sizeof(M127Mont),
                   cudaMemcpyDeviceToDevice) != cudaSuccess) {
        set_error_text("message device working-copy failed");
        return 1;
    }
    cudaEventRecord(context->events[2]);
    if (!launch_wht(context->middle, n, rows)) return 1;
    cudaEventRecord(context->events[3]);

    size_t blocks = (parity_elements + THREADS - 1) / THREADS;
    scale_blocks<<<blocks, THREADS>>>(
        context->middle, context->coefficients, context->parity, n, rows, c - 1);
    if (cudaGetLastError() != cudaSuccess) {
        set_error_text("QA Montgomery scaling kernel launch failed");
        return 1;
    }
    cudaEventRecord(context->events[4]);
    if (!launch_wht(context->parity, n, size_t(rows) * (c - 1))) return 1;
    cudaEventRecord(context->events[5]);

    blocks = (codeword_elements + THREADS - 1) / THREADS;
    assemble_codeword<<<blocks, THREADS>>>(
        context->messages,
        context->parity,
        commitment->codewords + row_offset * commitment->columns,
        n,
        rows,
        c);
    if (cudaGetLastError() != cudaSuccess) {
        set_error_text("QA codeword assembly kernel launch failed");
        return 1;
    }
    cudaEventRecord(context->events[6]);
    if (cudaEventSynchronize(context->events[6]) != cudaSuccess) {
        set_error_text("CUDA device-resident encoder synchronization failed");
        return 1;
    }

    if (!event_elapsed(&timing->host_to_device_ms, context->events[0], context->events[1]) ||
        !event_elapsed(&timing->device_input_copy_ms, context->events[1], context->events[2]) ||
        !event_elapsed(&timing->first_wht_ms, context->events[2], context->events[3]) ||
        !event_elapsed(&timing->scaling_ms, context->events[3], context->events[4]) ||
        !event_elapsed(&timing->second_wht_ms, context->events[4], context->events[5]) ||
        !event_elapsed(&timing->assemble_ms, context->events[5], context->events[6]) ||
        !event_elapsed(&timing->total_cuda_ms, context->events[0], context->events[6])) {
        return 1;
    }
    return 0;
}

extern "C" int qa_gpu_hash_columns_blake2b256_m127_mont(
    void* raw_commitment,
    uint8_t* host_digests,
    RawGpuMerkleTiming* timing) {
    last_error[0] = '\0';
    if (!raw_commitment || !host_digests || !timing) {
        set_error_text("null pointer passed to GPU column hashing");
        return 1;
    }
    DeviceCommitment* commitment = static_cast<DeviceCommitment*>(raw_commitment);
    std::memset(timing, 0, sizeof(*timing));
    cudaEventRecord(commitment->events[0]);
    const size_t blocks = (commitment->columns + THREADS - 1) / THREADS;
    hash_columns_blake2b256<<<blocks, THREADS>>>(
        commitment->codewords,
        commitment->leaf_digests,
        commitment->rows,
        commitment->columns);
    if (cudaGetLastError() != cudaSuccess) {
        set_error_text("BLAKE2b-256 column-hash kernel launch failed");
        return 1;
    }
    cudaEventRecord(commitment->events[1]);
    if (cudaMemcpyAsync(
            host_digests,
            commitment->leaf_digests,
            commitment->columns * 32,
            cudaMemcpyDeviceToHost) != cudaSuccess) {
        set_error_text("leaf-digest device-to-host copy failed");
        return 1;
    }
    cudaEventRecord(commitment->events[2]);
    if (cudaEventSynchronize(commitment->events[2]) != cudaSuccess) {
        set_error_text("GPU leaf-digest synchronization failed");
        return 1;
    }
    if (!event_elapsed(
            &timing->column_hash_ms,
            commitment->events[0],
            commitment->events[1]) ||
        !event_elapsed(
            &timing->digest_device_to_host_ms,
            commitment->events[1],
            commitment->events[2]) ||
        !event_elapsed(
            &timing->total_cuda_ms,
            commitment->events[0],
            commitment->events[2])) {
        return 1;
    }
    return 0;
}

extern "C" int qa_gpu_read_column_m127_mont(
    void* raw_commitment,
    size_t column,
    M127Mont* host_output) {
    last_error[0] = '\0';
    if (!raw_commitment || !host_output) {
        set_error_text("null pointer passed to device column opening");
        return 1;
    }
    DeviceCommitment* commitment = static_cast<DeviceCommitment*>(raw_commitment);
    if (column >= commitment->columns) {
        set_error_text("device column index is out of bounds");
        return 1;
    }
    const size_t blocks = (commitment->rows + THREADS - 1) / THREADS;
    gather_column<<<blocks, THREADS>>>(
        commitment->codewords,
        commitment->query_column,
        commitment->rows,
        commitment->columns,
        column);
    if (cudaGetLastError() != cudaSuccess) {
        set_error_text("queried-column gather kernel launch failed");
        return 1;
    }
    if (cudaMemcpy(
            host_output,
            commitment->query_column,
            commitment->rows * sizeof(M127Mont),
            cudaMemcpyDeviceToHost) != cudaSuccess) {
        set_error_text("queried-column device-to-host copy failed");
        return 1;
    }
    return 0;
}

extern "C" int qa_gpu_encode_m127_mont(
    void* raw,
    const M127Mont* messages,
    uint32_t rows,
    M127Mont* output,
    RawGpuTiming* timing) {
    last_error[0] = '\0';
    if (!raw || !messages || !output || !timing) {
        set_error_text("null pointer passed to qa_gpu_encode_m127_mont");
        return 1;
    }
    Context* context = static_cast<Context*>(raw);
    if (rows == 0 || rows > context->max_batch_rows) {
        set_error_text("row batch exceeds the configured CUDA buffer");
        return 1;
    }
    const size_t n = context->row_len;
    const size_t c = context->inverse_rate;
    const size_t message_elements = size_t(rows) * n;
    const size_t parity_elements = message_elements * (c - 1);
    const size_t codeword_elements = message_elements * c;
    std::memset(timing, 0, sizeof(*timing));

    cudaEventRecord(context->events[0]);
    if (cudaMemcpy(context->messages, messages, message_elements * sizeof(M127Mont),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        set_error_text("message host-to-device copy failed");
        return 1;
    }
    cudaEventRecord(context->events[1]);

    if (cudaMemcpy(context->middle, context->messages, message_elements * sizeof(M127Mont),
                   cudaMemcpyDeviceToDevice) != cudaSuccess) {
        set_error_text("message device working-copy failed");
        return 1;
    }
    cudaEventRecord(context->events[2]);

    if (!launch_wht(context->middle, n, rows)) return 1;
    cudaEventRecord(context->events[3]);

    size_t blocks = (parity_elements + THREADS - 1) / THREADS;
    scale_blocks<<<blocks, THREADS>>>(
        context->middle, context->coefficients, context->parity, n, rows, c - 1);
    if (cudaGetLastError() != cudaSuccess) {
        set_error_text("QA Montgomery scaling kernel launch failed");
        return 1;
    }
    cudaEventRecord(context->events[4]);

    if (!launch_wht(context->parity, n, size_t(rows) * (c - 1))) return 1;
    cudaEventRecord(context->events[5]);

    blocks = (codeword_elements + THREADS - 1) / THREADS;
    assemble_codeword<<<blocks, THREADS>>>(
        context->messages, context->parity, context->codewords, n, rows, c);
    if (cudaGetLastError() != cudaSuccess) {
        set_error_text("QA codeword assembly kernel launch failed");
        return 1;
    }
    cudaEventRecord(context->events[6]);

    if (cudaMemcpyAsync(output, context->codewords, codeword_elements * sizeof(M127Mont),
                        cudaMemcpyDeviceToHost) != cudaSuccess) {
        set_error_text("codeword device-to-host copy failed");
        return 1;
    }
    cudaEventRecord(context->events[7]);
    if (cudaEventSynchronize(context->events[7]) != cudaSuccess) {
        set_error_text("CUDA QA encoder synchronization failed");
        return 1;
    }

    if (!event_elapsed(&timing->host_to_device_ms, context->events[0], context->events[1]) ||
        !event_elapsed(&timing->device_input_copy_ms, context->events[1], context->events[2]) ||
        !event_elapsed(&timing->first_wht_ms, context->events[2], context->events[3]) ||
        !event_elapsed(&timing->scaling_ms, context->events[3], context->events[4]) ||
        !event_elapsed(&timing->second_wht_ms, context->events[4], context->events[5]) ||
        !event_elapsed(&timing->assemble_ms, context->events[5], context->events[6]) ||
        !event_elapsed(&timing->device_to_host_ms, context->events[6], context->events[7]) ||
        !event_elapsed(&timing->total_cuda_ms, context->events[0], context->events[7])) {
        return 1;
    }
    return 0;
}

extern "C" void qa_gpu_destroy_m127_mont(void* raw) {
    free_context(static_cast<Context*>(raw));
}
