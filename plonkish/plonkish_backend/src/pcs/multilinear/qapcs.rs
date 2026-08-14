//! QAPCS: Brakedown-style multilinear PCS with QA-code row encoding.
//!
//! This file intentionally keeps the Brakedown PCS protocol structure:
//!
//! 1. reshape multilinear evaluations into a row-major matrix;
//! 2. encode each row;
//! 3. Merkle-commit to encoded columns;
//! 4. during a full opening, send random proximity-folded rows;
//! 5. send the evaluation-folded row determined by the public evaluation point;
//! 6. sample and open encoded columns from the committed matrix;
//! 7. the verifier locally re-encodes every folded row, checks all sampled
//!    columns, and checks that the evaluation-folded message evaluates to the
//!    public claimed value.
//!
//! Thus `open`/`verify` implement a full multilinear opening, not merely a
//! proximity test.
//!
//! The only protocol-level replacement is the row code: instead of the GLSTW21
//! Brakedown/RAA code, each row is encoded by the QA code
//!
//!     m -> (m, WHT(E_0 * WHT(m)), ..., WHT(E_{rho-2} * WHT(m))).
//!
//! For the requested rate 1/2 instantiation, `rho = inverse_rate = 2`, so the
//! codeword is simply
//!
//!     m -> (m, WHT(E_0 * WHT(m))).
//!
//! Matrix shape selection follows the Brakedown implementation's proxy:
//!
//!     proof_proxy = (1 + num_proximity_testing) * row_len
//!                 + num_column_opening * num_rows,
//!
//! but `num_column_opening` is computed from the QABase QA-distance lower bound.

use rayon::{iter::{IntoParallelRefIterator, ParallelIterator}, slice::ParallelSlice};

use crate::{
    pcs::{multilinear::validate_input, Evaluation, Point, PolynomialCommitmentScheme},
    poly::{multilinear::MultilinearPolynomial, Polynomial},
    util::{
        arithmetic::{div_ceil, inner_product, Field, PrimeField},
        hash::{Hash, Output},
        parallel::{num_threads, parallelize, parallelize_iter},
        transcript::{FieldTranscript, TranscriptRead, TranscriptWrite},
        Deserialize, DeserializeOwned, Itertools, Serialize,
    },
    Error,
};

use rand::RngCore;
use std::{fmt::Debug, marker::PhantomData, mem::size_of, slice};

// -----------------------------------------------------------------------------
// Public PCS type
// -----------------------------------------------------------------------------

#[derive(Debug)]
pub struct MultilinearQAPCS<F: PrimeField, H: Hash, S: QAPCSSpec>(PhantomData<(F, H, S)>);

impl<F: PrimeField, H: Hash, S: QAPCSSpec> Clone for MultilinearQAPCS<F, H, S> {
    fn clone(&self) -> Self {
        Self(PhantomData)
    }
}

// -----------------------------------------------------------------------------
// QAPCS spec
// -----------------------------------------------------------------------------

/// Parameter policy for the Brakedown-style QA PCS.
///
/// The default values implement the requested setting:
///
/// - QA inverse rate rho = 2, i.e. code rate 1/2;
/// - 100-bit query soundness;
/// - 100-bit distance-failure budget;
/// - Brakedown-style matrix-shape search by proof-size proxy.
pub trait QAPCSSpec: Debug {
    /// QA inverse rate. `2` means rate 1/2.
    fn inverse_rate() -> usize {
        2
    }

    /// Target soundness bits for Merkle column queries.
    fn security_bits() -> usize {
        100
    }

    /// Failure budget used by the QA distance lower-bound search.
    fn distance_failure_bits() -> usize {
        100
    }

    /// Minimum row length exponent considered during matrix-shape search.
    ///
    /// Brakedown has a base-code threshold `n_0`; QA does not need that exact
    /// threshold, but excluding extremely tiny rows avoids degenerate parameter
    /// choices in tests and benchmarks.
    fn min_row_log_size() -> usize {
        5
    }

    /// Whether to include Merkle authentication hashes in the matrix-shape
    /// objective. The original Brakedown implementation only optimizes the
    /// field-element proxy, so the default is `false`.
    fn optimize_for_bytes() -> bool {
        false
    }

    /// Hash output size used only when `optimize_for_bytes() == true`.
    fn hash_bytes() -> usize {
        32
    }

    /// Field element size used only when `optimize_for_bytes() == true`.
    fn field_bytes(log2_q: usize) -> usize {
        div_ceil(log2_q, 8)
    }
}

/// Default requested instantiation: QA rate 1/2 and 100-bit security.
#[derive(Debug)]
pub struct QAPCSSpecRateHalf100;

impl QAPCSSpec for QAPCSSpecRateHalf100 {}

// -----------------------------------------------------------------------------
// QA distance and query selection, ported from QABase
// -----------------------------------------------------------------------------

/// p-ary GV-style exponent:
///
///     g_p(delta)
///       = 1 - delta log_p(p - 1)
///         + delta log_p(delta)
///         + (1 - delta) log_p(1 - delta).
///
/// For p roughly 2^field_bits, we approximate log_p(p - 1) by 1.
pub fn qabase_gp(delta: f64, field_bits: usize) -> f64 {
    assert!(delta > 0.0 && delta < 1.0);

    let bits = field_bits as f64;

    1.0 - delta
        + (delta * delta.log2() + (1.0 - delta) * (1.0 - delta).log2()) / bits
}

/// log2(2^a + 2^b), computed stably.
fn log2_add(a: f64, b: f64) -> f64 {
    let m = a.max(b);
    if !m.is_finite() {
        return m;
    }
    m + ((2.0f64).powf(a - m) + (2.0f64).powf(b - m)).log2()
}

/// Compute log2 of the QA random-code distance failure bound.
pub fn qabase_distance_failure_log2(
    delta: f64,
    row_log_size: usize,
    inverse_rate: usize,
    field_bits: usize,
) -> f64 {
    let c = inverse_rate;
    assert!(c >= 2, "inverse_rate must be at least 2");

    let log_n = row_log_size as f64;

    // Approximation: p ≈ 2^field_bits.
    // For Mersenne127, p = 2^127 - 1, so log2(p) and log2(p-1)
    // are both essentially 127 at this precision.
    let log_p = field_bits as f64;
    let log_p_minus_one = field_bits as f64;

    let eps =
        qabase_gp(delta, field_bits)
            - (1.0 + log_n / log_p) / (c as f64);

    if eps <= 0.0 {
        return f64::INFINITY;
    }

    let denom_log =
        if log_p * (c as f64) * eps < 60.0 {
            // log2(1 - p^{-c eps})
            let x = log_p * (c as f64) * eps;
            (-(-std::f64::consts::LN_2 * x).exp_m1()).log2()
        } else {
            0.0
        };

    // Bound 1:
    //
    //   c(c-1)N/(2p^2)
    //     + p^{-ceil((c-1)/(c delta)) c eps}
    //       / ((1 - p^{-c eps})(p - 1)).
    let log_term1_a =
        ((c * (c - 1)) as f64 / 2.0).log2()
            + log_n
            - 2.0 * log_p;

    let threshold1 =
        (((c - 1) as f64) / ((c as f64) * delta)).ceil();

    let log_term1_b =
        -log_p * threshold1 * (c as f64) * eps
            - denom_log
            - log_p_minus_one;

    let log_bound1 = log2_add(log_term1_a, log_term1_b);

    // Bound 2:
    //
    //   cN/p
    //     + p^{-ceil(1/delta) c eps}
    //       / ((1 - p^{-c eps})(p - 1)).
    let log_term2_a =
        (c as f64).log2() + log_n - log_p;

    let threshold2 = (1.0 / delta).ceil();

    let log_term2_b =
        -log_p * threshold2 * (c as f64) * eps
            - denom_log
            - log_p_minus_one;

    let log_bound2 = log2_add(log_term2_a, log_term2_b);

    log_bound1.min(log_bound2)
}

/// Find the largest delta whose distance-failure probability is at most
/// 2^{-failure_bits}.
pub fn qabase_distance_lower_bound(
    row_log_size: usize,
    inverse_rate: usize,
    field_bits: usize,
    failure_bits: usize,
) -> f64 {
    let target_log2 = -(failure_bits as f64);

    let mut lo = 1e-9f64;
    let mut hi = 1.0 - 1e-9f64;

    for _ in 0..100 {
        let mid = (lo + hi) * 0.5;
        let failure_log2 =
            qabase_distance_failure_log2(mid, row_log_size, inverse_rate, field_bits);

        if failure_log2 <= target_log2 {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    lo
}

/// Compute opened columns / Merkle queries from QA distance delta:
///
///     t = ceil(lambda / -log2(1 - delta/3)).
pub fn qabase_queries_from_distance(delta: f64, security_bits: usize) -> usize {
    assert!(delta > 0.0 && delta < 1.0);

    let effective = delta / 3.0;
    let denom = -(1.0 - effective).log2();

    ((security_bits as f64) / denom).ceil() as usize
}

fn ceil(v: f64) -> usize {
    v.ceil() as usize
}

/// Brakedown-style random-row-combination repetition count.
///
/// This mirrors the shape used in the Brakedown implementation:
///
///     ceil(lambda / (log2(|F|) - log2(codeword_len))).
fn qapcs_num_proximity_testing(
    field_bits: usize,
    codeword_len: usize,
    security_bits: usize,
) -> usize {
    let denom = field_bits as f64 - (codeword_len as f64).log2();
    assert!(
        denom > 0.0,
        "field too small for Brakedown-style folded-row soundness: field_bits={}, codeword_len={}",
        field_bits,
        codeword_len
    );
    ceil((security_bits as f64) / denom).max(1)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QAPCSShape {
    pub row_log_size: usize,
    pub row_len: usize,
    pub num_rows: usize,
    pub codeword_len: usize,
    pub delta: f64,
    pub num_column_opening: usize,
    pub num_proximity_testing: usize,
    pub proof_proxy: usize,
}

/// Brakedown-style objective using QA-derived query count.
fn qapcs_shape_score<S: QAPCSSpec>(
    total_log_size: usize,
    row_log_size: usize,
    field_bits: usize,
) -> QAPCSShape {
    assert!(row_log_size <= total_log_size);

    let inverse_rate = S::inverse_rate();
    assert!(inverse_rate >= 2);
    assert!(inverse_rate.is_power_of_two());

    let row_len = 1usize << row_log_size;
    let num_rows = 1usize << (total_log_size - row_log_size);
    let codeword_len = inverse_rate * row_len;

    let delta = qabase_distance_lower_bound(
        row_log_size,
        inverse_rate,
        field_bits,
        S::distance_failure_bits(),
    );
    let num_column_opening = qabase_queries_from_distance(delta, S::security_bits());
    let num_proximity_testing =
        qapcs_num_proximity_testing(field_bits, codeword_len, S::security_bits());

    let field_elems = (1 + num_proximity_testing) * row_len + num_column_opening * num_rows;

    let proof_proxy = if S::optimize_for_bytes() {
        let merkle_depth = codeword_len.next_power_of_two().ilog2() as usize;
        field_elems * S::field_bytes(field_bits)
            + num_column_opening * merkle_depth * S::hash_bytes()
    } else {
        field_elems
    };

    QAPCSShape {
        row_log_size,
        row_len,
        num_rows,
        codeword_len,
        delta,
        num_column_opening,
        num_proximity_testing,
        proof_proxy,
    }
}

fn choose_qapcs_shape<S: QAPCSSpec>(total_log_size: usize, field_bits: usize) -> QAPCSShape {
    let min_row_log_size = S::min_row_log_size().min(total_log_size);

    let mut best: Option<QAPCSShape> = None;
    for row_log_size in min_row_log_size..=total_log_size {
        let shape = qapcs_shape_score::<S>(total_log_size, row_log_size, field_bits);
        if best
            .as_ref()
            .map(|best| shape.proof_proxy < best.proof_proxy)
            .unwrap_or(true)
        {
            best = Some(shape);
        }
    }

    best.expect("shape search must have at least one candidate")
}

// -----------------------------------------------------------------------------
// QA code and encoding
// -----------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: DeserializeOwned"))]
pub struct QAParams<F: PrimeField> {
    pub inverse_rate: usize,
    pub e: Vec<Vec<F>>,
}

impl<F> QAParams<F>
where
    F: PrimeField,
{
    pub fn new_random(msg_len: usize, inverse_rate: usize, rng: &mut impl RngCore) -> Self {
        assert!(
            msg_len.is_power_of_two(),
            "QA message length must be a power of two"
        );
        assert!(inverse_rate >= 2, "inverse_rate must be at least 2");
        assert!(
            inverse_rate.is_power_of_two(),
            "inverse_rate must be a power of two"
        );

        let e = (0..inverse_rate - 1)
            .map(|_| {
                (0..msg_len)
                    .map(|_| F::random(&mut *rng))
                    .collect::<Vec<F>>()
            })
            .collect::<Vec<Vec<F>>>();

        Self { inverse_rate, e }
    }
}

/// In-place unnormalized Walsh--Hadamard transform.
pub fn wht<F: Field>(x: &mut [F]) {
    let len = x.len();
    assert!(len.is_power_of_two(), "WHT length must be a power of two");

    let mut step = 1usize;
    while 2 * step <= len {
        let chunk_len = 2 * step;

        for chunk in x.chunks_exact_mut(chunk_len) {
            let (left, right) = chunk.split_at_mut(step);
            for (l, r) in left.iter_mut().zip(right.iter_mut()) {
                let u = *l;
                let v = *r;
                *l = u + v;
                *r = u - v;
            }
        }

        step <<= 1;
    }
}

/// QA encoding for commitment/opening only.
///
/// Computes:
///
///     codeword = m || WHT(E_0 * WHT(m)) || ... || WHT(E_{rho-2} * WHT(m)).
pub fn qa_encode_codeword_only<F>(msg: &[F], params: &QAParams<F>) -> Vec<F>
where
    F: PrimeField,
{
    let n = msg.len();
    let rho = params.inverse_rate;

    assert!(n.is_power_of_two(), "message length must be a power of two");
    assert!(rho >= 2, "inverse_rate must be at least 2");
    assert!(rho.is_power_of_two(), "inverse_rate must be a power of two");
    assert_eq!(
        params.e.len(),
        rho - 1,
        "QA encoding needs rho - 1 coefficient vectors"
    );

    for coeffs in &params.e {
        assert_eq!(
            coeffs.len(),
            n,
            "each QA coefficient vector must have length msg.len()"
        );
    }

    let mut msg_wht = msg.to_vec();
    wht(&mut msg_wht);

    let mut codeword = Vec::with_capacity(rho * n);
    codeword.extend_from_slice(msg);

    for i in 0..(rho - 1) {
        let mut block = if i == rho - 2 {
            core::mem::take(&mut msg_wht)
        } else {
            msg_wht.clone()
        };

        for (x, e) in block.iter_mut().zip(params.e[i].iter()) {
            *x *= *e;
        }

        wht(&mut block);
        codeword.extend(block);
    }

    debug_assert_eq!(codeword.len(), rho * n);
    codeword
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: DeserializeOwned"))]
pub struct QACode<F: PrimeField> {
    row_len: usize,
    codeword_len: usize,
    num_column_opening: usize,
    num_proximity_testing: usize,
    delta: f64,
    qa_params: QAParams<F>,
}

impl<F: PrimeField> QACode<F> {
    pub fn new_random(
        shape: &QAPCSShape,
        inverse_rate: usize,
        rng: &mut impl RngCore,
    ) -> Self {
        assert_eq!(shape.codeword_len, inverse_rate * shape.row_len);
        let qa_params = QAParams::<F>::new_random(shape.row_len, inverse_rate, rng);

        Self {
            row_len: shape.row_len,
            codeword_len: shape.codeword_len,
            num_column_opening: shape.num_column_opening,
            num_proximity_testing: shape.num_proximity_testing,
            delta: shape.delta,
            qa_params,
        }
    }

    pub fn row_len(&self) -> usize {
        self.row_len
    }

    pub fn codeword_len(&self) -> usize {
        self.codeword_len
    }

    pub fn num_column_opening(&self) -> usize {
        self.num_column_opening
    }

    pub fn num_proximity_testing(&self) -> usize {
        self.num_proximity_testing
    }

    pub fn delta(&self) -> f64 {
        self.delta
    }

    pub fn qa_params(&self) -> &QAParams<F> {
        &self.qa_params
    }

    /// Encode `msg` directly into `target`.
    ///
    /// This avoids the extra full-codeword allocation used by
    /// `qa_encode_codeword_only` and is the preferred path during commitment.
    pub fn encode_msg_into(&self, msg: &[F], target: &mut [F]) {
        let n = self.row_len;
        let rho = self.qa_params.inverse_rate;

        assert_eq!(msg.len(), n, "QA message length mismatch");
        assert_eq!(target.len(), self.codeword_len, "QA target length mismatch");
        assert_eq!(self.codeword_len, rho * n, "QA codeword length mismatch");

        target[..n].copy_from_slice(msg);

        let mut msg_wht = msg.to_vec();
        wht(&mut msg_wht);

        for i in 0..(rho - 1) {
            let mut block = if i == rho - 2 {
                core::mem::take(&mut msg_wht)
            } else {
                msg_wht.clone()
            };

            for (x, e) in block.iter_mut().zip(self.qa_params.e[i].iter()) {
                *x *= *e;
            }

            wht(&mut block);

            let start = (i + 1) * n;
            let end = start + n;
            target[start..end].copy_from_slice(&block);
        }
    }

    /// Encode the first `row_len` entries of `target` in place.
    ///
    /// This is used by the verifier for folded rows: after reading a folded
    /// message row, the parity half is appended and filled here.
    pub fn encode(&self, mut target: impl AsMut<[F]>) {
        let target = target.as_mut();
        assert_eq!(target.len(), self.codeword_len);

        let msg = target[..self.row_len].to_vec();
        self.encode_msg_into(&msg, target);
    }
}

// -----------------------------------------------------------------------------
// PCS parameters and commitments
// -----------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: DeserializeOwned"))]
pub struct MultilinearQAPCSParams<F: PrimeField> {
    num_vars: usize,
    num_rows: usize,
    shape: QAPCSShape,
    qa: QACode<F>,
}

impl<F: PrimeField> MultilinearQAPCSParams<F> {
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    pub fn row_len(&self) -> usize {
        self.qa.row_len()
    }

    pub fn codeword_len(&self) -> usize {
        self.qa.codeword_len()
    }

    pub fn shape(&self) -> &QAPCSShape {
        &self.shape
    }

    pub fn qa(&self) -> &QACode<F> {
        &self.qa
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: DeserializeOwned"))]
pub struct MultilinearQAPCSCommitment<F, H: Hash> {
    /// Row-wise QA codewords.
    ///
    /// `rows[row][col]` is the `col`-th encoded coordinate of the `row`-th
    /// original message row.  This Vec<Vec<_>> layout matches the optimized
    /// QABase commitment path: each row is allocated and encoded independently,
    /// avoiding a full flattened zero-initialization followed by extra copies.
    rows: Vec<Vec<F>>,
    intermediate_hashes: Vec<Output<H>>,
    root: Output<H>,
}

impl<F: PrimeField, H: Hash> MultilinearQAPCSCommitment<F, H> {
    fn from_root(root: Output<H>) -> Self {
        Self {
            root,
            ..Default::default()
        }
    }

    pub fn rows(&self) -> &[Vec<F>] {
        &self.rows
    }

    pub fn intermediate_hashes(&self) -> &[Output<H>] {
        &self.intermediate_hashes
    }

    pub fn root(&self) -> &Output<H> {
        &self.root
    }
}

impl<F: PrimeField, H: Hash> AsRef<[Output<H>]> for MultilinearQAPCSCommitment<F, H> {
    fn as_ref(&self) -> &[Output<H>] {
        slice::from_ref(&self.root)
    }
}


// -----------------------------------------------------------------------------
// Full-opening helpers
// -----------------------------------------------------------------------------

/// Compute a row-linear combination of the row-major evaluation matrix.
///
/// The polynomial evaluations are interpreted as
///
///     matrix[row][column] = evals[row * row_len + column].
///
/// `coeffs` has one coefficient per matrix row and `combined_row` has
/// `row_len` entries.
fn combine_message_rows<F: PrimeField>(
    evals: &[F],
    num_rows: usize,
    row_len: usize,
    coeffs: &[F],
    combined_row: &mut [F],
) -> Result<(), Error> {
    if coeffs.len() != num_rows
        || combined_row.len() != row_len
        || evals.len() != num_rows * row_len
    {
        return Err(Error::InvalidPcsParam(
            "invalid QAPCS row-combination dimensions".to_string(),
        ));
    }

    parallelize(combined_row, |(combined_row, offset)| {
        combined_row
            .iter_mut()
            .zip(offset..)
            .for_each(|(combined, column)| {
                let mut acc = F::ZERO;
                for (row, coeff) in coeffs.iter().enumerate() {
                    acc += *coeff * evals[row * row_len + column];
                }
                *combined = acc;
            });
    });

    Ok(())
}

/// Validate the prover-side commitment state used to answer Merkle openings.
fn validate_qapcs_commitment_state<F: PrimeField, H: Hash>(
    pp: &MultilinearQAPCSParams<F>,
    comm: &MultilinearQAPCSCommitment<F, H>,
) -> Result<(), Error> {
    let codeword_len = pp.codeword_len();
    let expected_intermediate_hashes = 2 * codeword_len - 2;

    if comm.rows.len() != pp.num_rows
        || comm.rows.iter().any(|row| row.len() != codeword_len)
        || comm.intermediate_hashes.len() != expected_intermediate_hashes
    {
        return Err(Error::InvalidPcsParam(
            "invalid QAPCS prover commitment state".to_string(),
        ));
    }

    Ok(())
}

/// Full Brakedown-style QAPCS opening.
///
/// This proves both:
///
/// 1. proximity: random row combinations of the committed encoded matrix are
///    valid QA codewords and agree with sampled committed columns;
/// 2. evaluation: the deterministic row combination
///
///        p = eq(z_row, ·)^T M
///
///    is a valid QA codeword, agrees with the same sampled columns, and
///
///        p(z_col) = claimed_value.
///
/// The evaluation-folded row is sent before the Merkle query positions are
/// sampled, so it is bound by the sampled-column test.
pub fn qapcs_open_full<F, H>(
    pp: &MultilinearQAPCSParams<F>,
    poly: &MultilinearPolynomial<F>,
    comm: &MultilinearQAPCSCommitment<F, H>,
    point: &Point<F, MultilinearPolynomial<F>>,
    eval: &F,
    transcript: &mut impl TranscriptWrite<Output<H>, F>,
) -> Result<(), Error>
where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
{
    validate_input("open", pp.num_vars(), [poly], [point])?;
    validate_qapcs_commitment_state(pp, comm)?;

    let row_len = pp.row_len();
    let codeword_len = pp.codeword_len();
    debug_assert!(codeword_len.is_power_of_two());

    // For row-major flattening, the first variables index columns and the last
    // log2(num_rows) variables index rows.
    let (row_weights, column_weights) = point_to_tensor(pp.num_rows, point);

    // Construct and check the deterministic evaluation-folded row first.
    // The row is written later, after the random proximity rows, preserving the
    // transcript order of the original implementation.
    let mut evaluation_row = vec![F::ZERO; row_len];
    combine_message_rows(
        poly.evals(),
        pp.num_rows,
        row_len,
        &row_weights,
        &mut evaluation_row,
    )?;

    if inner_product(&evaluation_row, &column_weights) != *eval {
        return Err(Error::InvalidPcsOpen(
            "claimed QAPCS evaluation does not match the polynomial".to_string(),
        ));
    }

    // Random proximity-folded rows.
    if pp.num_rows > 1 {
        let mut combined_row = vec![F::ZERO; row_len];
        for _ in 0..pp.qa.num_proximity_testing() {
            let coeffs = transcript.squeeze_challenges(pp.num_rows);
            combine_message_rows(
                poly.evals(),
                pp.num_rows,
                row_len,
                &coeffs,
                &mut combined_row,
            )?;
            transcript.write_field_elements(&combined_row)?;
        }
    }

    // Full evaluation branch: send p = eq(z_row,·)^T M.
    transcript.write_field_elements(&evaluation_row)?;

    // The Merkle query challenges are sampled after all folded rows have been
    // absorbed into the transcript.
    let depth = codeword_len.ilog2() as usize;
    for _ in 0..pp.qa.num_column_opening() {
        let column = squeeze_challenge_idx(transcript, codeword_len);

        transcript.write_field_elements(
            comm.rows.iter().map(|row| &row[column]),
        )?;

        let mut offset = 0;
        for (idx, width) in (1..=depth)
            .rev()
            .map(|depth| 1usize << depth)
            .enumerate()
        {
            let neighbor_idx = (column >> idx) ^ 1;
            transcript.write_commitment(
                &comm.intermediate_hashes[offset + neighbor_idx],
            )?;
            offset += width;
        }
    }

    Ok(())
}

/// Verify the full Brakedown-style QAPCS opening.
pub fn qapcs_verify_full<F, H>(
    vp: &MultilinearQAPCSParams<F>,
    comm: &MultilinearQAPCSCommitment<F, H>,
    point: &Point<F, MultilinearPolynomial<F>>,
    eval: &F,
    transcript: &mut impl TranscriptRead<Output<H>, F>,
) -> Result<(), Error>
where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
{
    validate_input("verify", vp.num_vars(), [], [point])?;

    let row_len = vp.row_len();
    let codeword_len = vp.codeword_len();
    debug_assert!(codeword_len.is_power_of_two());

    let (row_weights, column_weights) = point_to_tensor(vp.num_rows, point);
    let mut combined_rows =
        Vec::with_capacity(vp.qa.num_proximity_testing() + 1);

    // Random proximity-folded rows.
    if vp.num_rows > 1 {
        for _ in 0..vp.qa.num_proximity_testing() {
            let coeffs = transcript.squeeze_challenges(vp.num_rows);
            let mut encoded_row = transcript.read_field_elements(row_len)?;
            encoded_row.resize(codeword_len, F::ZERO);
            vp.qa.encode(&mut encoded_row);
            combined_rows.push((coeffs, encoded_row));
        }
    }

    // Deterministic evaluation-folded row.
    let evaluation_message_row = transcript.read_field_elements(row_len)?;
    if inner_product(&evaluation_message_row, &column_weights) != *eval {
        return Err(Error::InvalidPcsOpen(
            "QAPCS evaluation consistency failure".to_string(),
        ));
    }

    let mut evaluation_encoded_row = evaluation_message_row;
    evaluation_encoded_row.resize(codeword_len, F::ZERO);
    vp.qa.encode(&mut evaluation_encoded_row);
    combined_rows.push((row_weights, evaluation_encoded_row));

    // All folded rows are now fixed. Sample and verify the committed columns.
    let depth = codeword_len.ilog2() as usize;
    for _ in 0..vp.qa.num_column_opening() {
        let column = squeeze_challenge_idx(transcript, codeword_len);
        let items = transcript.read_field_elements(vp.num_rows)?;
        let path = transcript.read_commitments(depth)?;

        for (coeffs, encoded_row) in &combined_rows {
            let projected = if vp.num_rows > 1 {
                inner_product(coeffs, &items)
            } else {
                items[0]
            };

            if projected != encoded_row[column] {
                return Err(Error::InvalidPcsOpen(
                    "QAPCS folded-column consistency failure".to_string(),
                ));
            }
        }

        let mut hasher = H::new();
        let mut output = {
            for item in &items {
                hasher.update_field_element(item);
            }
            hasher.finalize_fixed_reset()
        };

        for (idx, neighbor) in path.iter().enumerate() {
            if (column >> idx) & 1 == 0 {
                hasher.update(&output);
                hasher.update(neighbor);
            } else {
                hasher.update(neighbor);
                hasher.update(&output);
            }
            output = hasher.finalize_fixed_reset();
        }

        if &output != comm.root() {
            return Err(Error::InvalidPcsOpen(
                "invalid QAPCS Merkle opening".to_string(),
            ));
        }
    }

    Ok(())
}

// -----------------------------------------------------------------------------
// PolynomialCommitmentScheme implementation
// -----------------------------------------------------------------------------

impl<F, H, S> PolynomialCommitmentScheme<F> for MultilinearQAPCS<F, H, S>
where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
    S: QAPCSSpec,
{
    type Param = MultilinearQAPCSParams<F>;
    type ProverParam = MultilinearQAPCSParams<F>;
    type VerifierParam = MultilinearQAPCSParams<F>;
    type Polynomial = MultilinearPolynomial<F>;
    type Commitment = MultilinearQAPCSCommitment<F, H>;
    type CommitmentChunk = Output<H>;

    fn setup(poly_size: usize, _: usize, mut rng: impl RngCore) -> Result<Self::Param, Error> {
        assert!(poly_size.is_power_of_two());
        let num_vars = poly_size.ilog2() as usize;
        let field_bits = F::NUM_BITS as usize;

        let shape = choose_qapcs_shape::<S>(num_vars, field_bits);
        let qa = QACode::<F>::new_random(&shape, S::inverse_rate(), &mut rng);

        Ok(MultilinearQAPCSParams {
            num_vars,
            num_rows: shape.num_rows,
            shape,
            qa,
        })
    }

    fn trim(
        param: &Self::Param,
        poly_size: usize,
        _: usize,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), Error> {
        assert!(poly_size.is_power_of_two());
        if poly_size == 1 << param.num_vars {
            Ok((param.clone(), param.clone()))
        } else {
            Err(Error::InvalidPcsParam(
                "Can't trim MultilinearQAPCSParams into different poly_size".to_string(),
            ))
        }
    }

    fn commit(pp: &Self::ProverParam, poly: &Self::Polynomial) -> Result<Self::Commitment, Error> {
        validate_input("commit", pp.num_vars(), [poly], None)?;

        let row_len = pp.qa.row_len();
        let codeword_len = pp.qa.codeword_len();

        // Encode each row independently, matching QABase's optimized
        // `Vec<Vec<F>>` layout. This avoids allocating a flattened
        // `num_rows * codeword_len` zero-filled matrix and then copying the
        // generated codeword back into it.
        let rows = poly
            .evals()
            .par_chunks_exact(row_len)
            .map(|evals| qa_encode_codeword_only(evals, pp.qa.qa_params()))
            .collect::<Vec<Vec<F>>>();

        assert_eq!(rows.len(), pp.num_rows, "encoded row count mismatch");

        // Hash encoded columns. Each Merkle leaf is
        // Hash(C[0, col], C[1, col], ..., C[num_rows - 1, col]).
        let depth = codeword_len.next_power_of_two().ilog2() as usize;
        let mut hashes = vec![Output::<H>::default(); (2 << depth) - 1];

        parallelize(&mut hashes[..codeword_len], |(hashes, start)| {
            let mut hasher = H::new();
            for (hash, column) in hashes.iter_mut().zip(start..) {
                for row in rows.iter() {
                    hasher.update_field_element(&row[column]);
                }
                hasher.finalize_into_reset(hash);
            }
        });

        // Merklize column hashes.
        let mut offset = 0;
        for width in (1..=depth).rev().map(|depth| 1 << depth) {
            let (input, output) = hashes[offset..].split_at_mut(width);
            let chunk_size = div_ceil(output.len(), num_threads());
            parallelize_iter(
                input
                    .chunks(2 * chunk_size)
                    .zip(output.chunks_mut(chunk_size)),
                |(input, output)| {
                    let mut hasher = H::new();

                    for (input, output) in input.chunks_exact(2).zip(output.iter_mut()) {
                        hasher.update(&input[0]);
                        hasher.update(&input[1]);
                        hasher.finalize_into_reset(output);
                    }
                },
            );
            offset += width;
        }

        let (intermediate_hashes, root) = {
            let mut intermediate_hashes = hashes;
            let root = intermediate_hashes.pop().unwrap();
            (intermediate_hashes, root)
        };

        Ok(MultilinearQAPCSCommitment {
            rows,
            intermediate_hashes,
            root,
        })
    }

    fn batch_commit<'a>(
        pp: &Self::ProverParam,
        polys: impl IntoIterator<Item = &'a Self::Polynomial>,
    ) -> Result<Vec<Self::Commitment>, Error>
    where
        Self::Polynomial: 'a,
    {
        let polys_vec: Vec<&Self::Polynomial> = polys.into_iter().collect();
        polys_vec.par_iter().map(|poly| Self::commit(pp, poly)).collect()
    }

    fn open(
        pp: &Self::ProverParam,
        poly: &Self::Polynomial,
        comm: &Self::Commitment,
        point: &Point<F, Self::Polynomial>,
        eval: &F,
        transcript: &mut impl TranscriptWrite<Self::CommitmentChunk, F>,
    ) -> Result<(), Error> {
        qapcs_open_full::<F, H>(
            pp,
            poly,
            comm,
            point,
            eval,
            transcript,
        )
    }

    fn batch_open<'a>(
        pp: &Self::ProverParam,
        polys: impl IntoIterator<Item = &'a Self::Polynomial>,
        comms: impl IntoIterator<Item = &'a Self::Commitment>,
        points: &[Point<F, Self::Polynomial>],
        evals: &[Evaluation<F>],
        transcript: &mut impl TranscriptWrite<Self::CommitmentChunk, F>,
    ) -> Result<(), Error>
    where
        Self::Polynomial: 'a,
        Self::Commitment: 'a,
    {
        let polys = polys.into_iter().collect_vec();
        let comms = comms.into_iter().collect_vec();
        for eval in evals {
            Self::open(
                pp,
                polys[eval.poly()],
                comms[eval.poly()],
                &points[eval.point()],
                eval.value(),
                transcript,
            )?;
        }
        Ok(())
    }

    fn read_commitments(
        _: &Self::VerifierParam,
        num_polys: usize,
        transcript: &mut impl TranscriptRead<Self::CommitmentChunk, F>,
    ) -> Result<Vec<Self::Commitment>, Error> {
        transcript.read_commitments(num_polys).map(|roots| {
            roots
                .into_iter()
                .map(MultilinearQAPCSCommitment::from_root)
                .collect_vec()
        })
    }

    fn verify(
        vp: &Self::VerifierParam,
        comm: &Self::Commitment,
        point: &Point<F, Self::Polynomial>,
        eval: &F,
        transcript: &mut impl TranscriptRead<Self::CommitmentChunk, F>,
    ) -> Result<(), Error> {
        qapcs_verify_full::<F, H>(
            vp,
            comm,
            point,
            eval,
            transcript,
        )
    }

    fn batch_verify<'a>(
        vp: &Self::VerifierParam,
        comms: impl IntoIterator<Item = &'a Self::Commitment>,
        points: &[Point<F, Self::Polynomial>],
        evals: &[Evaluation<F>],
        transcript: &mut impl TranscriptRead<Self::CommitmentChunk, F>,
    ) -> Result<(), Error>
    where
        Self::Commitment: 'a,
    {
        let comms = comms.into_iter().collect_vec();
        for eval in evals {
            Self::verify(
                vp,
                comms[eval.poly()],
                &points[eval.point()],
                eval.value(),
                transcript,
            )?;
        }
        Ok(())
    }
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

fn point_to_tensor<F: PrimeField>(
    num_rows: usize,
    point: &[F],
) -> (Vec<F>, Vec<F>) {
    assert!(num_rows.is_power_of_two());

    let log_rows = num_rows.ilog2() as usize;
    assert!(
        point.len() >= log_rows,
        "evaluation point is too short for the QAPCS row dimension"
    );

    // Row-major flattening:
    //
    //     flat[row * row_len + column].
    //
    // MLE coordinates are little-endian, so the first coordinates index
    // columns and the last log_rows coordinates index rows.
    let (column_point, row_point) =
        point.split_at(point.len() - log_rows);

    let row_weights =
        MultilinearPolynomial::eq_xy(row_point).into_evals();
    let column_weights =
        MultilinearPolynomial::eq_xy(column_point).into_evals();

    (row_weights, column_weights)
}

fn squeeze_challenge_idx<F: PrimeField>(
    transcript: &mut impl FieldTranscript<F>,
    cap: usize,
) -> usize {
    assert!(cap > 0);

    let challenge = transcript.squeeze_challenge();
    let repr = challenge.to_repr();
    let bytes = repr.as_ref();
    let take = bytes.len().min(size_of::<usize>());

    let mut index = 0usize;
    for (i, byte) in bytes.iter().take(take).enumerate() {
        index |= (*byte as usize) << (8 * i);
    }

    index % cap
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod test {
    use super::*;

    use crate::{
        pcs::multilinear::test::{run_batch_commit_open_verify, run_commit_open_verify},
        util::{
            hash::Blake2s,
            transcript::{Blake2sTranscript, InMemoryTranscript},
        },
    };
    use halo2_curves::bn256::Fr;
    use rand_chacha::{rand_core::SeedableRng, ChaCha8Rng};
    use std::io::Cursor;

    #[derive(Debug)]
    struct TestSpec;

    impl QAPCSSpec for TestSpec {
        fn security_bits() -> usize {
            16
        }

        fn distance_failure_bits() -> usize {
            16
        }

        fn min_row_log_size() -> usize {
            3
        }
    }

    type Pcs = MultilinearQAPCS<Fr, Blake2s, TestSpec>;

    #[test]
    fn commit_open_verify() {
        run_commit_open_verify::<_, Pcs, Blake2sTranscript<_>>();
    }

    #[test]
    fn batch_commit_open_verify() {
        run_batch_commit_open_verify::<_, Pcs, Blake2sTranscript<_>>();
    }

    #[test]
    fn point_split_matches_row_major_mle() {
        let num_rows = 4usize;
        let row_len = 8usize;
        let evals = (0..num_rows * row_len)
            .map(|i| Fr::from((i + 1) as u64))
            .collect::<Vec<_>>();
        let point = vec![
            Fr::from(2u64),
            Fr::from(3u64),
            Fr::from(5u64),
            Fr::from(7u64),
            Fr::from(11u64),
        ];

        let direct =
            MultilinearPolynomial::new(evals.clone()).evaluate(&point);
        let (row_weights, column_weights) =
            point_to_tensor(num_rows, &point);

        let mut combined_row = vec![Fr::ZERO; row_len];
        combine_message_rows(
            &evals,
            num_rows,
            row_len,
            &row_weights,
            &mut combined_row,
        )
        .unwrap();

        assert_eq!(
            direct,
            inner_product(&combined_row, &column_weights),
        );
    }

    #[test]
    fn full_open_rejects_wrong_claimed_value() {
        type TestTranscript =
            Blake2sTranscript<Cursor<Vec<u8>>>;

        let poly_size = 1usize << 8;
        let mut rng = ChaCha8Rng::from_seed([42u8; 32]);
        let param = Pcs::setup(poly_size, 1, &mut rng).unwrap();
        let (pp, _) = Pcs::trim(&param, poly_size, 1).unwrap();

        let poly = MultilinearPolynomial::new(
            (0..poly_size)
                .map(|_| Fr::random(&mut rng))
                .collect(),
        );
        let comm = Pcs::commit(&pp, &poly).unwrap();
        let point = (0..pp.num_vars())
            .map(|_| Fr::random(&mut rng))
            .collect::<Vec<_>>();
        let actual = poly.evaluate(&point);
        let wrong = actual + Fr::ONE;

        let mut transcript = TestTranscript::new(());
        let result = Pcs::open(
            &pp,
            &poly,
            &comm,
            &point,
            &wrong,
            &mut transcript,
        );
        assert!(result.is_err());
    }


    #[test]
    fn shape_uses_rate_half() {
        let shape = choose_qapcs_shape::<QAPCSSpecRateHalf100>(20, 127);
        assert_eq!(shape.codeword_len, 2 * shape.row_len);
        assert_eq!(QAPCSSpecRateHalf100::inverse_rate(), 2);
        assert_eq!(QAPCSSpecRateHalf100::security_bits(), 100);
    }
}
