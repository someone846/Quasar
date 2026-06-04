#![allow(warnings, unused)]

use crate::{
    Error,
    pcs::{Evaluation, Point, PolynomialCommitmentScheme},
    pcs::multilinear::{
        Basefold, BasefoldCommitment, BasefoldExtParams, BasefoldParams,
        BasefoldProverParams, BasefoldVerifierParams,
    },
    poly::multilinear::MultilinearPolynomial,
    util::{
        arithmetic::{Field, PrimeField},
        hash::{Hash, Output, Update},
        transcript::{TranscriptRead, TranscriptWrite},
        Deserialize, DeserializeOwned, Serialize,
    },
};

use core::fmt::Debug;
use rand_chacha::{
    rand_core::{RngCore, SeedableRng},
    ChaCha8Rng,
};
use rayon::prelude::*;
use std::{slice, time::Instant};

use crate::piop::sum_check::{
    classic::{ClassicSumCheck, EvaluationsProver},
    SumCheck as _, VirtualPolynomial,
};
use crate::util::expression::{Expression, Query, Rotation};

pub type CommitmentChunk<H> = Output<H>;

const QABASE_BASEFOLD_RATE: usize = 1;


/// QABase PCS prototype.
/// 
/// The committed object is a matrix with `num_rows` rows and `2^num_vars`
/// columns. Each row is encoded by the QA code
///
///   m -> (m, WHT(E_0 * WHT(m)), ..., WHT(E_{c-2} * WHT(m))).
///
/// The prover commits to the row-wise QA codewords using a Merkle tree whose
/// leaves are columns. During opening, the verifier samples several codeword
/// columns, checks their Merkle authentication paths, and folds the opened
/// columns using random row challenges.
///
/// The folded row is then checked as a valid QA codeword via:
///
/// 1. batched WHT sumcheck relations;
/// 2. one scaling sumcheck relation;
/// 3. one selector-sumcheck for Merkle-opened column consistency;
/// 4. one global batched BaseFold opening for all remaining MLE claims.
///
/// Implementation note: the commitment path stores only the QA codeword, not
/// the full per-row `QAWitness`. The full witness is recomputed only for the
/// single folded row during the opening protocol.


// -----------------------------------------------------------------------------
// Security parameter selection
// -----------------------------------------------------------------------------


/// Security configuration for choosing QABase Merkle consistency queries.
///
/// This computes:
///
///   1. the QA row length N_QA = 2^(total_log_size - log_rows);
///   2. a lower bound delta on the QA relative distance;
///   3. the number of opened columns / Merkle queries
///
///          t = ceil(lambda / -log2(1 - delta / 3)).
///
/// This follows the Brakedown/QA-PCS soundness shape:
///
///   (1 - delta / 3)^t + (1 - 2 delta / 3)^t.
///
/// The first term dominates, so we use delta / 3.
#[derive(Clone, Debug)]
pub struct QABaseSecurityConfig {
    /// Total input size exponent.
    ///
    /// For total size 2^20, set total_log_size = 20.
    pub total_log_size: usize,

    /// Number of matrix rows is 2^log_rows.
    ///
    /// In our experiment, log_rows = 6, i.e. 64 rows.
    pub log_rows: usize,

    /// QA inverse rate c.
    ///
    /// Current implementation requires c to be a power of two, e.g. c = 2, 4, 8.
    pub inverse_rate: usize,

    /// Field size in bits.
    ///
    /// For F_{2^127 - 1}, use field_bits = 127.
    pub field_bits: usize,

    /// Target query soundness in bits.
    ///
    /// Our target is 100.
    pub security_bits: usize,

    /// Target failure probability for the random QA code distance bound.
    ///
    /// Usually set this equal to security_bits.
    pub distance_failure_bits: usize,
}

impl QABaseSecurityConfig {
    /// Row length exponent.
    ///
    /// If total size is 2^K and rows are 2^6, then each row has length 2^(K-6).
    pub fn row_log_size(&self) -> usize {
        assert!(
            self.total_log_size >= self.log_rows,
            "total_log_size must be at least log_rows"
        );
        self.total_log_size - self.log_rows
    }

    pub fn num_rows(&self) -> usize {
        1usize << self.log_rows
    }

    pub fn row_size(&self) -> usize {
        1usize << self.row_log_size()
    }

    /// Compute the QA relative distance lower bound delta.
    pub fn distance(&self) -> f64 {
        qabase_distance_lower_bound(
            self.row_log_size(),
            self.inverse_rate,
            self.field_bits,
            self.distance_failure_bits,
        )
    }

    /// Compute the number of opened columns / Merkle queries.
    pub fn num_queries(&self) -> usize {
        qabase_queries_from_distance(self.distance(), self.security_bits)
    }
}

/// p-ary GV-style exponent:
///
///   g_p(delta)
///     = 1 - delta log_p(p - 1)
///       + delta log_p(delta)
///       + (1 - delta) log_p(1 - delta).
///
/// For p = 2^field_bits - 1, we approximate log_p(p - 1) by 1.
/// This is accurate enough for parameter selection, and matches Table 2 well.
pub fn qabase_gp(delta: f64, field_bits: usize) -> f64 {
    assert!(delta > 0.0 && delta < 1.0);

    let bits = field_bits as f64;

    1.0
        - delta
        + (delta * delta.log2()
            + (1.0 - delta) * (1.0 - delta).log2())
            / bits
}

/// log2(2^a + 2^b), computed stably.
fn log2_add(a: f64, b: f64) -> f64 {
    let m = a.max(b);
    if !m.is_finite() {
        return m;
    }
    m + ((2.0f64).powf(a - m) + (2.0f64).powf(b - m)).log2()
}

/// Compute log2 of the Corollary 3.19 failure bound.
///
/// For G = Z_2^n, N = 2^row_log_size, index c, rate 1/c:
///
///   epsilon = g_p(delta) - (1 + log_p N) / c.
///
/// Corollary 3.19 gives two bounds and takes the better one.
/// We return log2(min(bound1, bound2)).
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

    let log_bound2 = log2_add(log_term1_a, log_term1_b)
        .min(log2_add(log_term2_a, log_term2_b));

    log_bound1.min(log_bound2)
}


/// Find the largest delta such that the QA code distance failure probability
/// is at most 2^{-failure_bits}.
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
            qabase_distance_failure_log2(
                mid,
                row_log_size,
                inverse_rate,
                field_bits,
            );

        if failure_log2 <= target_log2 {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    lo
}

/// Compute opened columns / Merkle queries from distance delta.
///
/// We use:
///
///   t = ceil(lambda / -log2(1 - delta/3)).
pub fn qabase_queries_from_distance(delta: f64, security_bits: usize) -> usize {
    assert!(delta > 0.0 && delta < 1.0);

    let effective = delta / 3.0;
    let denom = -(1.0 - effective).log2();

    ((security_bits as f64) / denom).ceil() as usize
}

pub fn setup_from_security_config<F, H>(
    cfg: &QABaseSecurityConfig,
    rng: impl RngCore,
) -> QABaseParams<F>
where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
{
    let poly_size = cfg.row_size();
    let num_rows = cfg.num_rows();
    let inverse_rate = cfg.inverse_rate;
    let num_queries = cfg.num_queries();

    println!(
        "QABase security config: total_size=2^{}, row_size=2^{}, rows={}, c={}, field_bits={}, security_bits={}, delta={:.6}, queries={}",
        cfg.total_log_size,
        cfg.row_log_size(),
        num_rows,
        inverse_rate,
        cfg.field_bits,
        cfg.security_bits,
        cfg.distance(),
        num_queries,
    );

    setup::<F, H>(
        poly_size,
        1,
        rng,
        Some(num_rows),
        Some(inverse_rate),
        Some(num_queries),
    )
}




// -----------------------------------------------------------------------------
// BaseFold configuration
// -----------------------------------------------------------------------------
#[derive(Clone, Copy, Debug)]
pub struct QABaseFoldConfig;

impl BasefoldExtParams for QABaseFoldConfig {
    fn get_reps() -> usize {
        // The current BaseFold verifier implementation assumes one query.
        //402
        241 // 241 for 100 bits soundness
    }

    fn get_rate() -> usize {
        QABASE_BASEFOLD_RATE
    }

    fn get_basecode_rounds() -> usize {
        0
    }

    fn get_rs_basecode() -> bool {
        false
    }

    fn get_code_type() -> String {
        "random".to_string()
    }
}

// -----------------------------------------------------------------------------
// QA code and encoding
// -----------------------------------------------------------------------------

//////////////////////////////////////////////////////////////////////////////
// QA encoding
//////////////////////////////////////////////////////////////////////////////

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
        assert!(msg_len.is_power_of_two(), "QA message length must be a power of two");
        assert!(inverse_rate >= 2, "inverse_rate must be at least 2");
        assert!(inverse_rate.is_power_of_two(), "inverse_rate must be a power of two");
        // c = 2 , 4

        let e = (0..inverse_rate - 1)
            .map(|_| (0..msg_len).map(|_| F::random(&mut *rng)).collect::<Vec<F>>())
            .collect::<Vec<Vec<F>>>();

        Self { inverse_rate, e }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: DeserializeOwned"))]
pub struct QAWitness<F: PrimeField> {
    pub msg: Vec<F>,
    pub msg_wht: Vec<F>,
    pub scaled_wht_blocks: Vec<Vec<F>>,
    pub parity_blocks: Vec<Vec<F>>,
    pub codeword: Vec<F>,
}

/// In-place Walsh--Hadamard transform.
///
/// The transform is not normalized: applying it twice returns `len * x`.
/// The concrete QA encoder only uses the linear transform relation, so this
/// convention is sufficient and avoids divisions.
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

/// Parallel in-place Walsh--Hadamard transform.
///
/// This is intended for a single large vector, most notably the folded row
/// witness created during opening. For row-wise commitment we already
/// parallelize over rows, so nested parallelism is usually not beneficial.
pub fn wht_parallel<F>(x: &mut [F])
where
    F: Field + Send + Sync,
{
    let len = x.len();
    assert!(len.is_power_of_two(), "WHT length must be a power of two");

    let num_threads = rayon::current_num_threads();
    let mut step = 1usize;

    while 2 * step <= len {
        let chunk_len = 2 * step;
        let num_chunks = len / chunk_len;

        if num_chunks >= 4 * num_threads {
            // Many independent chunks: parallelize across chunks.
            x.par_chunks_mut(chunk_len).for_each(|chunk| {
                let (left, right) = chunk.split_at_mut(step);
                for j in 0..step {
                    let u = left[j];
                    let v = right[j];
                    left[j] = u + v;
                    right[j] = u - v;
                }
            });
        } else if step >= 1024 {
            // Few large chunks: parallelize inside each butterfly layer.
            //
            // This matters for the last several WHT layers, where there are too few
            // chunks to occupy all threads.
            x.chunks_mut(chunk_len).for_each(|chunk| {
                let (left, right) = chunk.split_at_mut(step);
                left.par_iter_mut()
                    .zip(right.par_iter_mut())
                    .for_each(|(l, r)| {
                        let u = *l;
                        let v = *r;
                        *l = u + v;
                        *r = u - v;
                    });
            });
        } else {
            // Small layers: serial is often faster than spawning parallel work.
            for chunk in x.chunks_mut(chunk_len) {
                let (left, right) = chunk.split_at_mut(step);
                for j in 0..step {
                    let u = left[j];
                    let v = right[j];
                    left[j] = u + v;
                    right[j] = u - v;
                }
            }
        }

        step <<= 1;
    }
}

/// Full QA encoding with all intermediate witnesses.
///
/// For a message `m`, this computes:
///
///   v'      = WHT(m),
///   u'_i    = E_i ⊙ v',
///   u^{i+1} = WHT(u'_i),
///
/// and returns `codeword = m || u^1 || ... || u^{rho-1}`.
///
/// This function is used for the single folded row during opening, because the
/// WHT and scaling relations need access to `v'`, `u'_i`, and `u^{i+1}`.
pub fn qa_encode_with_witness<F: PrimeField>(msg: &[F], params: &QAParams<F>) -> QAWitness<F> {
    let n = msg.len();
    let rho = params.inverse_rate;

    assert!(n.is_power_of_two(), "message length must be a power of two");
    assert!(rho >= 2, "inverse_rate must be at least 2");
    assert!(rho.is_power_of_two(), "inverse_rate must be a power of two");
    assert_eq!(params.e.len(), rho - 1, "QA encoding needs rho - 1 coefficient vectors");
    for coeffs in &params.e {
        assert_eq!(coeffs.len(), n, "each QA coefficient vector must have length msg.len()");
    }

    let mut msg_wht = msg.to_vec();
    wht(&mut msg_wht);

    let mut scaled_wht_blocks = Vec::with_capacity(rho - 1);
    let mut parity_blocks = Vec::with_capacity(rho - 1);
    let mut codeword = Vec::with_capacity(rho * n);

    codeword.extend_from_slice(msg);

    for i in 0..(rho - 1) {
        let mut scaled_wht = msg_wht.clone();
        for j in 0..n {
            scaled_wht[j] *= params.e[i][j];
        }

        let mut parity = scaled_wht.clone();
        wht(&mut parity);

        codeword.extend_from_slice(&parity);
        scaled_wht_blocks.push(scaled_wht);
        parity_blocks.push(parity);
    }

    QAWitness {
        msg: msg.to_vec(),
        msg_wht,
        scaled_wht_blocks,
        parity_blocks,
        codeword,
    }
}

/// QA encoding for commitment only.
///
/// This computes only the final QA codeword:
///
///   codeword = m || WHT(E_0 * WHT(m)) || ... || WHT(E_{rho-2} * WHT(m)).
///
/// It deliberately does not store `msg_wht`, `scaled_wht_blocks`, or
/// `parity_blocks`. This avoids large allocations and clones during
/// commitment, where only the final codeword is needed for Merkle column
/// openings.
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

    // Compute v' = WHT(m).
    let mut msg_wht = msg.to_vec();
    wht(&mut msg_wht);

    // Directly write:
    //
    //   codeword = m || WHT(E_0 * v') || ... || WHT(E_{rho-2} * v').
    //
    // Avoid `vec![F::ZERO; rho * n]`, which zero-initializes the full
    // codeword. Also move `msg_wht` into the last parity block instead of
    // cloning it. For rho = 2 this saves the only length-N clone on the parity
    // path.
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


/// Parallel full QA encoding for one large folded row.
///
/// This is intended for opening, not row-wise commitment. During commitment we
/// parallelize over rows; for the single folded row we instead parallelize the
/// WHT and per-block scaling work.
pub fn qa_encode_with_witness_parallel<F>(
    msg: &[F],
    params: &QAParams<F>,
) -> QAWitness<F>
where
    F: PrimeField + Send + Sync,
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

    // v' = WHT(m)
    let mut msg_wht = msg.to_vec();
    wht_parallel(&mut msg_wht);

    // For each parity block:
    //
    //   u'_i      = E_i ⊙ v'
    //   u^{i+1}  = WHT(u'_i)
    //
    // Since rho = 4 in your benchmark, this creates 3 independent large tasks.
    let pairs = params
        .e
        .par_iter()
        .map(|coeffs| {
            let mut scaled_wht = msg_wht.clone();

            scaled_wht
                .par_iter_mut()
                .zip(coeffs.par_iter())
                .for_each(|(x, e)| {
                    *x *= *e;
                });

            let mut parity = scaled_wht.clone();
            wht_parallel(&mut parity);

            (scaled_wht, parity)
        })
        .collect::<Vec<_>>();

    let mut scaled_wht_blocks = Vec::with_capacity(rho - 1);
    let mut parity_blocks = Vec::with_capacity(rho - 1);

    for (scaled_wht, parity) in pairs {
        scaled_wht_blocks.push(scaled_wht);
        parity_blocks.push(parity);
    }

    let mut codeword = Vec::with_capacity(rho * n);
    codeword.extend_from_slice(msg);
    for parity in &parity_blocks {
        codeword.extend_from_slice(parity);
    }

    QAWitness {
        msg: msg.to_vec(),
        msg_wht,
        scaled_wht_blocks,
        parity_blocks,
        codeword,
    }
}

#[inline]
pub fn qa_message_block_index() -> usize {
    0
}

#[inline]
pub fn qa_parity_block_index(i: usize) -> usize {
    i + 1
}

#[inline]
pub fn qa_codeword_block<F>(codeword: &[F], block_len: usize, block_index: usize) -> &[F] {
    let start = block_index * block_len;
    let end = start + block_len;
    &codeword[start..end]
}

// -----------------------------------------------------------------------------
// Merkle helpers
// -----------------------------------------------------------------------------

//////////////////////////////////////////////////////////////////////////////
// Merkle helpers
//////////////////////////////////////////////////////////////////////////////

fn hash_field_slice<H, F>(values: &[F]) -> Output<H>
where
    H: Hash,
    F: PrimeField,
{
    let mut hasher = H::new();
    for value in values {
        hasher.update_field_element(value);
    }
    hasher.finalize_fixed()
}

fn hash_hash_pair<H>(left: &Output<H>, right: &Output<H>) -> Output<H>
where
    H: Hash,
{
    let mut hasher = H::new();
    hasher.update(left.as_ref());
    hasher.update(right.as_ref());
    hasher.finalize_fixed()
}

pub fn merkelize_long<H, F>(codeword: &Vec<Vec<F>>) -> Vec<Vec<Output<H>>>
where
    H: Hash,
    F: PrimeField,
{
    assert!(!codeword.is_empty(), "cannot Merkle-commit an empty codeword");

    let num_rows = codeword.len();
    let num_cols = codeword[0].len();
    assert!(num_cols.is_power_of_two(), "Merkle leaf count must be a power of two");

    for row in codeword {
        assert_eq!(row.len(), num_cols, "all codeword rows must have the same length");
    }

    let leaves = (0..num_cols)
        .into_par_iter()
        .map(|col| {
            let mut column = Vec::with_capacity(num_rows);
            for row in 0..num_rows {
                column.push(codeword[row][col]);
            }
            hash_field_slice::<H, F>(&column)
        })
        .collect::<Vec<_>>();

    let mut tree = Vec::new();
    tree.push(leaves);

    while tree.last().unwrap().len() > 1 {
        let prev = tree.last().unwrap();
        assert!(prev.len() % 2 == 0, "each Merkle layer must have even length");
        let next = prev
            .par_chunks_exact(2)
            .map(|pair| hash_hash_pair::<H>(&pair[0], &pair[1]))
            .collect::<Vec<_>>();
        tree.push(next);
    }

    tree
}

pub fn write_merkle_path<H, F>(
    tree: &Vec<Vec<Output<H>>>,
    index: usize,
    transcript: &mut impl TranscriptWrite<CommitmentChunk<H>, F>,
) where
    H: Hash,
    F: PrimeField,
{
    assert!(!tree.is_empty(), "empty Merkle tree");
    assert!(index < tree[0].len(), "Merkle query index out of range");

    let mut idx = index;
    for level in 0..(tree.len() - 1) {
        let nodes = &tree[level];
        let sibling_idx = if idx % 2 == 0 { idx + 1 } else { idx - 1 };
        transcript
            .write_commitment(&nodes[sibling_idx])
            .expect("failed to write Merkle path node to transcript");
        idx >>= 1;
    }
}

pub fn read_merkle_path<H, F>(
    num_leaves: usize,
    transcript: &mut impl TranscriptRead<CommitmentChunk<H>, F>,
) -> Vec<Output<H>>
where
    H: Hash,
    F: PrimeField,
{
    assert!(num_leaves.is_power_of_two(), "number of Merkle leaves must be a power of two");
    let depth = num_leaves.trailing_zeros() as usize;
    (0..depth)
        .map(|_| transcript.read_commitment().expect("failed to read Merkle path node"))
        .collect::<Vec<_>>()
}

pub fn authenticate_merkle_path<H>(leaf: &Output<H>, path: &[Output<H>], index: usize) -> Output<H>
where
    H: Hash,
{
    let mut idx = index;
    let mut cur = leaf.clone();

    for sibling in path {
        cur = if idx % 2 == 0 {
            hash_hash_pair::<H>(&cur, sibling)
        } else {
            hash_hash_pair::<H>(sibling, &cur)
        };
        idx >>= 1;
    }

    cur
}

pub fn verify_merkle_path<H>(
    root: &Output<H>,
    leaf: &Output<H>,
    path: &[Output<H>],
    index: usize,
) -> bool
where
    H: Hash,
{
    &authenticate_merkle_path::<H>(leaf, path, index) == root
}

pub fn verify_merkle_path_with_leaf<H, F>(
    root: &Output<H>,
    opened_column: &[F],
    path: &[Output<H>],
    index: usize,
) -> bool
where
    H: Hash,
    F: PrimeField,
{
    let leaf = hash_field_slice::<H, F>(opened_column);
    verify_merkle_path::<H>(root, &leaf, path, index)
}

fn field_challenge_to_index<F>(challenge: &F, num_cols: usize) -> usize
where
    F: PrimeField,
{
    assert!(num_cols > 0, "number of columns must be positive");

    let repr = challenge.to_repr();
    let bytes = repr.as_ref();
    let mut acc: usize = 0;
    let take = core::cmp::min(bytes.len(), core::mem::size_of::<usize>());

    for i in 0..take {
        acc |= (bytes[i] as usize) << (8 * i);
    }

    acc % num_cols
}

// -----------------------------------------------------------------------------
// QABase parameters, commitments, setup, trim, commit
// -----------------------------------------------------------------------------

//////////////////////////////////////////////////////////////////////////////
// QABase parameters and commitment
//////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: DeserializeOwned"))]
pub struct QABaseParams<F: PrimeField> {
    pub qa_params: QAParams<F>,
    pub num_vars: usize,
    pub num_rows: usize,
    pub inverse_rate: usize,
    pub num_queries: usize,
    pub basefold_params: BasefoldParams<F>,
    pub rng: ChaCha8Rng,
}

#[derive(Clone, Debug)]
pub struct QABaseProverParams<F: PrimeField, H: Hash> {
    pub qa_params: QAParams<F>,
    pub num_vars: usize,
    pub num_rows: usize,
    pub inverse_rate: usize,
    pub num_queries: usize,
    pub basefold_prover_param: BasefoldProverParams<F>,

    /// Preprocessed public polynomials for the random QA coefficients E_i.
    ///
    /// These are indexed once during `trim` and are not part of the online
    /// prover time. During proving we only use them to open E_i(r), where r is
    /// the scaling-relation sumcheck point.
    pub e_polys: Vec<MultilinearPolynomial<F>>,

    /// BaseFold commitments to the public E_i polynomials.
    ///
    /// These belong to the verifier key / indexing output. They are not written
    /// during the QABase auxiliary-commitment phase.
    pub e_commitments: Vec<BasefoldCommitment<F, H>>,
}

#[derive(Clone, Debug)]
pub struct QABaseVerifierParams<F: PrimeField, H: Hash> {
    pub qa_params: QAParams<F>,
    pub num_vars: usize,
    pub num_rows: usize,
    pub inverse_rate: usize,
    pub num_queries: usize,
    pub basefold_verifier_param: BasefoldVerifierParams<F>,

    /// Public BaseFold commitments to E_i, generated during indexing / trim.
    pub e_commitments: Vec<BasefoldCommitment<F, H>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: DeserializeOwned"))]
pub struct QABaseCommitment<F, H>
where
    F: PrimeField,
    H: Hash,
{
    /// Row-wise QA codewords.
    ///
    /// `codeword[row][col]` is the `col`-th QA codeword coordinate of the
    /// `row`-th original message row. This is retained because Item 1 later
    /// opens random columns.
    pub codeword: Vec<Vec<F>>,

    /// Merkle tree over codeword columns.
    ///
    /// Each leaf is `Hash(codeword[0][col], ..., codeword[num_rows-1][col])`.
    pub codeword_tree: Vec<Vec<Output<H>>>,

    /// Deprecated compatibility field.
    ///
    /// The online commitment no longer stores a copy of the original matrix.
    /// Opening receives the original matrix rows explicitly on the prover side
    /// and uses them to construct the folded witness.
    pub bh_evals: Vec<Vec<F>>,

    /// Compatibility field.
    ///
    /// Commitment no longer stores full per-row QA witnesses. The folded
    /// witness is recomputed from the folded message during opening.
    pub qa_witnesses: Vec<QAWitness<F>>,
}

impl<F, H> Default for QABaseCommitment<F, H>
where
    F: PrimeField,
    H: Hash,
{
    fn default() -> Self {
        Self {
            codeword: Vec::new(),
            codeword_tree: vec![vec![Output::<H>::default()]],
            bh_evals: Vec::new(),
            qa_witnesses: Vec::new(),
        }
    }
}

impl<F, H> AsRef<[Output<H>]> for QABaseCommitment<F, H>
where
    F: PrimeField,
    H: Hash,
{
    fn as_ref(&self) -> &[Output<H>] {
        if self.codeword_tree.is_empty() {
            &[]
        } else {
            let root = &self.codeword_tree[self.codeword_tree.len() - 1][0];
            slice::from_ref(root)
        }
    }
}

impl<F, H> AsRef<Output<H>> for QABaseCommitment<F, H>
where
    F: PrimeField,
    H: Hash,
{
    fn as_ref(&self) -> &Output<H> {
        &self.codeword_tree[self.codeword_tree.len() - 1][0]
    }
}

//////////////////////////////////////////////////////////////////////////////
// Setup / trim / commit
//////////////////////////////////////////////////////////////////////////////

pub fn setup<F, H>(
    poly_size: usize,
    _batch_size: usize,
    _rng: impl RngCore,
    num_rows: Option<usize>,
    inverse_rate: Option<usize>,
    num_queries: Option<usize>,
) -> QABaseParams<F>
where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
{
    assert!(poly_size.is_power_of_two(), "poly_size must be a power of two");
    type Pcs<F, H> = Basefold<F, H, QABaseFoldConfig>;

    let num_queries = num_queries.unwrap_or(504);
    let num_rows = num_rows.unwrap_or(64);
    let inverse_rate = inverse_rate.unwrap_or(2);
    assert!(inverse_rate.is_power_of_two(), "inverse_rate must be a power of two");

    let num_vars = poly_size.trailing_zeros() as usize;
    let mut rng = ChaCha8Rng::from_entropy();
    let qa_params = QAParams::<F>::new_random(poly_size, inverse_rate, &mut rng);
    let basefold_params = Pcs::<F, H>::setup(poly_size, 1, &mut rng).unwrap();

    QABaseParams {
        qa_params,
        num_vars,
        num_rows,
        inverse_rate,
        num_queries,
        basefold_params,
        rng,
    }
}

pub fn trim<F, H>(
    param: &QABaseParams<F>,
    poly_size: usize,
    batch_size: usize,
) -> (QABaseProverParams<F, H>, QABaseVerifierParams<F, H>)
where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
{
    type Pcs<F, H> = Basefold<F, H, QABaseFoldConfig>;

    let (basefold_pp, basefold_vp) =
        Pcs::<F, H>::trim(&param.basefold_params, poly_size, batch_size).unwrap();

    // Indexing/preprocessing for public QA coefficient polynomials E_i.
    //
    // These commitments are part of the public/verifier key. They should not
    // be counted as online proving time or proof size. The online proof still
    // includes E_i(r) values and their BaseFold opening claims.
    let e_polys = param
        .qa_params
        .e
        .iter()
        .map(|e_i| vec_to_mle(e_i.clone()))
        .collect::<Vec<_>>();

    let now = Instant::now();
    let e_commitments =
        Pcs::<F, H>::batch_commit(&basefold_pp, e_polys.iter())
            .expect("failed to commit public QA coefficient polynomials E_i");
    println!(
        "qabase indexing public E commitments {:?}, blocks {}",
        now.elapsed(),
        e_commitments.len()
    );

    (
        QABaseProverParams {
            qa_params: param.qa_params.clone(),
            num_vars: param.num_vars,
            num_rows: param.num_rows,
            inverse_rate: param.inverse_rate,
            num_queries: param.num_queries,
            basefold_prover_param: basefold_pp,
            e_polys,
            e_commitments: e_commitments.clone(),
        },
        QABaseVerifierParams {
            qa_params: param.qa_params.clone(),
            num_vars: param.num_vars,
            num_rows: param.num_rows,
            inverse_rate: param.inverse_rate,
            num_queries: param.num_queries,
            basefold_verifier_param: basefold_vp,
            e_commitments,
        },
    )
}


/// Commit to a matrix using row-wise QA encoding and a column Merkle tree.
///
/// This uses `qa_encode_codeword_only` instead of `qa_encode_with_witness`,
/// because per-row WHT/scaling witnesses are not needed after commitment.
/// The function still stores the final QA codewords for later Merkle column
/// openings and the original matrix rows for folded-witness construction.
pub fn commit_and_write<F, H>(
    pp: &QABaseProverParams<F, H>,
    word: &Vec<Vec<F>>,
    transcript: &mut impl TranscriptWrite<CommitmentChunk<H>, F>,
) -> QABaseCommitment<F, H>
where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
{
    assert_eq!(
        word.len(),
        pp.num_rows,
        "matrix row count does not match pp.num_rows"
    );

    let now = Instant::now();

    let codeword = word
        .par_iter()
        .map(|row| qa_encode_codeword_only(row, &pp.qa_params))
        .collect::<Vec<_>>();

    println!(
        "degree {:?}, qa codeword-only encode time {:?}",
        pp.num_vars,
        now.elapsed()
    );

    let now = Instant::now();
    let tree = merkelize_long::<H, F>(&codeword);
    println!("degree {:?}, qa merkle time {:?}", pp.num_vars, now.elapsed());

    transcript
        .write_commitment(&tree[tree.len() - 1][0])
        .expect("failed to write QABase Merkle root");

    println!("one qabase commitment written");

    QABaseCommitment {
        codeword,
        codeword_tree: tree,

        // Do not clone the original matrix into the commitment. The prover-side
        // opening API receives `word` explicitly and folds it there.
        bh_evals: Vec::new(),

        // No longer needed for the committed rows.
        // The folded witness is recomputed separately during opening.
        qa_witnesses: Vec::new(),
    }
}

// -----------------------------------------------------------------------------
// Generic polynomial and MLE helpers
// -----------------------------------------------------------------------------

//////////////////////////////////////////////////////////////////////////////
// Generic polynomial and MLE helpers
//////////////////////////////////////////////////////////////////////////////

fn log2_power_of_two(x: usize) -> usize {
    assert!(x.is_power_of_two(), "input must be a power of two");
    x.trailing_zeros() as usize
}

/// Evaluate the Boolean equality polynomial `eq(<index>, point)`.
///
/// The Boolean encoding of `index` uses the same little-endian convention as
/// `index_to_boolean_point`: bit `i` of `index` corresponds to coordinate
/// `point[i]`.
pub fn equality_mle_eval_at_index<F>(
    index: usize,
    num_vars: usize,
    point: &[F],
) -> F
where
    F: PrimeField,
{
    assert_eq!(
        point.len(),
        num_vars,
        "point length does not match num_vars"
    );
    assert!(
        index < (1usize << num_vars),
        "Boolean index out of range"
    );

    let mut acc = F::ONE;
    for i in 0..num_vars {
        acc *= if ((index >> i) & 1) == 1 {
            point[i]
        } else {
            F::ONE - point[i]
        };
    }
    acc
}

/// Build the sparse selector h over the full QA codeword domain of size c * N.
///
/// If the sampled indices contain duplicates, the weights are added.

pub fn hadamard_tensor_mle_eval<F>(a: &[F], b: &[F]) -> F
where
    F: PrimeField,
{
    assert_eq!(a.len(), b.len(), "Hadamard tensor MLE input lengths must match");
    let two = F::from(2u64);
    let mut acc = F::ONE;
    for (a_i, b_i) in a.iter().zip(b.iter()) {
        acc *= F::ONE - two * (*a_i) * (*b_i);
    }
    acc
}

pub fn hadamard_tensor_mle_evals_on_hypercube<F>(gamma: &[F]) -> Vec<F>
where
    F: PrimeField,
{
    let size = 1usize << gamma.len();
    let two = F::from(2u64);

    let mut evals = Vec::with_capacity(size);
    evals.push(F::ONE);

    for &g in gamma {
        let factor = F::ONE - two * g;
        let old_len = evals.len();
        evals.resize(old_len << 1, F::ZERO);

        // Little-endian order: the new variable is the next high bit.
        // First half corresponds to bit 0 and is unchanged; second half
        // corresponds to bit 1 and is multiplied by (1 - 2 * gamma_i).
        for i in 0..old_len {
            let old = evals[i];
            evals[old_len + i] = old * factor;
        }
    }

    debug_assert_eq!(evals.len(), size);
    evals
}

pub fn eval_mle_from_evals<F>(evals: &[F], point: &[F]) -> F
where
    F: PrimeField,
{
    assert!(evals.len().is_power_of_two(), "MLE eval vector length must be a power of two");
    assert_eq!(evals.len(), 1usize << point.len(), "point length does not match eval length");

    let mut layer = evals.to_vec();
    let mut size = layer.len();
    for r in point {
        let half = size >> 1;
        for i in 0..half {
            let left = layer[2 * i];
            let right = layer[2 * i + 1];
            layer[i] = left * (F::ONE - *r) + right * (*r);
        }
        size = half;
    }
    layer[0]
}

pub fn equality_mle_eval<F>(a: &[F], b: &[F]) -> F
where
    F: PrimeField,
{
    assert_eq!(a.len(), b.len(), "equality polynomial input lengths must match");
    let mut acc = F::ONE;
    for (a_i, b_i) in a.iter().zip(b.iter()) {
        acc *= (F::ONE - *a_i) * (F::ONE - *b_i) + (*a_i) * (*b_i);
    }
    acc
}

pub fn equality_mle_evals_on_hypercube<F>(beta: &[F]) -> Vec<F>
where
    F: PrimeField,
{
    let size = 1usize << beta.len();

    let mut evals = Vec::with_capacity(size);
    evals.push(F::ONE);

    for &r in beta {
        let old_len = evals.len();
        evals.resize(old_len << 1, F::ZERO);

        // Little-endian order: first half has the new bit equal to 0,
        // second half has the new bit equal to 1.
        for i in 0..old_len {
            let old = evals[i];
            evals[i] = old * (F::ONE - r);
            evals[old_len + i] = old * r;
        }
    }

    debug_assert_eq!(evals.len(), size);
    evals
}

pub fn batched_linear_combination_evals<F>(blocks: &[Vec<F>], alpha: F) -> Vec<F>
where
    F: PrimeField,
{
    assert!(!blocks.is_empty(), "cannot batch an empty list of evaluation vectors");
    let n = blocks[0].len();
    for block in blocks {
        assert_eq!(block.len(), n, "all batched eval vectors must have same length");
    }

    let mut out = vec![F::ZERO; n];
    let mut power = F::ONE;
    for block in blocks {
        for j in 0..n {
            out[j] += power * block[j];
        }
        power *= alpha;
    }
    out
}

pub fn index_to_boolean_point<F>(index: usize, num_vars: usize) -> Vec<F>
where
    F: PrimeField,
{
    assert!(index < (1usize << num_vars), "Boolean index out of range");
    (0..num_vars)
        .map(|i| if ((index >> i) & 1) == 1 { F::ONE } else { F::ZERO })
        .collect::<Vec<_>>()
}

pub fn qabase_codeword_column_to_block(
    full_index: usize,
    block_len: usize,
    inverse_rate: usize,
) -> (usize, usize) {
    assert!(block_len > 0, "block length must be positive");
    let num_cols = inverse_rate * block_len;
    assert!(full_index < num_cols, "QA codeword column index out of range");
    let block_index = full_index / block_len;
    let local_index = full_index % block_len;
    assert!(block_index < inverse_rate, "QA block index out of range");
    (block_index, local_index)
}

pub fn fold_opened_column<F>(opened_column: &[F], row_challenges: &[F]) -> F
where
    F: PrimeField,
{
    assert_eq!(opened_column.len(), row_challenges.len(), "opened column length mismatch");
    let mut acc = F::ZERO;
    for (value, challenge) in opened_column.iter().zip(row_challenges.iter()) {
        acc += *challenge * *value;
    }
    acc
}


//////////////////////////////////////////////////////////////////////////////
// Folded witness and auxiliary BaseFold commitments
//////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: DeserializeOwned"))]
pub struct QABaseFoldedWitness<F>
where
    F: PrimeField,
{
    pub row_challenges: Vec<F>,
    pub folded_msg: Vec<F>,
    pub folded_qa_witness: QAWitness<F>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: DeserializeOwned"))]
pub struct QABaseAuxCommitments<F, H>
where
    F: PrimeField,
    H: Hash,
{
    pub u_block_commitments: Vec<BasefoldCommitment<F, H>>,
    pub v_prime_commitment: BasefoldCommitment<F, H>,
    pub u_prime_commitments: Vec<BasefoldCommitment<F, H>>,
}

/// Prover-side auxiliary data for the folded QA witness.
///
/// The `polys` vector contains the BaseFold polynomials for
///
///   u^{(0)}, ..., u^{(rho-1)}, v', u'_0, ..., u'_{rho-2}.
///
/// It is constructed once and then reused both for auxiliary commitments and
/// for the final BaseFold batch opening. This avoids reconstructing these
/// length-N multilinear polynomials from the folded witness a second time.
pub struct QABaseAuxProverData<F, H>
where
    F: PrimeField,
    H: Hash,
{
    pub commitments: QABaseAuxCommitments<F, H>,
    pub polys: Vec<MultilinearPolynomial<F>>,
}

/// Serial row folding helper used mainly by tests.
fn linear_combine_rows<F>(rows: &Vec<Vec<F>>, challenges: &[F]) -> Vec<F>
where
    F: PrimeField,
{
    assert!(!rows.is_empty(), "cannot fold an empty matrix");
    assert_eq!(rows.len(), challenges.len(), "number of row challenges must match rows");

    let row_len = rows[0].len();
    for row in rows {
        assert_eq!(row.len(), row_len, "all rows must have the same length");
    }

    let mut out = vec![F::ZERO; row_len];
    for (challenge, row) in challenges.iter().zip(rows.iter()) {
        for i in 0..row_len {
            out[i] += *challenge * row[i];
        }
    }
    out
}

/// Parallel row folding helper for the prover's folded witness construction.
///
/// For each column index `j`, compute `sum_i challenges[i] * rows[i][j]`.
fn linear_combine_rows_parallel<F>(rows: &Vec<Vec<F>>, challenges: &[F]) -> Vec<F>
where
    F: PrimeField + Send + Sync,
{
    assert!(!rows.is_empty(), "cannot fold an empty matrix");
    assert_eq!(
        rows.len(),
        challenges.len(),
        "number of row challenges must match rows"
    );

    let row_len = rows[0].len();
    for row in rows {
        assert_eq!(
            row.len(),
            row_len,
            "all rows must have the same length"
        );
    }

    (0..row_len)
        .into_par_iter()
        .map(|i| {
            let mut acc = F::ZERO;
            for (challenge, row) in challenges.iter().zip(rows.iter()) {
                acc += *challenge * row[i];
            }
            acc
        })
        .collect()
}

fn vec_to_mle<F>(evals: Vec<F>) -> MultilinearPolynomial<F>
where
    F: PrimeField,
{
    assert!(evals.len().is_power_of_two(), "MLE evaluation length must be a power of two");
    MultilinearPolynomial::new(evals)
}

/// Build the folded QA witness used by the opening protocol.
///
/// The verifier first samples random row challenges `r_i`. The prover folds
/// the committed message matrix into a single row:
///
///   folded_msg = sum_i r_i * row_i.
///
/// Since QA encoding is linear, the folded codeword should equal the same
/// linear combination of the committed QA codeword rows. We recompute the full
/// QA witness only for this folded row, because the encoding-relation
/// sumchecks need access to the intermediate WHT/scaling witnesses.
pub fn build_folded_qa_witness_from_rows<F, H>(
    pp: &QABaseProverParams<F, H>,
    rows: &Vec<Vec<F>>,
    row_challenges: Vec<F>,
) -> QABaseFoldedWitness<F>
where
    F: PrimeField + Serialize + DeserializeOwned + Send + Sync,
    H: Hash,
{
    assert_eq!(
        row_challenges.len(),
        pp.num_rows,
        "row challenge length mismatch"
    );
    assert_eq!(rows.len(), pp.num_rows, "row count mismatch");
    assert!(
        !rows.is_empty(),
        "cannot build a folded witness from an empty matrix"
    );
    assert_eq!(rows[0].len(), 1usize << pp.num_vars, "row length mismatch");

    let folded_msg = linear_combine_rows_parallel(rows, &row_challenges);

    let folded_qa_witness =
        qa_encode_with_witness_parallel(&folded_msg, &pp.qa_params);

    QABaseFoldedWitness {
        row_challenges,
        folded_msg,
        folded_qa_witness,
    }
}

pub fn commit_folded_witness_with_basefold<F, H>(
    pp: &QABaseProverParams<F, H>,
    rows: &Vec<Vec<F>>,
    transcript: &mut impl TranscriptWrite<CommitmentChunk<H>, F>,
) -> Result<(QABaseFoldedWitness<F>, QABaseAuxProverData<F, H>), Error>
where
    F: PrimeField + Serialize + DeserializeOwned + Send + Sync,
    H: Hash,
{
    type Pcs<F, H> = Basefold<F, H, QABaseFoldConfig>;

    assert!(!rows.is_empty(), "cannot fold an empty committed matrix");
    assert_eq!(rows.len(), pp.num_rows, "row count mismatch");
    assert_eq!(rows[0].len(), 1usize << pp.num_vars, "row length mismatch");

    let now = Instant::now();
    let row_challenges = transcript.squeeze_challenges(pp.num_rows);
    let folded_witness =
        build_folded_qa_witness_from_rows::<F, H>(pp, rows, row_challenges);
    println!("qabase folded witness construction {:?}", now.elapsed());

    let rho = pp.inverse_rate;
    assert_eq!(folded_witness.folded_qa_witness.parity_blocks.len(), rho - 1);
    assert_eq!(folded_witness.folded_qa_witness.scaled_wht_blocks.len(), rho - 1);

    // Construct auxiliary BaseFold polynomials exactly once.
    //
    // Previously the code constructed these polynomials once for committing and
    // then reconstructed them again in `build_qabase_prover_opening_accumulator`.
    // Each construction clones several length-N vectors from the folded witness.
    // We keep the owned polynomials here and later borrow the same objects for
    // the final BaseFold batch opening.
    let aux_polys = build_aux_polys_from_folded_witness::<F>(&folded_witness, rho);
    assert_eq!(aux_polys.len(), 2 * rho);

    // Commit all online auxiliary BaseFold polynomials as one batch.
    //
    // This is important for the optimized BaseFold batch opening: all online
    // auxiliary codewords share a single vector-leaf Merkle root, so later the
    // final global batch opening only needs one authentication path for this
    // whole group at each BaseFold query position.  The previous code committed
    // u-blocks, v_prime, and u_prime blocks in three independent batches, which
    // prevented path sharing across the whole online auxiliary set.
    let now = Instant::now();
    let aux_commitments_all = Pcs::<F, H>::batch_commit_and_write(
        &pp.basefold_prover_param,
        aux_polys.iter(),
        transcript,
    )?;
    assert_eq!(aux_commitments_all.len(), 2 * rho);
    println!(
        "qabase basefold commit all aux polynomials as one batch {:?}, blocks {}",
        now.elapsed(),
        aux_commitments_all.len(),
    );

    let u_block_commitments = aux_commitments_all[0..rho].to_vec();
    let v_prime_commitment = aux_commitments_all[rho].clone();
    let u_prime_commitments = aux_commitments_all[rho + 1..2 * rho].to_vec();

    let commitments = QABaseAuxCommitments {
        u_block_commitments,
        v_prime_commitment,
        u_prime_commitments,
    };

    Ok((
        folded_witness,
        QABaseAuxProverData {
            commitments,
            polys: aux_polys,
        },
    ))
}

pub fn read_aux_commitments_from_transcript<F, H>(
    vp: &QABaseVerifierParams<F, H>,
    comm: &QABaseCommitment<F, H>,
    transcript: &mut impl TranscriptRead<CommitmentChunk<H>, F>,
) -> Result<(Vec<F>, QABaseAuxCommitments<F, H>), Error>
where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
{
    type Pcs<F, H> = Basefold<F, H, QABaseFoldConfig>;

    let root_from_transcript = transcript.read_commitment()?;
    let committed_root = <QABaseCommitment<F, H> as AsRef<Output<H>>>::as_ref(comm);
    assert!(&root_from_transcript == committed_root, "QABase root mismatch");

    let row_challenges = transcript.squeeze_challenges(vp.num_rows);

    // The prover committed all online auxiliary BaseFold polynomials in one
    // batch, so the transcript contains 2*rho individual roots followed by one
    // vector-leaf batch root.  Read them as one batch to preserve the batch_root
    // metadata used by BaseFold::batch_verify.
    let rho = vp.inverse_rate;
    let aux_commitments_all = Pcs::<F, H>::read_commitments(
        &vp.basefold_verifier_param,
        2 * rho,
        transcript,
    )?;
    assert_eq!(aux_commitments_all.len(), 2 * rho);

    let u_block_commitments = aux_commitments_all[0..rho].to_vec();
    let v_prime_commitment = aux_commitments_all[rho].clone();
    let u_prime_commitments = aux_commitments_all[rho + 1..2 * rho].to_vec();

    Ok((
        row_challenges,
        QABaseAuxCommitments {
            u_block_commitments,
            v_prime_commitment,
            u_prime_commitments,
        },
    ))
}

// -----------------------------------------------------------------------------
// Scaling relation sumcheck
// -----------------------------------------------------------------------------

//////////////////////////////////////////////////////////////////////////////
// Scaling relation
//////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: DeserializeOwned"))]
pub struct QABaseScalingRelationProof<F>
where
    F: PrimeField,
{
    pub alpha: F,
    pub beta: Vec<F>,
    pub sc_point: Vec<F>,
    pub v_prime_eval_at_sc_point: F,
    pub u_prime_batch_eval_at_sc_point: F,
    /// Individual public-coefficient evaluations E_i(sc_point).
    ///
    /// The verifier linearly combines these values to obtain
    /// e_batch_eval_at_sc_point and verifies each value via BaseFold opening.
    pub e_evals_at_sc_point: Vec<F>,

    pub e_batch_eval_at_sc_point: F,
    pub eq_eval_at_sc_point: F,
    pub terminal_eval: F,
}

pub fn prove_scaling_relation_sumcheck<F, H>(
    qa_params: &QAParams<F>,
    v_prime_evals: &[F],
    u_prime_evals: &[Vec<F>],
    transcript: &mut impl TranscriptWrite<CommitmentChunk<H>, F>,
) -> Result<QABaseScalingRelationProof<F>, Error>
where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
{
    assert!(v_prime_evals.len().is_power_of_two(), "v_prime length must be a power of two");
    let n = v_prime_evals.len();
    let num_vars = n.trailing_zeros() as usize;
    assert_eq!(qa_params.e.len(), u_prime_evals.len(), "number of E_i vectors mismatch");

    for e_i in &qa_params.e {
        assert_eq!(e_i.len(), n, "each E_i must have length n");
    }
    for u_i in u_prime_evals {
        assert_eq!(u_i.len(), n, "each u_prime_i must have length n");
    }

    type Sc<F> = ClassicSumCheck<EvaluationsProver<F>>;

    let now = Instant::now();
    let alpha = transcript.squeeze_challenges(1)[0];
    let beta = transcript.squeeze_challenges(num_vars);

    let eq_beta_poly =
        MultilinearPolynomial::new(equality_mle_evals_on_hypercube::<F>(&beta));

    let u_prime_batch_evals =
        batched_linear_combination_evals::<F>(u_prime_evals, alpha);

    let e_batch_evals =
        batched_linear_combination_evals::<F>(&qa_params.e, alpha);

    let u_prime_batch_poly = MultilinearPolynomial::new(u_prime_batch_evals);
    let e_batch_poly = MultilinearPolynomial::new(e_batch_evals.clone());
    let v_prime_poly = MultilinearPolynomial::new(v_prime_evals.to_vec());

    let eq_query = Expression::<F>::Polynomial(Query::new(0, Rotation::cur()));
    let u_batch_query = Expression::<F>::Polynomial(Query::new(1, Rotation::cur()));
    let e_batch_query = Expression::<F>::Polynomial(Query::new(2, Rotation::cur()));
    let v_prime_query = Expression::<F>::Polynomial(Query::new(3, Rotation::cur()));

    let expression: Expression<F> =
        eq_query * (u_batch_query - e_batch_query * v_prime_query);

    let polys = vec![eq_beta_poly, u_prime_batch_poly, e_batch_poly, v_prime_poly];
    let challenges: Vec<F> = Vec::new();
    let ys: Vec<Vec<F>> = Vec::new();

    let virtual_poly: VirtualPolynomial<F> =
        VirtualPolynomial::new(&expression, &polys, &challenges, &ys);

    let (sc_point, terminal_evals) =
        Sc::<F>::prove(&(), num_vars, virtual_poly, F::ZERO, transcript)?;
    assert_eq!(terminal_evals.len(), 4);

    let eq_eval_at_sc_point = terminal_evals[0];
    let u_prime_batch_eval_at_sc_point = terminal_evals[1];
    let e_batch_eval_at_sc_point = terminal_evals[2];
    let v_prime_eval_at_sc_point = terminal_evals[3];

    // Witness-side values. These are later checked through BaseFold openings.
    transcript.write_field_element(&v_prime_eval_at_sc_point)?;
    transcript.write_field_element(&u_prime_batch_eval_at_sc_point)?;

    // Public E_i evaluations. The verifier will not recompute them by scanning
    // qa_params.e; instead, it will verify these values via BaseFold openings.
    let mut e_evals_at_sc_point = Vec::with_capacity(qa_params.e.len());
    let mut alpha_power = F::ONE;
    let mut e_batch_check = F::ZERO;

    for e_i in qa_params.e.iter() {
        let value = eval_mle_from_evals::<F>(e_i, &sc_point);
        transcript.write_field_element(&value)?;

        e_batch_check += alpha_power * value;
        alpha_power *= alpha;

        e_evals_at_sc_point.push(value);
    }

    assert_eq!(e_batch_check, e_batch_eval_at_sc_point);

    let terminal_eval =
        eq_eval_at_sc_point
            * (u_prime_batch_eval_at_sc_point
                - e_batch_eval_at_sc_point * v_prime_eval_at_sc_point);

    println!("qabase scaling relation sumcheck prove {:?}", now.elapsed());

    Ok(QABaseScalingRelationProof {
        alpha,
        beta,
        sc_point,
        v_prime_eval_at_sc_point,
        u_prime_batch_eval_at_sc_point,
        e_evals_at_sc_point,
        e_batch_eval_at_sc_point,
        eq_eval_at_sc_point,
        terminal_eval,
    })
}

pub fn verify_scaling_relation_sumcheck<F, H>(
    qa_params: &QAParams<F>,
    transcript: &mut impl TranscriptRead<CommitmentChunk<H>, F>,
) -> Result<(bool, QABaseScalingRelationProof<F>), Error>
where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
{
    assert!(
        !qa_params.e.is_empty(),
        "QAParams must contain at least one E_i vector"
    );

    let n = qa_params.e[0].len();
    assert!(n.is_power_of_two(), "E_i length must be a power of two");
    let num_vars = n.trailing_zeros() as usize;

    type Sc<F> = ClassicSumCheck<EvaluationsProver<F>>;

    let now = Instant::now();
    let alpha = transcript.squeeze_challenges(1)[0];
    let beta = transcript.squeeze_challenges(num_vars);

    let (terminal_eval, sc_point) =
        Sc::<F>::verify(&(), num_vars, 3usize, F::ZERO, transcript)?;

    let v_prime_eval_at_sc_point = transcript.read_field_element()?;
    let u_prime_batch_eval_at_sc_point = transcript.read_field_element()?;

    // Read E_i(sc_point) from transcript. These values are not trusted yet;
    // Item 2 will add BaseFold opening claims for them.
    let mut e_evals_at_sc_point = Vec::with_capacity(qa_params.e.len());
    let mut alpha_power = F::ONE;
    let mut e_batch_eval_at_sc_point = F::ZERO;

    for _ in 0..qa_params.e.len() {
        let value = transcript.read_field_element()?;

        e_batch_eval_at_sc_point += alpha_power * value;
        alpha_power *= alpha;

        e_evals_at_sc_point.push(value);
    }

    let eq_eval_at_sc_point = equality_mle_eval::<F>(&beta, &sc_point);

    let expected_terminal =
        eq_eval_at_sc_point
            * (u_prime_batch_eval_at_sc_point
                - e_batch_eval_at_sc_point * v_prime_eval_at_sc_point);

    let ok = terminal_eval == expected_terminal;

    println!("qabase scaling relation sumcheck verify {:?}", now.elapsed());

    Ok((
        ok,
        QABaseScalingRelationProof {
            alpha,
            beta,
            sc_point: sc_point.to_vec(),
            v_prime_eval_at_sc_point,
            u_prime_batch_eval_at_sc_point,
            e_evals_at_sc_point,
            e_batch_eval_at_sc_point,
            eq_eval_at_sc_point,
            terminal_eval,
        },
    ))
}

// -----------------------------------------------------------------------------
// Batched WHT relation sumcheck
// -----------------------------------------------------------------------------

//////////////////////////////////////////////////////////////////////////////
// Batched WHT relation
//////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: DeserializeOwned"))]
pub struct QABaseBatchedWhtRelationProof<F>
where
    F: PrimeField,
{
    pub eta: F,
    pub gammas: Vec<Vec<F>>,
    pub output_evals_at_gammas: Vec<F>,
    pub sc_point: Vec<F>,
    pub input_evals_at_sc_point: Vec<F>,
    pub scaled_h_evals_at_sc_point: Vec<F>,
    pub terminal_eval: F,
    pub claimed_sum: F,
}

pub fn prove_batched_wht_relations_sumcheck<F, H>(
    input_evals_list: &[Vec<F>],
    output_evals_list: &[Vec<F>],
    transcript: &mut impl TranscriptWrite<CommitmentChunk<H>, F>,
) -> Result<QABaseBatchedWhtRelationProof<F>, Error>
where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
{
    assert!(!input_evals_list.is_empty(), "batched WHT proof needs at least one relation");
    assert_eq!(input_evals_list.len(), output_evals_list.len(), "WHT input/output count mismatch");

    let num_relations = input_evals_list.len();
    let n = input_evals_list[0].len();
    assert!(n.is_power_of_two(), "WHT input length must be a power of two");
    let num_vars = n.trailing_zeros() as usize;
    for input in input_evals_list {
        assert_eq!(input.len(), n, "all WHT inputs must have same length");
    }
    for output in output_evals_list {
        assert_eq!(output.len(), n, "all WHT outputs must have same length");
    }

    type Sc<F> = ClassicSumCheck<EvaluationsProver<F>>;

    let now = Instant::now();
    let eta = transcript.squeeze_challenges(1)[0];
    let mut gammas = Vec::with_capacity(num_relations);
    for _ in 0..num_relations {
        gammas.push(transcript.squeeze_challenges(num_vars));
    }

    let mut output_evals_at_gammas = Vec::with_capacity(num_relations);
    let mut claimed_sum = F::ZERO;
    let mut eta_power = F::ONE;
    for t in 0..num_relations {
        let y_t = eval_mle_from_evals(&output_evals_list[t], &gammas[t]);
        transcript.write_field_element(&y_t)?;
        output_evals_at_gammas.push(y_t);
        claimed_sum += eta_power * y_t;
        eta_power *= eta;
    }

    let mut polys = Vec::with_capacity(2 * num_relations);
    let mut eta_power = F::ONE;
    for t in 0..num_relations {
        let mut h_evals = hadamard_tensor_mle_evals_on_hypercube::<F>(&gammas[t]);
        for value in h_evals.iter_mut() {
            *value *= eta_power;
        }
        polys.push(MultilinearPolynomial::new(h_evals));
        polys.push(MultilinearPolynomial::new(input_evals_list[t].clone()));
        eta_power *= eta;
    }

    let mut expression: Expression<F> = Expression::<F>::Polynomial(Query::new(0, Rotation::cur()))
        * Expression::<F>::Polynomial(Query::new(1, Rotation::cur()));
    for t in 1..num_relations {
        let h_query = Expression::<F>::Polynomial(Query::new(2 * t, Rotation::cur()));
        let input_query = Expression::<F>::Polynomial(Query::new(2 * t + 1, Rotation::cur()));
        expression = expression + h_query * input_query;
    }

    let challenges: Vec<F> = Vec::new();
    let ys: Vec<Vec<F>> = Vec::new();
    let virtual_poly: VirtualPolynomial<F> = VirtualPolynomial::new(&expression, &polys, &challenges, &ys);

    let (sc_point, terminal_evals) = Sc::<F>::prove(&(), num_vars, virtual_poly, claimed_sum, transcript)?;
    assert_eq!(terminal_evals.len(), 2 * num_relations);

    let mut scaled_h_evals_at_sc_point = Vec::with_capacity(num_relations);
    let mut input_evals_at_sc_point = Vec::with_capacity(num_relations);
    let mut terminal_eval = F::ZERO;
    for t in 0..num_relations {
        let h_t = terminal_evals[2 * t];
        let x_t = terminal_evals[2 * t + 1];
        scaled_h_evals_at_sc_point.push(h_t);
        input_evals_at_sc_point.push(x_t);
        terminal_eval += h_t * x_t;
    }

    for value in &input_evals_at_sc_point {
        transcript.write_field_element(value)?;
    }

    println!("qabase batched WHT relations sumcheck prove {:?}, relations {:?}", now.elapsed(), num_relations);

    Ok(QABaseBatchedWhtRelationProof {
        eta,
        gammas,
        output_evals_at_gammas,
        sc_point,
        input_evals_at_sc_point,
        scaled_h_evals_at_sc_point,
        terminal_eval,
        claimed_sum,
    })
}

pub fn verify_batched_wht_relations_sumcheck<F, H>(
    num_vars: usize,
    num_relations: usize,
    transcript: &mut impl TranscriptRead<CommitmentChunk<H>, F>,
) -> Result<(bool, QABaseBatchedWhtRelationProof<F>), Error>
where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
{
    assert!(num_relations >= 1, "batched WHT verifier needs at least one relation");
    type Sc<F> = ClassicSumCheck<EvaluationsProver<F>>;

    let now = Instant::now();
    let eta = transcript.squeeze_challenges(1)[0];
    let mut gammas = Vec::with_capacity(num_relations);
    for _ in 0..num_relations {
        gammas.push(transcript.squeeze_challenges(num_vars));
    }

    let mut output_evals_at_gammas = Vec::with_capacity(num_relations);
    let mut claimed_sum = F::ZERO;
    let mut eta_power = F::ONE;
    for _ in 0..num_relations {
        let y_t = transcript.read_field_element()?;
        output_evals_at_gammas.push(y_t);
        claimed_sum += eta_power * y_t;
        eta_power *= eta;
    }

    let (terminal_eval, sc_point) = Sc::<F>::verify(&(), num_vars, 2usize, claimed_sum, transcript)?;

    let mut input_evals_at_sc_point = Vec::with_capacity(num_relations);
    for _ in 0..num_relations {
        input_evals_at_sc_point.push(transcript.read_field_element()?);
    }

    let mut scaled_h_evals_at_sc_point = Vec::with_capacity(num_relations);
    let mut expected_terminal = F::ZERO;
    let mut eta_power = F::ONE;
    for t in 0..num_relations {
        let h_t = hadamard_tensor_mle_eval::<F>(&gammas[t], &sc_point) * eta_power;
        scaled_h_evals_at_sc_point.push(h_t);
        expected_terminal += h_t * input_evals_at_sc_point[t];
        eta_power *= eta;
    }
    let ok = terminal_eval == expected_terminal;

    println!("qabase batched WHT relations sumcheck verify {:?}, relations {:?}", now.elapsed(), num_relations);

    Ok((
        ok,
        QABaseBatchedWhtRelationProof {
            eta,
            gammas,
            output_evals_at_gammas,
            sc_point: sc_point.to_vec(),
            input_evals_at_sc_point,
            scaled_h_evals_at_sc_point,
            terminal_eval,
            claimed_sum,
        },
    ))
}

// -----------------------------------------------------------------------------
// Combined QA encoding relation
// -----------------------------------------------------------------------------

//////////////////////////////////////////////////////////////////////////////
// Batched-WHT QA encoding relations
//////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: DeserializeOwned"))]
pub struct QABaseEncodingRelationsBatchedWhtProof<F>
where
    F: PrimeField,
{
    pub batched_wht: QABaseBatchedWhtRelationProof<F>,
    pub scaling: QABaseScalingRelationProof<F>,
}

pub fn prove_qabase_encoding_relations_batched_wht_sumcheck<F, H>(
    qa_params: &QAParams<F>,
    witness: &QAWitness<F>,
    transcript: &mut impl TranscriptWrite<CommitmentChunk<H>, F>,
) -> Result<QABaseEncodingRelationsBatchedWhtProof<F>, Error>
where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
{
    let now = Instant::now();
    let rho = qa_params.inverse_rate;
    assert!(rho >= 2, "inverse_rate must be at least 2");
    assert_eq!(qa_params.e.len(), rho - 1);
    assert_eq!(witness.scaled_wht_blocks.len(), rho - 1);
    assert_eq!(witness.parity_blocks.len(), rho - 1);
    assert_eq!(witness.msg.len(), witness.msg_wht.len());
    for i in 0..(rho - 1) {
        assert_eq!(qa_params.e[i].len(), witness.msg.len());
        assert_eq!(witness.scaled_wht_blocks[i].len(), witness.msg.len());
        assert_eq!(witness.parity_blocks[i].len(), witness.msg.len());
    }

    let mut wht_inputs = Vec::with_capacity(rho);
    let mut wht_outputs = Vec::with_capacity(rho);
    wht_inputs.push(witness.msg.clone());
    wht_outputs.push(witness.msg_wht.clone());
    for i in 0..(rho - 1) {
        wht_inputs.push(witness.scaled_wht_blocks[i].clone());
        wht_outputs.push(witness.parity_blocks[i].clone());
    }

    let batched_wht = prove_batched_wht_relations_sumcheck::<F, H>(&wht_inputs, &wht_outputs, transcript)?;
    let scaling = prove_scaling_relation_sumcheck::<F, H>(
        qa_params,
        &witness.msg_wht,
        &witness.scaled_wht_blocks,
        transcript,
    )?;

    println!("qabase batched-WHT encoding relations prove {:?}", now.elapsed());

    Ok(QABaseEncodingRelationsBatchedWhtProof { batched_wht, scaling })
}

pub fn verify_qabase_encoding_relations_batched_wht_sumcheck<F, H>(
    qa_params: &QAParams<F>,
    transcript: &mut impl TranscriptRead<CommitmentChunk<H>, F>,
) -> Result<(bool, QABaseEncodingRelationsBatchedWhtProof<F>), Error>
where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
{
    let now = Instant::now();
    assert!(!qa_params.e.is_empty(), "QAParams must contain at least one coefficient vector");
    let n = qa_params.e[0].len();
    assert!(n.is_power_of_two(), "QA coefficient vector length must be a power of two");
    let num_vars = n.trailing_zeros() as usize;
    let rho = qa_params.inverse_rate;
    assert_eq!(qa_params.e.len(), rho - 1);

    let (ok_wht, batched_wht) =
        verify_batched_wht_relations_sumcheck::<F, H>(num_vars, rho, transcript)?;
    let (ok_scaling, scaling) = verify_scaling_relation_sumcheck::<F, H>(qa_params, transcript)?;
    let ok = ok_wht && ok_scaling;

    println!("qabase batched-WHT encoding relations verify {:?}", now.elapsed());

    Ok((ok, QABaseEncodingRelationsBatchedWhtProof { batched_wht, scaling }))
}

// -----------------------------------------------------------------------------
// BaseFold opening accumulator
// -----------------------------------------------------------------------------

//////////////////////////////////////////////////////////////////////////////
// BaseFold opening accumulator
//////////////////////////////////////////////////////////////////////////////

#[inline]
fn aux_poly_index_u_block(block_index: usize) -> usize {
    block_index
}

#[inline]
fn aux_poly_index_v_prime(rho: usize) -> usize {
    rho
}

#[inline]
fn aux_poly_index_u_prime(rho: usize, i: usize) -> usize {
    rho + 1 + i
}

#[inline]
fn aux_poly_index_e(rho: usize, i: usize) -> usize {
    2 * rho + i
}

pub fn build_aux_polys_from_folded_witness<F>(
    witness: &QABaseFoldedWitness<F>,
    inverse_rate: usize,
) -> Vec<MultilinearPolynomial<F>>
where
    F: PrimeField + Serialize + DeserializeOwned,
{
    let rho = inverse_rate;

    assert_eq!(witness.folded_qa_witness.parity_blocks.len(), rho - 1);
    assert_eq!(witness.folded_qa_witness.scaled_wht_blocks.len(), rho - 1);

    let mut polys = Vec::with_capacity(2 * rho);

    // u-blocks
    polys.push(vec_to_mle(witness.folded_msg.clone()));
    for parity in witness.folded_qa_witness.parity_blocks.iter() {
        polys.push(vec_to_mle(parity.clone()));
    }

    // v_prime
    polys.push(vec_to_mle(witness.folded_qa_witness.msg_wht.clone()));

    // u_prime blocks
    for block in witness.folded_qa_witness.scaled_wht_blocks.iter() {
        polys.push(vec_to_mle(block.clone()));
    }

    assert_eq!(polys.len(), 2 * rho);
    polys
}

pub fn flatten_aux_commitments<F, H>(
    aux: &QABaseAuxCommitments<F, H>,
    inverse_rate: usize,
) -> Vec<BasefoldCommitment<F, H>>
where
    F: PrimeField,
    H: Hash,
{
    let rho = inverse_rate;

    assert_eq!(aux.u_block_commitments.len(), rho);
    assert_eq!(aux.u_prime_commitments.len(), rho - 1);

    let mut comms = Vec::with_capacity(2 * rho);

    for comm in aux.u_block_commitments.iter() {
        comms.push(comm.clone());
    }

    comms.push(aux.v_prime_commitment.clone());

    for comm in aux.u_prime_commitments.iter() {
        comms.push(comm.clone());
    }

    assert_eq!(comms.len(), 2 * rho);
    comms
}

fn push_basefold_opening_claim<F>(
    points: &mut Vec<Point<F, MultilinearPolynomial<F>>>,
    evals: &mut Vec<Evaluation<F>>,
    poly_index: usize,
    point: Vec<F>,
    value: F,
) where
    F: PrimeField,
{
    let point_index = points.len();
    points.push(point);
    evals.push(Evaluation::new(poly_index, point_index, value));
}

pub struct QABaseProverOpeningAccumulator<'a, F, H>
where
    F: PrimeField,
    H: Hash,
{
    /// Borrowed auxiliary polynomials.
    ///
    /// The first `2 * rho` entries are owned by `QABaseAuxProverData`; the last
    /// `rho - 1` entries are the preprocessed public E_i polynomials in the
    /// prover parameters. Borrowing avoids cloning length-N vectors just to call
    /// `BaseFold::batch_open`.
    pub polys: Vec<&'a MultilinearPolynomial<F>>,

    /// Borrowed BaseFold commitments in the same order as `polys`.
    pub comms: Vec<&'a BasefoldCommitment<F, H>>,
    pub points: Vec<Point<F, MultilinearPolynomial<F>>>,
    pub evals: Vec<Evaluation<F>>,
}

pub struct QABaseVerifierOpeningAccumulator<'a, F, H>
where
    F: PrimeField,
    H: Hash,
{
    /// Borrowed BaseFold commitments.
    pub comms: Vec<&'a BasefoldCommitment<F, H>>,
    pub points: Vec<Point<F, MultilinearPolynomial<F>>>,
    pub evals: Vec<Evaluation<F>>,
}

pub fn flatten_aux_commitment_refs<'a, F, H>(
    aux: &'a QABaseAuxCommitments<F, H>,
    inverse_rate: usize,
) -> Vec<&'a BasefoldCommitment<F, H>>
where
    F: PrimeField,
    H: Hash,
{
    let rho = inverse_rate;

    assert_eq!(aux.u_block_commitments.len(), rho);
    assert_eq!(aux.u_prime_commitments.len(), rho - 1);

    let mut comms = Vec::with_capacity(2 * rho);

    for comm in aux.u_block_commitments.iter() {
        comms.push(comm);
    }

    comms.push(&aux.v_prime_commitment);

    for comm in aux.u_prime_commitments.iter() {
        comms.push(comm);
    }

    assert_eq!(comms.len(), 2 * rho);
    comms
}

pub fn build_qabase_prover_opening_accumulator<'a, F, H>(
    aux_data: &'a QABaseAuxProverData<F, H>,
    pp: &'a QABaseProverParams<F, H>,
) -> QABaseProverOpeningAccumulator<'a, F, H>
where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
{
    let rho = pp.inverse_rate;

    assert_eq!(aux_data.polys.len(), 2 * rho);
    assert_eq!(pp.e_polys.len(), rho - 1);

    let mut polys = Vec::with_capacity(3 * rho - 1);
    polys.extend(aux_data.polys.iter());
    polys.extend(pp.e_polys.iter());

    // Public E_i polynomials and commitments are preprocessed in `trim`.
    // They are only borrowed here, so the online prover does not clone the
    // preprocessed public E polynomials or commitments.
    let mut comms = flatten_aux_commitment_refs::<F, H>(&aux_data.commitments, rho);
    assert_eq!(pp.e_commitments.len(), rho - 1);
    comms.extend(pp.e_commitments.iter());

    assert_eq!(polys.len(), 3 * rho - 1);
    assert_eq!(polys.len(), comms.len());

    QABaseProverOpeningAccumulator {
        polys,
        comms,
        points: Vec::new(),
        evals: Vec::new(),
    }
}

pub fn build_qabase_verifier_opening_accumulator<'a, F, H>(
    vp: &'a QABaseVerifierParams<F, H>,
    aux_commitments: &'a QABaseAuxCommitments<F, H>,
) -> QABaseVerifierOpeningAccumulator<'a, F, H>
where
    F: PrimeField,
    H: Hash,
{
    let rho = vp.inverse_rate;
    let mut comms = flatten_aux_commitment_refs::<F, H>(aux_commitments, rho);

    assert_eq!(vp.e_commitments.len(), rho - 1);
    comms.extend(vp.e_commitments.iter());

    assert_eq!(comms.len(), 3 * rho - 1);

    QABaseVerifierOpeningAccumulator {
        comms,
        points: Vec::new(),
        evals: Vec::new(),
    }
}

// -----------------------------------------------------------------------------
// Item 1: column consistency via selector-sumcheck
// -----------------------------------------------------------------------------

//////////////////////////////////////////////////////////////////////////////
// Item 1: column consistency via one selector-sumcheck
//////////////////////////////////////////////////////////////////////////////

pub fn qabase_selector_evals_from_queries<F>(
    num_cols: usize,
    query_indices: &[usize],
    tau: F,
) -> Vec<F>
where
    F: PrimeField,
{
    assert!(
        num_cols.is_power_of_two(),
        "selector domain size must be a power of two"
    );

    let mut selector = vec![F::ZERO; num_cols];
    let mut tau_power = F::ONE;

    for &idx in query_indices {
        assert!(idx < num_cols, "query index out of range");
        selector[idx] += tau_power;
        tau_power *= tau;
    }

    selector
}

/// Compute the right-hand side of the Item 1 selector-sumcheck:
///
///   sum_j tau^j * <row_challenges, C(:, query_indices[j])>.
pub fn qabase_weighted_folded_sum<F>(
    folded_values: &[F],
    tau: F,
) -> F
where
    F: PrimeField,
{
    let mut acc = F::ZERO;
    let mut tau_power = F::ONE;

    for &value in folded_values {
        acc += tau_power * value;
        tau_power *= tau;
    }

    acc
}

/// Evaluate the sparse selector at an arbitrary point:
///
///   h(z) = sum_j tau^j * eq(<query_indices[j]>, z).
///
/// This is used by the verifier after the selector-sumcheck reduces to one
/// random point.
pub fn qabase_selector_eval_from_queries<F>(
    num_cols: usize,
    query_indices: &[usize],
    tau: F,
    point: &[F],
) -> F
where
    F: PrimeField,
{
    let total_vars = log2_power_of_two(num_cols);
    assert_eq!(
        point.len(),
        total_vars,
        "selector evaluation point has wrong dimension"
    );

    let mut acc = F::ZERO;
    let mut tau_power = F::ONE;

    for &idx in query_indices {
        assert!(idx < num_cols, "query index out of range");
        acc += tau_power * equality_mle_eval_at_index::<F>(idx, total_vars, point);
        tau_power *= tau;
    }

    acc
}


#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: DeserializeOwned"))]
pub struct QABaseItem1ProverOutput<F>
where
    F: PrimeField,
{
    pub query_indices: Vec<usize>,
    pub folded_values: Vec<F>,

    /// Authenticated opened columns C(:, query_indices[j]).
    ///
    /// This is returned so the evaluation check can reuse the same sampled
    /// columns in the DP24-style merged path, instead of performing another
    /// Merkle opening.
    pub opened_columns: Vec<Vec<F>>,

    /// Selector randomizer.
    pub tau: F,

    /// Random point returned by the Item 1 selector-sumcheck.
    ///
    /// The coordinate order follows the concrete codeword layout:
    /// first `num_vars` coordinates are the local column coordinates,
    /// and the last `log2(inverse_rate)` coordinates are the block coordinates.
    pub sc_point: Vec<F>,

    pub selector_eval_at_sc_point: F,
    pub u_eval_at_sc_point: F,

    /// The evaluations of u^{(0)}, ..., u^{(c-1)} at the local part of sc_point.
    pub u_block_evals_at_sc_point: Vec<F>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: DeserializeOwned"))]
pub struct QABaseItem1VerifierOutput<F>
where
    F: PrimeField,
{
    pub query_indices: Vec<usize>,
    pub folded_values: Vec<F>,

    /// Authenticated opened columns C(:, query_indices[j]).
    ///
    /// These columns are verified against the Merkle root before being stored.
    /// They are needed by the full evaluation check to compute
    /// <eq(z_L, .), C(:, query_indices[j])> without repeating Merkle openings.
    pub opened_columns: Vec<Vec<F>>,

    pub tau: F,
    pub sc_point: Vec<F>,
    pub selector_eval_at_sc_point: F,
    pub u_eval_at_sc_point: F,
    pub u_block_evals_at_sc_point: Vec<F>,
}

/// Prove that the Merkle-opened columns are consistent with the folded QA row.
///
/// The verifier samples `num_queries` codeword columns. The prover opens those
/// columns under the Merkle commitment. The verifier locally folds each opened
/// column using the row challenges.
///
/// Instead of adding one BaseFold opening claim per sampled column, all sampled
/// column checks are aggregated into one selector-sumcheck:
///
///   sum_x h(x) * u(x)
///     = sum_j tau^j * <row_challenges, C(:, query_indices[j])>.
///
/// The sumcheck reduces the condition to one evaluation `u(z)`. Since this
/// implementation commits to QA blocks separately, `u(z)` is decomposed into
/// `rho` BaseFold opening claims, one for each block.
pub fn prove_qabase_item1_column_consistency_collect<F, H>(
    acc: &mut QABaseProverOpeningAccumulator<'_, F, H>,
    pp: &QABaseProverParams<F, H>,
    comm: &QABaseCommitment<F, H>,
    folded_witness: &QABaseFoldedWitness<F>,
    transcript: &mut impl TranscriptWrite<CommitmentChunk<H>, F>,
) -> Result<QABaseItem1ProverOutput<F>, Error>
where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
{
    type Sc<F> = ClassicSumCheck<EvaluationsProver<F>>;

    let now = Instant::now();

    let block_len = 1usize << pp.num_vars;
    let rho = pp.inverse_rate;
    let log_rho = log2_power_of_two(rho);
    let total_vars = pp.num_vars + log_rho;
    let num_cols = rho * block_len;

    assert_eq!(
        num_cols,
        1usize << total_vars,
        "full QA codeword domain size mismatch"
    );
    assert_eq!(
        folded_witness.row_challenges.len(),
        pp.num_rows,
        "row challenge length mismatch"
    );
    assert_eq!(
        comm.codeword.len(),
        pp.num_rows,
        "committed row count mismatch"
    );
    for row in &comm.codeword {
        assert_eq!(row.len(), num_cols, "committed codeword row length mismatch");
    }
    assert_eq!(
        folded_witness.folded_qa_witness.codeword.len(),
        num_cols,
        "folded QA codeword length mismatch"
    );

    // 1. Sample Merkle query positions.
    let query_challenges = transcript.squeeze_challenges(pp.num_queries);
    let query_indices = query_challenges
        .iter()
        .map(|challenge| field_challenge_to_index::<F>(challenge, num_cols))
        .collect::<Vec<_>>();

    // 2. Open the sampled columns and compute their folded values.
    let mut folded_values = Vec::with_capacity(query_indices.len());
    let mut opened_columns = Vec::with_capacity(query_indices.len());

    for &full_index in &query_indices {
        write_merkle_path::<H, F>(&comm.codeword_tree, full_index, transcript);

        let opened_column = comm
            .codeword
            .iter()
            .map(|row| row[full_index])
            .collect::<Vec<F>>();

        for value in &opened_column {
            transcript.write_field_element(value)?;
        }

        let folded_value =
            fold_opened_column::<F>(&opened_column, &folded_witness.row_challenges);

        folded_values.push(folded_value);
        opened_columns.push(opened_column);
    }

    // 3. Selector randomizer tau.
    //
    // We use weights tau^j for the j-th sampled column.
    // This also handles duplicate sampled indices correctly.
    let tau = transcript.squeeze_challenges(1)[0];

    let claimed_sum = qabase_weighted_folded_sum::<F>(&folded_values, tau);

    // 4. Build selector h and full folded QA codeword u.
    //
    // This proves:
    //
    //   sum_x h(x) u(x) = sum_j tau^j * <r, C(:, query_indices[j])>.
    //
    // The right-hand side is `claimed_sum`.
    let selector_evals =
        qabase_selector_evals_from_queries::<F>(num_cols, &query_indices, tau);

    let u_full_evals = folded_witness.folded_qa_witness.codeword.clone();

    let selector_poly = MultilinearPolynomial::new(selector_evals);
    let u_full_poly = MultilinearPolynomial::new(u_full_evals);

    let h_query = Expression::<F>::Polynomial(Query::new(0, Rotation::cur()));
    let u_query = Expression::<F>::Polynomial(Query::new(1, Rotation::cur()));
    let expression: Expression<F> = h_query * u_query;

    let polys = vec![selector_poly, u_full_poly];
    let challenges: Vec<F> = Vec::new();
    let ys: Vec<Vec<F>> = Vec::new();

    let virtual_poly =
        VirtualPolynomial::new(&expression, &polys, &challenges, &ys);

    let (sc_point, terminal_evals) =
        Sc::<F>::prove(&(), total_vars, virtual_poly, claimed_sum, transcript)?;
    assert_eq!(terminal_evals.len(), 2);

    let selector_eval_at_sc_point = terminal_evals[0];
    let u_eval_at_sc_point = terminal_evals[1];

    let selector_eval_check = qabase_selector_eval_from_queries::<F>(
        num_cols,
        &query_indices,
        tau,
        &sc_point,
    );
    assert!(
        selector_eval_check == selector_eval_at_sc_point,
        "selector terminal evaluation mismatch"
    );

    // Send the claimed u(sc_point), because the verifier cannot compute it locally.
    transcript.write_field_element(&u_eval_at_sc_point)?;

    // 5. Reduce the full u evaluation to the c block commitments.
    //
    // Codeword layout is:
    //
    //   [u^(0) | u^(1) | ... | u^(c-1)].
    //
    // Since the memory layout uses low bits for the local coordinate and high bits
    // for the block coordinate, the sumcheck point is parsed as:
    //
    //   sc_point = (local_point, block_point).
    let local_point = sc_point[..pp.num_vars].to_vec();
    let block_point = sc_point[pp.num_vars..].to_vec();

    let mut u_block_evals_at_sc_point = Vec::with_capacity(rho);
    let mut recomposed_u_eval = F::ZERO;

    for block_index in 0..rho {
        let block_evals = qa_codeword_block::<F>(
            &folded_witness.folded_qa_witness.codeword,
            block_len,
            block_index,
        );

        let value = eval_mle_from_evals::<F>(block_evals, &local_point);
        transcript.write_field_element(&value)?;
        u_block_evals_at_sc_point.push(value);

        let block_weight =
            equality_mle_eval_at_index::<F>(block_index, log_rho, &block_point);
        recomposed_u_eval += block_weight * value;

        push_basefold_opening_claim::<F>(
            &mut acc.points,
            &mut acc.evals,
            aux_poly_index_u_block(block_index),
            local_point.clone(),
            value,
        );
    }

    assert!(
        recomposed_u_eval == u_eval_at_sc_point,
        "block decomposition of u(sc_point) failed"
    );

    println!(
        "qabase Item 1 selector-sumcheck prove {:?}, queries {}, basefold claims added {}",
        now.elapsed(),
        query_indices.len(),
        rho
    );

    Ok(QABaseItem1ProverOutput {
        query_indices,
        folded_values,
        opened_columns,
        tau,
        sc_point,
        selector_eval_at_sc_point,
        u_eval_at_sc_point,
        u_block_evals_at_sc_point,
    })
}

/// Verify Item 1 column consistency.
///
/// This verifies all sampled Merkle column openings, checks the terminal
/// equation of the selector-sumcheck, decomposes the full QA-codeword
/// evaluation into block evaluations, and appends the corresponding block
/// evaluations to the global BaseFold opening accumulator.
pub fn verify_qabase_item1_column_consistency_collect<F, H>(
    acc: &mut QABaseVerifierOpeningAccumulator<'_, F, H>,
    vp: &QABaseVerifierParams<F, H>,
    comm: &QABaseCommitment<F, H>,
    row_challenges: &[F],
    transcript: &mut impl TranscriptRead<CommitmentChunk<H>, F>,
) -> Result<(bool, QABaseItem1VerifierOutput<F>), Error>
where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
{
    type Sc<F> = ClassicSumCheck<EvaluationsProver<F>>;

    let now = Instant::now();

    let block_len = 1usize << vp.num_vars;
    let rho = vp.inverse_rate;
    let log_rho = log2_power_of_two(rho);
    let total_vars = vp.num_vars + log_rho;
    let num_cols = rho * block_len;

    assert_eq!(
        num_cols,
        1usize << total_vars,
        "full QA codeword domain size mismatch"
    );
    assert_eq!(
        row_challenges.len(),
        vp.num_rows,
        "row challenge length mismatch"
    );

    let root = <QABaseCommitment<F, H> as AsRef<Output<H>>>::as_ref(comm);

    // 1. Recompute Merkle query positions.
    let query_challenges = transcript.squeeze_challenges(vp.num_queries);
    let query_indices = query_challenges
        .iter()
        .map(|challenge| field_challenge_to_index::<F>(challenge, num_cols))
        .collect::<Vec<_>>();

    // 2. Read opened columns, verify Merkle paths, and compute folded values.
    let mut folded_values = Vec::with_capacity(query_indices.len());
    let mut opened_columns = Vec::with_capacity(query_indices.len());

    for &full_index in &query_indices {
        let path = read_merkle_path::<H, F>(num_cols, transcript);

        let opened_column = (0..vp.num_rows)
            .map(|_| transcript.read_field_element())
            .collect::<Result<Vec<F>, Error>>()?;

        let ok_merkle =
            verify_merkle_path_with_leaf::<H, F>(root, &opened_column, &path, full_index);

        if !ok_merkle {
            return Ok((
                false,
                QABaseItem1VerifierOutput {
                    query_indices,
                    folded_values,
                    opened_columns,
                    tau: F::ZERO,
                    sc_point: Vec::new(),
                    selector_eval_at_sc_point: F::ZERO,
                    u_eval_at_sc_point: F::ZERO,
                    u_block_evals_at_sc_point: Vec::new(),
                },
            ));
        }

        let folded_value = fold_opened_column::<F>(&opened_column, row_challenges);
        folded_values.push(folded_value);
        opened_columns.push(opened_column);
    }

    // 3. Selector randomizer tau and claimed right-hand side.
    let tau = transcript.squeeze_challenges(1)[0];

    let claimed_sum = qabase_weighted_folded_sum::<F>(&folded_values, tau);

    // 4. Verify the selector-sumcheck.
    let (terminal_eval, sc_point) =
        Sc::<F>::verify(&(), total_vars, 2usize, claimed_sum, transcript)?;

    let u_eval_at_sc_point = transcript.read_field_element()?;

    let selector_eval_at_sc_point = qabase_selector_eval_from_queries::<F>(
        num_cols,
        &query_indices,
        tau,
        &sc_point,
    );

    let ok_sumcheck =
        terminal_eval == selector_eval_at_sc_point * u_eval_at_sc_point;

    // 5. Verify that u(sc_point) is reconstructed from the c block commitments.
    let local_point = sc_point[..vp.num_vars].to_vec();
    let block_point = sc_point[vp.num_vars..].to_vec();

    let mut u_block_evals_at_sc_point = Vec::with_capacity(rho);
    let mut recomposed_u_eval = F::ZERO;

    for block_index in 0..rho {
        let value = transcript.read_field_element()?;
        u_block_evals_at_sc_point.push(value);

        let block_weight =
            equality_mle_eval_at_index::<F>(block_index, log_rho, &block_point);
        recomposed_u_eval += block_weight * value;

        push_basefold_opening_claim::<F>(
            &mut acc.points,
            &mut acc.evals,
            aux_poly_index_u_block(block_index),
            local_point.clone(),
            value,
        );
    }

    let ok_decomposition = recomposed_u_eval == u_eval_at_sc_point;
    let ok = ok_sumcheck && ok_decomposition;

    println!(
        "qabase Item 1 selector-sumcheck verify {:?}, queries {}, basefold claims added {}, ok_sumcheck {}, ok_decomposition {}",
        now.elapsed(),
        query_indices.len(),
        rho,
        ok_sumcheck,
        ok_decomposition,
    );

    Ok((
        ok,
        QABaseItem1VerifierOutput {
            query_indices,
            folded_values,
            opened_columns,
            tau,
            sc_point: sc_point.to_vec(),
            selector_eval_at_sc_point,
            u_eval_at_sc_point,
            u_block_evals_at_sc_point,
        },
    ))
}

// -----------------------------------------------------------------------------
// Item 2: collect BaseFold claims from QA encoding relations
// -----------------------------------------------------------------------------

//////////////////////////////////////////////////////////////////////////////
// Item 2: collect BaseFold claims from QA encoding relations
//////////////////////////////////////////////////////////////////////////////

/// Add BaseFold opening claims required by the batched WHT and scaling
/// sumchecks.
///
/// The batched WHT proof contributes `2 * rho` witness claims: one input and
/// one output evaluation per WHT relation. The scaling proof contributes
/// `rho` witness claims plus `rho - 1` public E_i opening claims.
///
/// For `rho = 4`, this contributes `8 + 4 + 3 = 15` claims. Together with
/// Item 1's four block openings, the final accumulator has 19 claims.
pub fn collect_qabase_item2_batched_wht_basefold_claims_prover<F, H>(
    acc: &mut QABaseProverOpeningAccumulator<'_, F, H>,
    pp: &QABaseProverParams<F, H>,
    folded_witness: &QABaseFoldedWitness<F>,
    encoding_relations: &QABaseEncodingRelationsBatchedWhtProof<F>,
    transcript: &mut impl TranscriptWrite<CommitmentChunk<H>, F>,
) -> Result<(), Error>
where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
{
    let rho = pp.inverse_rate;
    let batched_wht = &encoding_relations.batched_wht;
    assert_eq!(batched_wht.gammas.len(), rho);
    assert_eq!(batched_wht.output_evals_at_gammas.len(), rho);
    assert_eq!(batched_wht.input_evals_at_sc_point.len(), rho);

    // WHT t=0: input=u^(0), output=v_prime.
    push_basefold_opening_claim::<F>(
        &mut acc.points,
        &mut acc.evals,
        aux_poly_index_v_prime(rho),
        batched_wht.gammas[0].clone(),
        batched_wht.output_evals_at_gammas[0],
    );
    push_basefold_opening_claim::<F>(
        &mut acc.points,
        &mut acc.evals,
        aux_poly_index_u_block(0),
        batched_wht.sc_point.clone(),
        batched_wht.input_evals_at_sc_point[0],
    );

    // WHT t=i+1: input=u_prime_i, output=parity_i=u^(i+1).
    for i in 0..(rho - 1) {
        let t = i + 1;
        push_basefold_opening_claim::<F>(
            &mut acc.points,
            &mut acc.evals,
            aux_poly_index_u_block(i + 1),
            batched_wht.gammas[t].clone(),
            batched_wht.output_evals_at_gammas[t],
        );
        push_basefold_opening_claim::<F>(
            &mut acc.points,
            &mut acc.evals,
            aux_poly_index_u_prime(rho, i),
            batched_wht.sc_point.clone(),
            batched_wht.input_evals_at_sc_point[t],
        );
    }

    // Scaling relation claims.
    let scaling = &encoding_relations.scaling;
    push_basefold_opening_claim::<F>(
        &mut acc.points,
        &mut acc.evals,
        aux_poly_index_v_prime(rho),
        scaling.sc_point.clone(),
        scaling.v_prime_eval_at_sc_point,
    );

    let mut alpha_power = F::ONE;
    let mut u_prime_batch_check = F::ZERO;
    for i in 0..(rho - 1) {
        let value = eval_mle_from_evals(
            &folded_witness.folded_qa_witness.scaled_wht_blocks[i],
            &scaling.sc_point,
        );
        transcript.write_field_element(&value)?;
        u_prime_batch_check += alpha_power * value;
        alpha_power *= scaling.alpha;
        push_basefold_opening_claim::<F>(
            &mut acc.points,
            &mut acc.evals,
            aux_poly_index_u_prime(rho, i),
            scaling.sc_point.clone(),
            value,
        );
    }
    
    assert_eq!(u_prime_batch_check, scaling.u_prime_batch_eval_at_sc_point);

    // Public E_i openings used by the scaling terminal check.
    assert_eq!(scaling.e_evals_at_sc_point.len(), rho - 1);

    let mut alpha_power = F::ONE;
    let mut e_batch_check = F::ZERO;

    for i in 0..(rho - 1) {
        let value = scaling.e_evals_at_sc_point[i];

        e_batch_check += alpha_power * value;
        alpha_power *= scaling.alpha;

        push_basefold_opening_claim::<F>(
            &mut acc.points,
            &mut acc.evals,
            aux_poly_index_e(rho, i),
            scaling.sc_point.clone(),
            value,
        );
    }

    assert_eq!(e_batch_check, scaling.e_batch_eval_at_sc_point);

    Ok(())
}

/// Verifier-side counterpart of
/// `collect_qabase_item2_batched_wht_basefold_claims_prover`.
///
/// It reads the prover-supplied `u_prime_i` evaluations for the scaling
/// relation, checks their random linear combination, and appends all resulting
/// claims to the global BaseFold opening accumulator.
pub fn collect_qabase_item2_batched_wht_basefold_claims_verifier<F, H>(
    acc: &mut QABaseVerifierOpeningAccumulator<'_, F, H>,
    vp: &QABaseVerifierParams<F, H>,
    encoding_relations: &QABaseEncodingRelationsBatchedWhtProof<F>,
    transcript: &mut impl TranscriptRead<CommitmentChunk<H>, F>,
) -> Result<(), Error>
where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
{
    let rho = vp.inverse_rate;
    let batched_wht = &encoding_relations.batched_wht;
    assert_eq!(batched_wht.gammas.len(), rho);
    assert_eq!(batched_wht.output_evals_at_gammas.len(), rho);
    assert_eq!(batched_wht.input_evals_at_sc_point.len(), rho);

    push_basefold_opening_claim::<F>(
        &mut acc.points,
        &mut acc.evals,
        aux_poly_index_v_prime(rho),
        batched_wht.gammas[0].clone(),
        batched_wht.output_evals_at_gammas[0],
    );
    push_basefold_opening_claim::<F>(
        &mut acc.points,
        &mut acc.evals,
        aux_poly_index_u_block(0),
        batched_wht.sc_point.clone(),
        batched_wht.input_evals_at_sc_point[0],
    );

    for i in 0..(rho - 1) {
        let t = i + 1;
        push_basefold_opening_claim::<F>(
            &mut acc.points,
            &mut acc.evals,
            aux_poly_index_u_block(i + 1),
            batched_wht.gammas[t].clone(),
            batched_wht.output_evals_at_gammas[t],
        );
        push_basefold_opening_claim::<F>(
            &mut acc.points,
            &mut acc.evals,
            aux_poly_index_u_prime(rho, i),
            batched_wht.sc_point.clone(),
            batched_wht.input_evals_at_sc_point[t],
        );
    }

    let scaling = &encoding_relations.scaling;
    push_basefold_opening_claim::<F>(
        &mut acc.points,
        &mut acc.evals,
        aux_poly_index_v_prime(rho),
        scaling.sc_point.clone(),
        scaling.v_prime_eval_at_sc_point,
    );

    let mut alpha_power = F::ONE;
    let mut u_prime_batch_check = F::ZERO;
    for i in 0..(rho - 1) {
        let value = transcript.read_field_element()?;
        u_prime_batch_check += alpha_power * value;
        alpha_power *= scaling.alpha;
        push_basefold_opening_claim::<F>(
            &mut acc.points,
            &mut acc.evals,
            aux_poly_index_u_prime(rho, i),
            scaling.sc_point.clone(),
            value,
        );
    }
    
    assert_eq!(u_prime_batch_check, scaling.u_prime_batch_eval_at_sc_point);

    // Public E_i openings used by the scaling terminal check.
    assert_eq!(scaling.e_evals_at_sc_point.len(), rho - 1);

    let mut alpha_power = F::ONE;
    let mut e_batch_check = F::ZERO;

    for i in 0..(rho - 1) {
        let value = scaling.e_evals_at_sc_point[i];

        e_batch_check += alpha_power * value;
        alpha_power *= scaling.alpha;

        push_basefold_opening_claim::<F>(
            &mut acc.points,
            &mut acc.evals,
            aux_poly_index_e(rho, i),
            scaling.sc_point.clone(),
            value,
        );
    }

    assert_eq!(e_batch_check, scaling.e_batch_eval_at_sc_point);

    Ok(())
}


// -----------------------------------------------------------------------------
// Evaluation check
// -----------------------------------------------------------------------------
//
// This section implements the evaluation branch:
//
//   p = eq(z_L, .)^T M,
//   q = QAEnc(p),
//   q^(0)(z_R) = y.
//
// It is intentionally not wired into the full opening scaffold yet.
// The intended DP24-style merged usage is:
//
//   - the proximity test samples and authenticates Merkle columns once;
//   - the evaluation check reuses the same query_indices and tau;
//   - the RHS values are
//       <eq(z_L, .), C(:, query_indices[j])>;
//   - the final BaseFold batch opening includes both proximity and evaluation
//     branch claims.

//////////////////////////////////////////////////////////////////////////////
// Evaluation witness and auxiliary BaseFold commitments
//////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: DeserializeOwned"))]
pub struct QABaseEvalWitness<F>
where
    F: PrimeField,
{
    /// Left part of the evaluation point.
    ///
    /// If the committed multilinear polynomial is reshaped as
    /// M in F^{N_1 x N_2}, then z_left has log2(N_1) coordinates.
    pub z_left: Vec<F>,

    /// row_weights[i] = eq(<i>, z_left).
    pub row_weights: Vec<F>,

    /// p = sum_i row_weights[i] * M[i].
    pub eval_msg: Vec<F>,

    /// q = QAEnc(p).
    pub eval_qa_witness: QAWitness<F>,
}

/// Compute row weights
///
///   row_weights[i] = eq(<i>, z_left).
///
/// These weights are used to fold the committed matrix rows according to the
/// left part of the evaluation point.
pub fn qabase_row_weights_from_z_left<F>(
    num_rows: usize,
    z_left: &[F],
) -> Vec<F>
where
    F: PrimeField,
{
    assert!(num_rows.is_power_of_two(), "num_rows must be a power of two");

    let log_rows = log2_power_of_two(num_rows);

    assert_eq!(
        z_left.len(),
        log_rows,
        "z_left length must equal log2(num_rows)"
    );

    (0..num_rows)
        .map(|i| equality_mle_eval_at_index::<F>(i, log_rows, z_left))
        .collect::<Vec<_>>()
}

/// Build the evaluation witness:
///
///   p = eq(z_L, .)^T M,
///   q = QAEnc(p).
///
/// This is the evaluation analogue of `build_folded_qa_witness_from_rows`.
pub fn build_eval_qa_witness_from_rows<F, H>(
    pp: &QABaseProverParams<F, H>,
    rows: &Vec<Vec<F>>,
    z_left: Vec<F>,
) -> QABaseEvalWitness<F>
where
    F: PrimeField + Serialize + DeserializeOwned + Send + Sync,
    H: Hash,
{
    assert_eq!(rows.len(), pp.num_rows, "row count mismatch");
    assert!(
        !rows.is_empty(),
        "cannot build evaluation witness from an empty matrix"
    );

    let row_len = 1usize << pp.num_vars;

    for row in rows {
        assert_eq!(row.len(), row_len, "row length mismatch");
    }

    let row_weights = qabase_row_weights_from_z_left::<F>(pp.num_rows, &z_left);
    let eval_msg = linear_combine_rows_parallel(rows, &row_weights);
    let eval_qa_witness = qa_encode_with_witness_parallel(&eval_msg, &pp.qa_params);

    QABaseEvalWitness {
        z_left,
        row_weights,
        eval_msg,
        eval_qa_witness,
    }
}

/// Build BaseFold polynomials for q = QAEnc(p).
///
/// Layout:
///
///   q^(0), ..., q^(rho-1), q_v_prime, q_u'_0, ..., q_u'_{rho-2}.
pub fn build_aux_polys_from_eval_witness<F>(
    witness: &QABaseEvalWitness<F>,
    inverse_rate: usize,
) -> Vec<MultilinearPolynomial<F>>
where
    F: PrimeField + Serialize + DeserializeOwned,
{
    let rho = inverse_rate;

    assert_eq!(witness.eval_qa_witness.parity_blocks.len(), rho - 1);
    assert_eq!(
        witness.eval_qa_witness.scaled_wht_blocks.len(),
        rho - 1
    );

    let mut polys = Vec::with_capacity(2 * rho);

    // q-blocks: q^(0), q^(1), ..., q^(rho-1).
    polys.push(vec_to_mle(witness.eval_msg.clone()));

    for parity in witness.eval_qa_witness.parity_blocks.iter() {
        polys.push(vec_to_mle(parity.clone()));
    }

    // q_v_prime.
    polys.push(vec_to_mle(witness.eval_qa_witness.msg_wht.clone()));

    // q_u'_i blocks.
    for block in witness.eval_qa_witness.scaled_wht_blocks.iter() {
        polys.push(vec_to_mle(block.clone()));
    }

    assert_eq!(polys.len(), 2 * rho);

    polys
}

/// Commit all evaluation-branch auxiliary polynomials by BaseFold.
///
/// This writes only the eval-branch BaseFold commitments into the transcript.
/// It does not add any opening claims into the global accumulator.
pub fn commit_eval_witness_with_basefold<F, H>(
    pp: &QABaseProverParams<F, H>,
    rows: &Vec<Vec<F>>,
    z_left: Vec<F>,
    transcript: &mut impl TranscriptWrite<CommitmentChunk<H>, F>,
) -> Result<(QABaseEvalWitness<F>, QABaseAuxProverData<F, H>), Error>
where
    F: PrimeField + Serialize + DeserializeOwned + Send + Sync,
    H: Hash,
{
    type Pcs<F, H> = Basefold<F, H, QABaseFoldConfig>;

    let now = Instant::now();

    let eval_witness =
        build_eval_qa_witness_from_rows::<F, H>(pp, rows, z_left);

    println!("qabase eval witness construction {:?}", now.elapsed());

    let rho = pp.inverse_rate;

    let aux_polys =
        build_aux_polys_from_eval_witness::<F>(&eval_witness, rho);

    assert_eq!(aux_polys.len(), 2 * rho);

    let now = Instant::now();

    let aux_commitments_all = Pcs::<F, H>::batch_commit_and_write(
        &pp.basefold_prover_param,
        aux_polys.iter(),
        transcript,
    )?;

    assert_eq!(aux_commitments_all.len(), 2 * rho);

    println!(
        "qabase eval basefold commit all aux polynomials {:?}, blocks {}",
        now.elapsed(),
        aux_commitments_all.len()
    );

    let q_block_commitments = aux_commitments_all[0..rho].to_vec();
    let q_v_prime_commitment = aux_commitments_all[rho].clone();
    let q_u_prime_commitments = aux_commitments_all[rho + 1..2 * rho].to_vec();

    let commitments = QABaseAuxCommitments {
        u_block_commitments: q_block_commitments,
        v_prime_commitment: q_v_prime_commitment,
        u_prime_commitments: q_u_prime_commitments,
    };

    Ok((
        eval_witness,
        QABaseAuxProverData {
            commitments,
            polys: aux_polys,
        },
    ))
}

/// Verifier reads the evaluation-branch BaseFold commitments.
///
/// Unlike `read_aux_commitments_from_transcript`, this function does not read
/// the main QABase Merkle root and does not sample row challenges.
pub fn read_eval_aux_commitments_from_transcript<F, H>(
    vp: &QABaseVerifierParams<F, H>,
    transcript: &mut impl TranscriptRead<CommitmentChunk<H>, F>,
) -> Result<QABaseAuxCommitments<F, H>, Error>
where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
{
    type Pcs<F, H> = Basefold<F, H, QABaseFoldConfig>;

    let rho = vp.inverse_rate;

    let aux_commitments_all = Pcs::<F, H>::read_commitments(
        &vp.basefold_verifier_param,
        2 * rho,
        transcript,
    )?;

    assert_eq!(aux_commitments_all.len(), 2 * rho);

    let q_block_commitments = aux_commitments_all[0..rho].to_vec();
    let q_v_prime_commitment = aux_commitments_all[rho].clone();
    let q_u_prime_commitments = aux_commitments_all[rho + 1..2 * rho].to_vec();

    Ok(QABaseAuxCommitments {
        u_block_commitments: q_block_commitments,
        v_prime_commitment: q_v_prime_commitment,
        u_prime_commitments: q_u_prime_commitments,
    })
}

//////////////////////////////////////////////////////////////////////////////
// Evaluation branch accumulator layout
//////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Copy, Debug)]
pub struct QABaseEvalAuxLayout {
    pub rho: usize,

    /// Start index of q auxiliary polynomials inside the global accumulator.
    ///
    /// q_aux_start + 0              = q^(0)
    /// q_aux_start + 1              = q^(1)
    /// ...
    /// q_aux_start + rho - 1        = q^(rho-1)
    /// q_aux_start + rho            = q_v_prime
    /// q_aux_start + rho + 1 + i    = q_u'_i
    pub q_aux_start: usize,

    /// Start index of public E_i polynomials inside the global accumulator.
    ///
    /// With the current base accumulator layout, this is 2*rho.
    pub e_start: usize,
}

#[inline]
fn eval_poly_index_q_block(
    layout: QABaseEvalAuxLayout,
    block_index: usize,
) -> usize {
    assert!(block_index < layout.rho, "q block index out of range");
    layout.q_aux_start + block_index
}

#[inline]
fn eval_poly_index_q_v_prime(layout: QABaseEvalAuxLayout) -> usize {
    layout.q_aux_start + layout.rho
}

#[inline]
fn eval_poly_index_q_u_prime(
    layout: QABaseEvalAuxLayout,
    i: usize,
) -> usize {
    assert!(i + 1 < layout.rho, "q_u_prime index out of range");
    layout.q_aux_start + layout.rho + 1 + i
}

#[inline]
fn eval_poly_index_e(
    layout: QABaseEvalAuxLayout,
    i: usize,
) -> usize {
    assert!(i + 1 < layout.rho, "E_i index out of range");
    layout.e_start + i
}

/// Append evaluation-branch auxiliary polynomials and commitments to the prover
/// opening accumulator.
///
/// This assumes the accumulator was first created by
/// `build_qabase_prover_opening_accumulator`, so it already contains:
///
///   proximity aux polynomials: 0 .. 2*rho-1,
///   public E_i polynomials:    2*rho .. 3*rho-2.
pub fn extend_prover_opening_accumulator_with_eval_aux<'a, F, H>(
    acc: &mut QABaseProverOpeningAccumulator<'a, F, H>,
    eval_aux_data: &'a QABaseAuxProverData<F, H>,
) -> QABaseEvalAuxLayout
where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
{
    let rho = eval_aux_data.commitments.u_block_commitments.len();

    assert_eq!(eval_aux_data.polys.len(), 2 * rho);
    assert_eq!(
        eval_aux_data.commitments.u_prime_commitments.len(),
        rho - 1
    );
    assert_eq!(acc.polys.len(), acc.comms.len());

    let e_start = 2 * rho;

    assert!(
        acc.polys.len() >= 3 * rho - 1,
        "base accumulator should already contain proximity aux and public E_i"
    );

    let q_aux_start = acc.polys.len();

    acc.polys.extend(eval_aux_data.polys.iter());

    let q_aux_commitment_refs =
        flatten_aux_commitment_refs::<F, H>(&eval_aux_data.commitments, rho);

    acc.comms.extend(q_aux_commitment_refs);

    assert_eq!(acc.polys.len(), acc.comms.len());

    QABaseEvalAuxLayout {
        rho,
        q_aux_start,
        e_start,
    }
}

/// Append evaluation-branch auxiliary commitments to the verifier opening
/// accumulator.
///
/// This assumes the verifier accumulator was first created by
/// `build_qabase_verifier_opening_accumulator`, so it already contains:
///
///   proximity aux commitments,
///   public E_i commitments.
pub fn extend_verifier_opening_accumulator_with_eval_aux<'a, F, H>(
    acc: &mut QABaseVerifierOpeningAccumulator<'a, F, H>,
    eval_aux_commitments: &'a QABaseAuxCommitments<F, H>,
) -> QABaseEvalAuxLayout
where
    F: PrimeField,
    H: Hash,
{
    let rho = eval_aux_commitments.u_block_commitments.len();

    assert_eq!(
        eval_aux_commitments.u_prime_commitments.len(),
        rho - 1
    );

    let e_start = 2 * rho;

    assert!(
        acc.comms.len() >= 3 * rho - 1,
        "base verifier accumulator should already contain proximity aux and public E_i"
    );

    let q_aux_start = acc.comms.len();

    let q_aux_commitment_refs =
        flatten_aux_commitment_refs::<F, H>(eval_aux_commitments, rho);

    acc.comms.extend(q_aux_commitment_refs);

    QABaseEvalAuxLayout {
        rho,
        q_aux_start,
        e_start,
    }
}

//////////////////////////////////////////////////////////////////////////////
// Evaluation branch: QA encoding relation claims
//////////////////////////////////////////////////////////////////////////////

/// Add BaseFold opening claims required by the evaluation-branch QA encoding
/// sumchecks.
///
/// This is the evaluation analogue of
/// `collect_qabase_item2_batched_wht_basefold_claims_prover`, except all q
/// witness polynomials are shifted by `layout.q_aux_start`, while the public
/// E_i polynomials are shared with the proximity branch.
pub fn collect_qabase_eval_encoding_basefold_claims_prover<F, H>(
    acc: &mut QABaseProverOpeningAccumulator<'_, F, H>,
    layout: QABaseEvalAuxLayout,
    eval_witness: &QABaseEvalWitness<F>,
    encoding_relations: &QABaseEncodingRelationsBatchedWhtProof<F>,
    transcript: &mut impl TranscriptWrite<CommitmentChunk<H>, F>,
) -> Result<(), Error>
where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
{
    let rho = layout.rho;

    let batched_wht = &encoding_relations.batched_wht;

    assert_eq!(batched_wht.gammas.len(), rho);
    assert_eq!(batched_wht.output_evals_at_gammas.len(), rho);
    assert_eq!(batched_wht.input_evals_at_sc_point.len(), rho);

    // WHT relation t = 0:
    //
    //   input  = q^(0),
    //   output = q_v_prime.
    push_basefold_opening_claim::<F>(
        &mut acc.points,
        &mut acc.evals,
        eval_poly_index_q_v_prime(layout),
        batched_wht.gammas[0].clone(),
        batched_wht.output_evals_at_gammas[0],
    );

    push_basefold_opening_claim::<F>(
        &mut acc.points,
        &mut acc.evals,
        eval_poly_index_q_block(layout, 0),
        batched_wht.sc_point.clone(),
        batched_wht.input_evals_at_sc_point[0],
    );

    // WHT relation t = i + 1:
    //
    //   input  = q_u'_i,
    //   output = q^(i+1).
    for i in 0..(rho - 1) {
        let t = i + 1;

        push_basefold_opening_claim::<F>(
            &mut acc.points,
            &mut acc.evals,
            eval_poly_index_q_block(layout, i + 1),
            batched_wht.gammas[t].clone(),
            batched_wht.output_evals_at_gammas[t],
        );

        push_basefold_opening_claim::<F>(
            &mut acc.points,
            &mut acc.evals,
            eval_poly_index_q_u_prime(layout, i),
            batched_wht.sc_point.clone(),
            batched_wht.input_evals_at_sc_point[t],
        );
    }

    // Scaling relation:
    //
    //   q_u'_i = E_i * q_v_prime.
    let scaling = &encoding_relations.scaling;

    push_basefold_opening_claim::<F>(
        &mut acc.points,
        &mut acc.evals,
        eval_poly_index_q_v_prime(layout),
        scaling.sc_point.clone(),
        scaling.v_prime_eval_at_sc_point,
    );

    let mut alpha_power = F::ONE;
    let mut q_u_prime_batch_check = F::ZERO;

    for i in 0..(rho - 1) {
        let value = eval_mle_from_evals::<F>(
            &eval_witness.eval_qa_witness.scaled_wht_blocks[i],
            &scaling.sc_point,
        );

        transcript.write_field_element(&value)?;

        q_u_prime_batch_check += alpha_power * value;
        alpha_power *= scaling.alpha;

        push_basefold_opening_claim::<F>(
            &mut acc.points,
            &mut acc.evals,
            eval_poly_index_q_u_prime(layout, i),
            scaling.sc_point.clone(),
            value,
        );
    }

    assert_eq!(
        q_u_prime_batch_check,
        scaling.u_prime_batch_eval_at_sc_point
    );

    // Public E_i openings. These are shared with the proximity branch.
    assert_eq!(scaling.e_evals_at_sc_point.len(), rho - 1);

    let mut alpha_power = F::ONE;
    let mut e_batch_check = F::ZERO;

    for i in 0..(rho - 1) {
        let value = scaling.e_evals_at_sc_point[i];

        e_batch_check += alpha_power * value;
        alpha_power *= scaling.alpha;

        push_basefold_opening_claim::<F>(
            &mut acc.points,
            &mut acc.evals,
            eval_poly_index_e(layout, i),
            scaling.sc_point.clone(),
            value,
        );
    }

    assert_eq!(e_batch_check, scaling.e_batch_eval_at_sc_point);

    Ok(())
}

/// Verifier-side counterpart of
/// `collect_qabase_eval_encoding_basefold_claims_prover`.
pub fn collect_qabase_eval_encoding_basefold_claims_verifier<F, H>(
    acc: &mut QABaseVerifierOpeningAccumulator<'_, F, H>,
    layout: QABaseEvalAuxLayout,
    encoding_relations: &QABaseEncodingRelationsBatchedWhtProof<F>,
    transcript: &mut impl TranscriptRead<CommitmentChunk<H>, F>,
) -> Result<bool, Error>
where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
{
    let rho = layout.rho;

    let batched_wht = &encoding_relations.batched_wht;

    assert_eq!(batched_wht.gammas.len(), rho);
    assert_eq!(batched_wht.output_evals_at_gammas.len(), rho);
    assert_eq!(batched_wht.input_evals_at_sc_point.len(), rho);

    push_basefold_opening_claim::<F>(
        &mut acc.points,
        &mut acc.evals,
        eval_poly_index_q_v_prime(layout),
        batched_wht.gammas[0].clone(),
        batched_wht.output_evals_at_gammas[0],
    );

    push_basefold_opening_claim::<F>(
        &mut acc.points,
        &mut acc.evals,
        eval_poly_index_q_block(layout, 0),
        batched_wht.sc_point.clone(),
        batched_wht.input_evals_at_sc_point[0],
    );

    for i in 0..(rho - 1) {
        let t = i + 1;

        push_basefold_opening_claim::<F>(
            &mut acc.points,
            &mut acc.evals,
            eval_poly_index_q_block(layout, i + 1),
            batched_wht.gammas[t].clone(),
            batched_wht.output_evals_at_gammas[t],
        );

        push_basefold_opening_claim::<F>(
            &mut acc.points,
            &mut acc.evals,
            eval_poly_index_q_u_prime(layout, i),
            batched_wht.sc_point.clone(),
            batched_wht.input_evals_at_sc_point[t],
        );
    }

    let scaling = &encoding_relations.scaling;

    push_basefold_opening_claim::<F>(
        &mut acc.points,
        &mut acc.evals,
        eval_poly_index_q_v_prime(layout),
        scaling.sc_point.clone(),
        scaling.v_prime_eval_at_sc_point,
    );

    let mut alpha_power = F::ONE;
    let mut q_u_prime_batch_check = F::ZERO;

    for i in 0..(rho - 1) {
        let value = transcript.read_field_element()?;

        q_u_prime_batch_check += alpha_power * value;
        alpha_power *= scaling.alpha;

        push_basefold_opening_claim::<F>(
            &mut acc.points,
            &mut acc.evals,
            eval_poly_index_q_u_prime(layout, i),
            scaling.sc_point.clone(),
            value,
        );
    }

    let ok_q_batch =
        q_u_prime_batch_check == scaling.u_prime_batch_eval_at_sc_point;

    assert_eq!(scaling.e_evals_at_sc_point.len(), rho - 1);

    let mut alpha_power = F::ONE;
    let mut e_batch_check = F::ZERO;

    for i in 0..(rho - 1) {
        let value = scaling.e_evals_at_sc_point[i];

        e_batch_check += alpha_power * value;
        alpha_power *= scaling.alpha;

        push_basefold_opening_claim::<F>(
            &mut acc.points,
            &mut acc.evals,
            eval_poly_index_e(layout, i),
            scaling.sc_point.clone(),
            value,
        );
    }

    let ok_e_batch = e_batch_check == scaling.e_batch_eval_at_sc_point;

    Ok(ok_q_batch && ok_e_batch)
}

//////////////////////////////////////////////////////////////////////////////
// Evaluation branch: sampled-column consistency
//////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: DeserializeOwned"))]
pub struct QABaseEvalColumnConsistencyProverOutput<F>
where
    F: PrimeField,
{
    pub query_indices: Vec<usize>,

    /// eval_folded_values[j] =
    ///
    ///   <eq(z_L, .), C(:, query_indices[j])>.
    pub eval_folded_values: Vec<F>,

    /// Reused selector randomizer.
    ///
    /// In the DP24-merged version, this should be the same tau used by the
    /// proximity sampled-column consistency check.
    pub tau: F,

    pub sc_point: Vec<F>,
    pub selector_eval_at_sc_point: F,
    pub q_eval_at_sc_point: F,
    pub q_block_evals_at_sc_point: Vec<F>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: DeserializeOwned"))]
pub struct QABaseEvalColumnConsistencyVerifierOutput<F>
where
    F: PrimeField,
{
    pub query_indices: Vec<usize>,
    pub eval_folded_values: Vec<F>,
    pub tau: F,
    pub sc_point: Vec<F>,
    pub selector_eval_at_sc_point: F,
    pub q_eval_at_sc_point: F,
    pub q_block_evals_at_sc_point: Vec<F>,
}

/// Prover-side helper.
///
/// Given query indices, compute
///
///   <eq(z_L, .), C(:, query_indices[j])>
///
/// directly from the committed codeword stored in prover state.
///
/// This does not write Merkle paths and is intended for the DP24-merged
/// version, where Merkle openings are already performed by the proximity test.
pub fn qabase_eval_folded_values_from_comm<F, H>(
    comm: &QABaseCommitment<F, H>,
    query_indices: &[usize],
    row_weights: &[F],
) -> Vec<F>
where
    F: PrimeField,
    H: Hash,
{
    assert_eq!(
        comm.codeword.len(),
        row_weights.len(),
        "row_weights length must equal number of committed rows"
    );

    query_indices
        .iter()
        .map(|&full_index| {
            let opened_column = comm
                .codeword
                .iter()
                .map(|row| row[full_index])
                .collect::<Vec<F>>();

            fold_opened_column::<F>(&opened_column, row_weights)
        })
        .collect::<Vec<_>>()
}

/// Verifier-side helper for the DP24-merged version.
///
/// `opened_columns[j]` must be the authenticated column
///
///   C(:, query_indices[j])
///
/// obtained from the proximity sampled-column check.
pub fn qabase_eval_folded_values_from_opened_columns<F>(
    opened_columns: &[Vec<F>],
    row_weights: &[F],
) -> Vec<F>
where
    F: PrimeField,
{
    opened_columns
        .iter()
        .map(|opened_column| {
            fold_opened_column::<F>(opened_column, row_weights)
        })
        .collect::<Vec<_>>()
}

/// Prove evaluation sampled-column consistency:
///
///   sum_x h(x) q(x)
///     = sum_j tau^j * <eq(z_L, .), C(:, query_indices[j])>.
///
/// This function does not authenticate Merkle paths. It assumes the sampled
/// columns have already been authenticated by the proximity test.
pub fn prove_qabase_eval_column_consistency_collect<F, H>(
    acc: &mut QABaseProverOpeningAccumulator<'_, F, H>,
    layout: QABaseEvalAuxLayout,
    pp: &QABaseProverParams<F, H>,
    query_indices: &[usize],
    eval_folded_values: &[F],
    tau: F,
    eval_witness: &QABaseEvalWitness<F>,
    transcript: &mut impl TranscriptWrite<CommitmentChunk<H>, F>,
) -> Result<QABaseEvalColumnConsistencyProverOutput<F>, Error>
where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
{
    type Sc<F> = ClassicSumCheck<EvaluationsProver<F>>;

    let now = Instant::now();

    let block_len = 1usize << pp.num_vars;
    let rho = pp.inverse_rate;
    let log_rho = log2_power_of_two(rho);
    let total_vars = pp.num_vars + log_rho;
    let num_cols = rho * block_len;

    assert_eq!(layout.rho, rho);
    assert_eq!(
        num_cols,
        1usize << total_vars,
        "full QA codeword domain size mismatch"
    );
    assert_eq!(query_indices.len(), eval_folded_values.len());
    assert_eq!(
        eval_witness.eval_qa_witness.codeword.len(),
        num_cols,
        "evaluation QA codeword length mismatch"
    );

    let claimed_sum =
        qabase_weighted_folded_sum::<F>(eval_folded_values, tau);

    let selector_evals =
        qabase_selector_evals_from_queries::<F>(num_cols, query_indices, tau);

    let q_full_evals = eval_witness.eval_qa_witness.codeword.clone();

    let selector_poly = MultilinearPolynomial::new(selector_evals);
    let q_full_poly = MultilinearPolynomial::new(q_full_evals);

    let h_query = Expression::<F>::Polynomial(Query::new(0, Rotation::cur()));
    let q_query = Expression::<F>::Polynomial(Query::new(1, Rotation::cur()));

    let expression: Expression<F> = h_query * q_query;

    let polys = vec![selector_poly, q_full_poly];
    let challenges: Vec<F> = Vec::new();
    let ys: Vec<Vec<F>> = Vec::new();

    let virtual_poly =
        VirtualPolynomial::new(&expression, &polys, &challenges, &ys);

    let (sc_point, terminal_evals) =
        Sc::<F>::prove(&(), total_vars, virtual_poly, claimed_sum, transcript)?;

    assert_eq!(terminal_evals.len(), 2);

    let selector_eval_at_sc_point = terminal_evals[0];
    let q_eval_at_sc_point = terminal_evals[1];

    let selector_eval_check = qabase_selector_eval_from_queries::<F>(
        num_cols,
        query_indices,
        tau,
        &sc_point,
    );

    assert_eq!(
        selector_eval_check,
        selector_eval_at_sc_point,
        "evaluation selector terminal evaluation mismatch"
    );

    // Send q(sc_point). The verifier cannot compute it locally.
    transcript.write_field_element(&q_eval_at_sc_point)?;

    // Decompose q(sc_point) into q-block evaluations.
    let local_point = sc_point[..pp.num_vars].to_vec();
    let block_point = sc_point[pp.num_vars..].to_vec();

    let mut q_block_evals_at_sc_point = Vec::with_capacity(rho);
    let mut recomposed_q_eval = F::ZERO;

    for block_index in 0..rho {
        let block_evals = qa_codeword_block::<F>(
            &eval_witness.eval_qa_witness.codeword,
            block_len,
            block_index,
        );

        let value = eval_mle_from_evals::<F>(block_evals, &local_point);

        transcript.write_field_element(&value)?;

        q_block_evals_at_sc_point.push(value);

        let block_weight =
            equality_mle_eval_at_index::<F>(block_index, log_rho, &block_point);

        recomposed_q_eval += block_weight * value;

        push_basefold_opening_claim::<F>(
            &mut acc.points,
            &mut acc.evals,
            eval_poly_index_q_block(layout, block_index),
            local_point.clone(),
            value,
        );
    }

    assert_eq!(
        recomposed_q_eval,
        q_eval_at_sc_point,
        "block decomposition of q(sc_point) failed"
    );

    println!(
        "qabase evaluation selector-sumcheck prove {:?}, queries {}, basefold claims added {}",
        now.elapsed(),
        query_indices.len(),
        rho
    );

    Ok(QABaseEvalColumnConsistencyProverOutput {
        query_indices: query_indices.to_vec(),
        eval_folded_values: eval_folded_values.to_vec(),
        tau,
        sc_point,
        selector_eval_at_sc_point,
        q_eval_at_sc_point,
        q_block_evals_at_sc_point,
    })
}

/// Verify evaluation sampled-column consistency.
///
/// This assumes `eval_folded_values` were computed from authenticated opened
/// columns:
///
///   eval_folded_values[j]
///     = <eq(z_L, .), C(:, query_indices[j])>.
pub fn verify_qabase_eval_column_consistency_collect<F, H>(
    acc: &mut QABaseVerifierOpeningAccumulator<'_, F, H>,
    layout: QABaseEvalAuxLayout,
    vp: &QABaseVerifierParams<F, H>,
    query_indices: &[usize],
    eval_folded_values: &[F],
    tau: F,
    transcript: &mut impl TranscriptRead<CommitmentChunk<H>, F>,
) -> Result<(bool, QABaseEvalColumnConsistencyVerifierOutput<F>), Error>
where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
{
    type Sc<F> = ClassicSumCheck<EvaluationsProver<F>>;

    let now = Instant::now();

    let block_len = 1usize << vp.num_vars;
    let rho = vp.inverse_rate;
    let log_rho = log2_power_of_two(rho);
    let total_vars = vp.num_vars + log_rho;
    let num_cols = rho * block_len;

    assert_eq!(layout.rho, rho);
    assert_eq!(
        num_cols,
        1usize << total_vars,
        "full QA codeword domain size mismatch"
    );
    assert_eq!(query_indices.len(), eval_folded_values.len());

    let claimed_sum =
        qabase_weighted_folded_sum::<F>(eval_folded_values, tau);

    let (terminal_eval, sc_point) =
        Sc::<F>::verify(&(), total_vars, 2usize, claimed_sum, transcript)?;

    let q_eval_at_sc_point = transcript.read_field_element()?;

    let selector_eval_at_sc_point = qabase_selector_eval_from_queries::<F>(
        num_cols,
        query_indices,
        tau,
        &sc_point,
    );

    let ok_sumcheck =
        terminal_eval == selector_eval_at_sc_point * q_eval_at_sc_point;

    let local_point = sc_point[..vp.num_vars].to_vec();
    let block_point = sc_point[vp.num_vars..].to_vec();

    let mut q_block_evals_at_sc_point = Vec::with_capacity(rho);
    let mut recomposed_q_eval = F::ZERO;

    for block_index in 0..rho {
        let value = transcript.read_field_element()?;

        q_block_evals_at_sc_point.push(value);

        let block_weight =
            equality_mle_eval_at_index::<F>(block_index, log_rho, &block_point);

        recomposed_q_eval += block_weight * value;

        push_basefold_opening_claim::<F>(
            &mut acc.points,
            &mut acc.evals,
            eval_poly_index_q_block(layout, block_index),
            local_point.clone(),
            value,
        );
    }

    let ok_decomposition = recomposed_q_eval == q_eval_at_sc_point;

    let ok = ok_sumcheck && ok_decomposition;

    println!(
        "qabase evaluation selector-sumcheck verify {:?}, queries {}, basefold claims added {}, ok_sumcheck {}, ok_decomposition {}",
        now.elapsed(),
        query_indices.len(),
        rho,
        ok_sumcheck,
        ok_decomposition
    );

    Ok((
        ok,
        QABaseEvalColumnConsistencyVerifierOutput {
            query_indices: query_indices.to_vec(),
            eval_folded_values: eval_folded_values.to_vec(),
            tau,
            sc_point: sc_point.to_vec(),
            selector_eval_at_sc_point,
            q_eval_at_sc_point,
            q_block_evals_at_sc_point,
        },
    ))
}

//////////////////////////////////////////////////////////////////////////////
// Evaluation value claim: q^(0)(z_R) = claimed_value
//////////////////////////////////////////////////////////////////////////////

/// Prover-side final evaluation claim.
///
/// This adds one BaseFold claim:
///
///   q^(0)(z_right) = claimed_value.
///
/// It also locally checks that the witness value equals `claimed_value`.
pub fn collect_qabase_eval_value_claim_prover<F, H>(
    acc: &mut QABaseProverOpeningAccumulator<'_, F, H>,
    layout: QABaseEvalAuxLayout,
    z_right: &[F],
    claimed_value: F,
    eval_witness: &QABaseEvalWitness<F>,
) -> Result<bool, Error>
where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
{
    let msg_len = eval_witness.eval_msg.len();

    assert!(
        msg_len.is_power_of_two(),
        "evaluation message length must be a power of two"
    );

    let num_vars = log2_power_of_two(msg_len);

    assert_eq!(
        z_right.len(),
        num_vars,
        "z_right length must match row polynomial dimension"
    );

    let actual_value =
        eval_mle_from_evals::<F>(&eval_witness.eval_msg, z_right);

    let ok = actual_value == claimed_value;

    push_basefold_opening_claim::<F>(
        &mut acc.points,
        &mut acc.evals,
        eval_poly_index_q_block(layout, 0),
        z_right.to_vec(),
        claimed_value,
    );

    Ok(ok)
}

/// Verifier-side final evaluation claim.
///
/// The verifier cannot compute q^(0)(z_right), so it only appends the public
/// claimed value to the global BaseFold opening accumulator.
pub fn collect_qabase_eval_value_claim_verifier<F, H>(
    acc: &mut QABaseVerifierOpeningAccumulator<'_, F, H>,
    layout: QABaseEvalAuxLayout,
    z_right: &[F],
    claimed_value: F,
) -> Result<(), Error>
where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
{
    push_basefold_opening_claim::<F>(
        &mut acc.points,
        &mut acc.evals,
        eval_poly_index_q_block(layout, 0),
        z_right.to_vec(),
        claimed_value,
    );

    Ok(())
}

//////////////////////////////////////////////////////////////////////////////
// Optional helpers for running eval encoding sumcheck
//////////////////////////////////////////////////////////////////////////////

/// Prove the QA encoding relations for the evaluation branch q = QAEnc(p).
///
/// This is just a thin wrapper around the existing batched-WHT encoding
/// relation prover.
pub fn prove_qabase_eval_encoding_relations<F, H>(
    pp: &QABaseProverParams<F, H>,
    eval_witness: &QABaseEvalWitness<F>,
    transcript: &mut impl TranscriptWrite<CommitmentChunk<H>, F>,
) -> Result<QABaseEncodingRelationsBatchedWhtProof<F>, Error>
where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
{
    prove_qabase_encoding_relations_batched_wht_sumcheck::<F, H>(
        &pp.qa_params,
        &eval_witness.eval_qa_witness,
        transcript,
    )
}

/// Verify the QA encoding relations for the evaluation branch q = QAEnc(p).
///
/// This is just a thin wrapper around the existing batched-WHT encoding
/// relation verifier.
pub fn verify_qabase_eval_encoding_relations<F, H>(
    vp: &QABaseVerifierParams<F, H>,
    transcript: &mut impl TranscriptRead<CommitmentChunk<H>, F>,
) -> Result<(bool, QABaseEncodingRelationsBatchedWhtProof<F>), Error>
where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
{
    verify_qabase_encoding_relations_batched_wht_sumcheck::<F, H>(
        &vp.qa_params,
        transcript,
    )
}

// -----------------------------------------------------------------------------
// Opening modes
// -----------------------------------------------------------------------------

/// Opening mode used by benchmarks.
///
/// - Merge keeps the existing benchmark path. It runs only the current
///   proximity scaffold and is used as the DP24-style merged timing path.
/// - Full runs the complete protocol, including both the proximity branch and
///   the evaluation branch q = QAEnc(eq(z_L, .)^T M).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QABaseOpeningMode {
    Merge,
    Full,
}

/// Split the full multilinear evaluation point into row and column parts.
///
/// If M has 2^log_rows rows and each row has 2^num_vars entries, then the full
/// polynomial has log_rows + num_vars variables.
///
/// Convention:
///   point = (z_left, z_right),
/// where z_left indexes rows and z_right indexes columns.
pub fn qabase_split_evaluation_point<F>(
    point: &[F],
    num_rows: usize,
    num_vars: usize,
) -> (Vec<F>, Vec<F>)
where
    F: PrimeField,
{
    assert!(num_rows.is_power_of_two(), "num_rows must be a power of two");

    let log_rows = log2_power_of_two(num_rows);

    assert_eq!(
        point.len(),
        log_rows + num_vars,
        "evaluation point length mismatch"
    );

    let z_left = point[..log_rows].to_vec();
    let z_right = point[log_rows..].to_vec();

    (z_left, z_right)
}

// -----------------------------------------------------------------------------
// Full opening scaffold
// -----------------------------------------------------------------------------

//////////////////////////////////////////////////////////////////////////////
// Full opening scaffold
//////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: DeserializeOwned"))]
pub struct QABaseOpenBatchedWhtProverOutput<F, H>
where
    F: PrimeField,
    H: Hash,
{
    pub folded_witness: QABaseFoldedWitness<F>,
    pub aux_commitments: QABaseAuxCommitments<F, H>,
    pub encoding_relations: QABaseEncodingRelationsBatchedWhtProof<F>,
    pub item1: QABaseItem1ProverOutput<F>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: DeserializeOwned"))]
pub struct QABaseOpenBatchedWhtVerifierOutput<F, H>
where
    F: PrimeField,
    H: Hash,
{
    pub row_challenges: Vec<F>,
    pub aux_commitments: QABaseAuxCommitments<F, H>,
    pub encoding_relations: QABaseEncodingRelationsBatchedWhtProof<F>,
    pub item1: QABaseItem1VerifierOutput<F>,
}


#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: DeserializeOwned"))]
pub struct QABaseFullOpenBatchedWhtProverOutput<F, H>
where
    F: PrimeField,
    H: Hash,
{
    // Proximity branch.
    pub folded_witness: QABaseFoldedWitness<F>,
    pub aux_commitments: QABaseAuxCommitments<F, H>,
    pub encoding_relations: QABaseEncodingRelationsBatchedWhtProof<F>,
    pub item1: QABaseItem1ProverOutput<F>,

    // Evaluation branch.
    pub eval_witness: QABaseEvalWitness<F>,
    pub eval_aux_commitments: QABaseAuxCommitments<F, H>,
    pub eval_encoding_relations: QABaseEncodingRelationsBatchedWhtProof<F>,
    pub eval_item1: QABaseEvalColumnConsistencyProverOutput<F>,

    // Local prover-side check of q^(0)(z_R) = claimed_value.
    pub ok_eval_value: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: DeserializeOwned"))]
pub struct QABaseFullOpenBatchedWhtVerifierOutput<F, H>
where
    F: PrimeField,
    H: Hash,
{
    // Proximity branch.
    pub row_challenges: Vec<F>,
    pub aux_commitments: QABaseAuxCommitments<F, H>,
    pub encoding_relations: QABaseEncodingRelationsBatchedWhtProof<F>,
    pub item1: QABaseItem1VerifierOutput<F>,

    // Evaluation branch.
    pub eval_aux_commitments: QABaseAuxCommitments<F, H>,
    pub eval_encoding_relations: QABaseEncodingRelationsBatchedWhtProof<F>,
    pub eval_item1: QABaseEvalColumnConsistencyVerifierOutput<F>,
}

/// Full QABase opening protocol using:
///
/// - one folded QA witness;
/// - batched WHT + scaling sumchecks for QA encoding correctness;
/// - one selector-sumcheck for sampled-column consistency;
/// - one global batched BaseFold opening for all remaining MLE claims.
///
///
/// The public E commitments are generated during indexing / trim. The online
/// proof only includes E_i(r) values and their batched BaseFold opening claims.
pub fn prove_qabase_open_scaffold_global_batch_batched_wht<F, H>(
    pp: &QABaseProverParams<F, H>,
    word: &Vec<Vec<F>>,
    comm: &QABaseCommitment<F, H>,
    transcript: &mut impl TranscriptWrite<CommitmentChunk<H>, F>,
) -> Result<QABaseOpenBatchedWhtProverOutput<F, H>, Error>
where
    F: PrimeField + Serialize + DeserializeOwned + Send + Sync,
    H: Hash,
{
    type Pcs<F, H> = Basefold<F, H, QABaseFoldConfig>;

    let now = Instant::now();
    let (folded_witness, aux_data) =
        commit_folded_witness_with_basefold::<F, H>(pp, word, transcript)?;

    let mut opening_acc = build_qabase_prover_opening_accumulator::<F, H>(
        &aux_data,
        pp,
    );

    let encoding_relations = prove_qabase_encoding_relations_batched_wht_sumcheck::<F, H>(
        &pp.qa_params,
        &folded_witness.folded_qa_witness,
        transcript,
    )?;

    collect_qabase_item2_batched_wht_basefold_claims_prover::<F, H>(
        &mut opening_acc,
        pp,
        &folded_witness,
        &encoding_relations,
        transcript,
    )?;

    let item1 = prove_qabase_item1_column_consistency_collect::<F, H>(
        &mut opening_acc,
        pp,
        comm,
        &folded_witness,
        transcript,
    )?;

    let now_basefold_open = Instant::now();
    Pcs::<F, H>::batch_open(
        &pp.basefold_prover_param,
        opening_acc.polys.iter().copied(),
        opening_acc.comms.iter().copied(),
        &opening_acc.points,
        &opening_acc.evals,
        transcript,
    )?;
    println!(
        "qabase basefold batch_open only {:?}, claims {:?}",
        now_basefold_open.elapsed(),
        opening_acc.evals.len()
    );

    println!(
        "qabase global batch + batched WHT prove {:?}, claims {:?}",
        now.elapsed(),
        opening_acc.evals.len()
    );

    drop(opening_acc);

    Ok(QABaseOpenBatchedWhtProverOutput {
        folded_witness,
        aux_commitments: aux_data.commitments,
        encoding_relations,
        item1,
    })
}

/// Verify the full QABase opening protocol.
///
/// The verifier reconstructs the same global BaseFold opening accumulator as
/// the prover: read auxiliary commitments, verify encoding-relation
/// sumchecks, collect Item 2 claims, verify Item 1 selector-sumcheck, and then
/// verify all accumulated BaseFold claims in one batch.
pub fn verify_qabase_open_scaffold_global_batch_batched_wht<F, H>(
    vp: &QABaseVerifierParams<F, H>,
    comm: &QABaseCommitment<F, H>,
    transcript: &mut impl TranscriptRead<CommitmentChunk<H>, F>,
) -> Result<(bool, QABaseOpenBatchedWhtVerifierOutput<F, H>), Error>
where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
{
    type Pcs<F, H> = Basefold<F, H, QABaseFoldConfig>;

    let now = Instant::now();
    let (row_challenges, aux_commitments) =
        read_aux_commitments_from_transcript::<F, H>(vp, comm, transcript)?;

    let mut opening_acc =
        build_qabase_verifier_opening_accumulator::<F, H>(vp, &aux_commitments);

    let (ok_encoding, encoding_relations) =
        verify_qabase_encoding_relations_batched_wht_sumcheck::<F, H>(&vp.qa_params, transcript)?;

    collect_qabase_item2_batched_wht_basefold_claims_verifier::<F, H>(
        &mut opening_acc,
        vp,
        &encoding_relations,
        transcript,
    )?;

    let (ok_item1, item1) = verify_qabase_item1_column_consistency_collect::<F, H>(
        &mut opening_acc,
        vp,
        comm,
        &row_challenges,
        transcript,
    )?;

    let now_basefold_verify = Instant::now();
    Pcs::<F, H>::batch_verify(
        &vp.basefold_verifier_param,
        opening_acc.comms.iter().copied(),
        &opening_acc.points,
        &opening_acc.evals,
        transcript,
    )?;
    println!(
        "qabase basefold batch_verify only {:?}, claims {:?}",
        now_basefold_verify.elapsed(),
        opening_acc.evals.len()
    );

    let ok = ok_encoding && ok_item1;

    println!(
        "qabase global batch + batched WHT verify {:?}, claims {:?}",
        now.elapsed(),
        opening_acc.evals.len()
    );

    Ok((
        ok,
        QABaseOpenBatchedWhtVerifierOutput {
            row_challenges,
            aux_commitments,
            encoding_relations,
            item1,
        },
    ))
}




// -----------------------------------------------------------------------------
// Full QABase/Quasar opening path
// -----------------------------------------------------------------------------

/// Full QABase/Quasar opening protocol.
///
/// This runs both:
///
///   1. proximity branch:
///        u = QAEnc(r^T M);
///
///   2. evaluation branch:
///        p = eq(z_L, .)^T M,
///        q = QAEnc(p),
///        q^(0)(z_R) = claimed_value.
///
/// The two branches share the sampled Merkle columns through Item 1 output,
/// and all BaseFold claims are opened in one final batch.
pub fn prove_qabase_open_full_global_batch_batched_wht<F, H>(
    pp: &QABaseProverParams<F, H>,
    word: &Vec<Vec<F>>,
    comm: &QABaseCommitment<F, H>,
    z_left: Vec<F>,
    z_right: Vec<F>,
    claimed_value: F,
    transcript: &mut impl TranscriptWrite<CommitmentChunk<H>, F>,
) -> Result<QABaseFullOpenBatchedWhtProverOutput<F, H>, Error>
where
    F: PrimeField + Serialize + DeserializeOwned + Send + Sync,
    H: Hash,
{
    type Pcs<F, H> = Basefold<F, H, QABaseFoldConfig>;

    let now = Instant::now();

    // 1. Commit proximity auxiliary polynomials:
    //      u^(0), ..., u^(rho-1), v', u'_0, ..., u'_{rho-2}.
    let (folded_witness, aux_data) =
        commit_folded_witness_with_basefold::<F, H>(pp, word, transcript)?;

    // 2. Commit evaluation auxiliary polynomials:
    //      q^(0), ..., q^(rho-1), q_v', q_u'_0, ..., q_u'_{rho-2}.
    // These commitments are written before any evaluation-branch sumcheck
    // challenges are sampled.
    let (eval_witness, eval_aux_data) =
        commit_eval_witness_with_basefold::<F, H>(
            pp,
            word,
            z_left,
            transcript,
        )?;

    // 3. Build one global BaseFold opening accumulator.
    let mut opening_acc =
        build_qabase_prover_opening_accumulator::<F, H>(&aux_data, pp);

    let eval_layout =
        extend_prover_opening_accumulator_with_eval_aux::<F, H>(
            &mut opening_acc,
            &eval_aux_data,
        );

    // 4. Proximity QA encoding relation.
    let encoding_relations =
        prove_qabase_encoding_relations_batched_wht_sumcheck::<F, H>(
            &pp.qa_params,
            &folded_witness.folded_qa_witness,
            transcript,
        )?;

    collect_qabase_item2_batched_wht_basefold_claims_prover::<F, H>(
        &mut opening_acc,
        pp,
        &folded_witness,
        &encoding_relations,
        transcript,
    )?;

    // 5. Evaluation QA encoding relation.
    let eval_encoding_relations =
        prove_qabase_eval_encoding_relations::<F, H>(
            pp,
            &eval_witness,
            transcript,
        )?;

    collect_qabase_eval_encoding_basefold_claims_prover::<F, H>(
        &mut opening_acc,
        eval_layout,
        &eval_witness,
        &eval_encoding_relations,
        transcript,
    )?;

    // 6. Proximity sampled-column consistency.
    // This samples query_indices and tau. The evaluation sampled-column
    // consistency check reuses the same query_indices and tau.
    let item1 = prove_qabase_item1_column_consistency_collect::<F, H>(
        &mut opening_acc,
        pp,
        comm,
        &folded_witness,
        transcript,
    )?;

    // 7. Evaluation sampled-column consistency.
    let eval_folded_values =
        qabase_eval_folded_values_from_comm::<F, H>(
            comm,
            &item1.query_indices,
            &eval_witness.row_weights,
        );

    let eval_item1 =
        prove_qabase_eval_column_consistency_collect::<F, H>(
            &mut opening_acc,
            eval_layout,
            pp,
            &item1.query_indices,
            &eval_folded_values,
            item1.tau,
            &eval_witness,
            transcript,
        )?;

    // 8. Final evaluation value claim:
    //      q^(0)(z_R) = claimed_value.
    let ok_eval_value =
        collect_qabase_eval_value_claim_prover::<F, H>(
            &mut opening_acc,
            eval_layout,
            &z_right,
            claimed_value,
            &eval_witness,
        )?;

    assert!(
        ok_eval_value,
        "claimed evaluation value does not match prover witness"
    );

    // 9. One final global BaseFold batch opening for all claims.
    let now_basefold_open = Instant::now();

    Pcs::<F, H>::batch_open(
        &pp.basefold_prover_param,
        opening_acc.polys.iter().copied(),
        opening_acc.comms.iter().copied(),
        &opening_acc.points,
        &opening_acc.evals,
        transcript,
    )?;

    println!(
        "qabase full basefold batch_open only {:?}, claims {:?}",
        now_basefold_open.elapsed(),
        opening_acc.evals.len()
    );

    println!(
        "qabase full global batch + batched WHT prove {:?}, claims {:?}",
        now.elapsed(),
        opening_acc.evals.len()
    );

    drop(opening_acc);

    Ok(QABaseFullOpenBatchedWhtProverOutput {
        folded_witness,
        aux_commitments: aux_data.commitments,
        encoding_relations,
        item1,

        eval_witness,
        eval_aux_commitments: eval_aux_data.commitments,
        eval_encoding_relations,
        eval_item1,

        ok_eval_value,
    })
}

/// Verify the full QABase/Quasar opening protocol.
///
/// This verifies both proximity consistency and the evaluation branch, and then
/// verifies all accumulated BaseFold claims in one final batch.
pub fn verify_qabase_open_full_global_batch_batched_wht<F, H>(
    vp: &QABaseVerifierParams<F, H>,
    comm: &QABaseCommitment<F, H>,
    z_left: Vec<F>,
    z_right: Vec<F>,
    claimed_value: F,
    transcript: &mut impl TranscriptRead<CommitmentChunk<H>, F>,
) -> Result<(bool, QABaseFullOpenBatchedWhtVerifierOutput<F, H>), Error>
where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
{
    type Pcs<F, H> = Basefold<F, H, QABaseFoldConfig>;

    let now = Instant::now();

    // 1. Read proximity auxiliary commitments.
    let (row_challenges, aux_commitments) =
        read_aux_commitments_from_transcript::<F, H>(vp, comm, transcript)?;

    // 2. Read evaluation auxiliary commitments.
    let eval_aux_commitments =
        read_eval_aux_commitments_from_transcript::<F, H>(
            vp,
            transcript,
        )?;

    // 3. Build one global BaseFold opening accumulator.
    let mut opening_acc =
        build_qabase_verifier_opening_accumulator::<F, H>(
            vp,
            &aux_commitments,
        );

    let eval_layout =
        extend_verifier_opening_accumulator_with_eval_aux::<F, H>(
            &mut opening_acc,
            &eval_aux_commitments,
        );

    // 4. Verify proximity QA encoding relation.
    let (ok_encoding, encoding_relations) =
        verify_qabase_encoding_relations_batched_wht_sumcheck::<F, H>(
            &vp.qa_params,
            transcript,
        )?;

    collect_qabase_item2_batched_wht_basefold_claims_verifier::<F, H>(
        &mut opening_acc,
        vp,
        &encoding_relations,
        transcript,
    )?;

    // 5. Verify evaluation QA encoding relation.
    let (ok_eval_encoding, eval_encoding_relations) =
        verify_qabase_eval_encoding_relations::<F, H>(
            vp,
            transcript,
        )?;

    let ok_eval_encoding_claims =
        collect_qabase_eval_encoding_basefold_claims_verifier::<F, H>(
            &mut opening_acc,
            eval_layout,
            &eval_encoding_relations,
            transcript,
        )?;

    // 6. Verify proximity sampled-column consistency.
    // This authenticates Merkle opened columns. The authenticated columns are
    // reused below by the evaluation sampled-column consistency check.
    let (ok_item1, item1) =
        verify_qabase_item1_column_consistency_collect::<F, H>(
            &mut opening_acc,
            vp,
            comm,
            &row_challenges,
            transcript,
        )?;

    // 7. Verify evaluation sampled-column consistency using the same opened
    // columns, query_indices, and tau.
    let row_weights =
        qabase_row_weights_from_z_left::<F>(vp.num_rows, &z_left);

    let eval_folded_values =
        qabase_eval_folded_values_from_opened_columns::<F>(
            &item1.opened_columns,
            &row_weights,
        );

    let (ok_eval_item1, eval_item1) =
        verify_qabase_eval_column_consistency_collect::<F, H>(
            &mut opening_acc,
            eval_layout,
            vp,
            &item1.query_indices,
            &eval_folded_values,
            item1.tau,
            transcript,
        )?;

    // 8. Add public evaluation value claim:
    //      q^(0)(z_R) = claimed_value.
    collect_qabase_eval_value_claim_verifier::<F, H>(
        &mut opening_acc,
        eval_layout,
        &z_right,
        claimed_value,
    )?;

    // 9. Verify one final global BaseFold batch opening.
    let now_basefold_verify = Instant::now();

    Pcs::<F, H>::batch_verify(
        &vp.basefold_verifier_param,
        opening_acc.comms.iter().copied(),
        &opening_acc.points,
        &opening_acc.evals,
        transcript,
    )?;

    println!(
        "qabase full basefold batch_verify only {:?}, claims {:?}",
        now_basefold_verify.elapsed(),
        opening_acc.evals.len()
    );

    let ok =
        ok_encoding
            && ok_eval_encoding
            && ok_eval_encoding_claims
            && ok_item1
            && ok_eval_item1;

    println!(
        "qabase full global batch + batched WHT verify {:?}, claims {:?}, ok {}",
        now.elapsed(),
        opening_acc.evals.len(),
        ok,
    );

    Ok((
        ok,
        QABaseFullOpenBatchedWhtVerifierOutput {
            row_challenges,
            aux_commitments,
            encoding_relations,
            item1,

            eval_aux_commitments,
            eval_encoding_relations,
            eval_item1,
        },
    ))
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

//////////////////////////////////////////////////////////////////////////////
// Tests
//////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {
    use super::*;

    use crate::util::{
        hash::Blake2s,
        transcript::{Blake2sTranscript, InMemoryTranscript, TranscriptRead, TranscriptWrite},
    };
    use halo2_curves::bn256::Fr;
    use rand_chacha::rand_core::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::io::Cursor;

    type TestTranscript = Blake2sTranscript<Cursor<Vec<u8>>>;

    #[test]
    fn test_qabase_setup_trim_commit() {
        let num_vars = 8;
        let poly_size = 1usize << num_vars;
        let num_rows = 4usize;
        let inverse_rate = 2usize;
        let num_queries = 16usize;
        let mut rng = ChaCha8Rng::from_seed([7u8; 32]);

        let param = setup::<Fr, Blake2s>(
            poly_size,
            1,
            &mut rng,
            Some(num_rows),
            Some(inverse_rate),
            Some(num_queries),
        );
        let (pp, vp) = trim::<Fr, Blake2s>(&param, poly_size, 1);
        assert_eq!(pp.num_vars, num_vars);
        assert_eq!(vp.num_rows, num_rows);

        let word = (0..num_rows)
            .map(|_| (0..poly_size).map(|_| Fr::random(&mut rng)).collect::<Vec<_>>())
            .collect::<Vec<Vec<Fr>>>();
        let mut transcript = TestTranscript::new(());
        let comm = commit_and_write::<Fr, Blake2s>(&pp, &word, &mut transcript);

        assert_eq!(comm.codeword.len(), num_rows);
        assert_eq!(comm.codeword_tree[0].len(), inverse_rate * poly_size);
        for row in 0..num_rows {
            let msg_block = qa_codeword_block(&comm.codeword[row], poly_size, qa_message_block_index());
            assert_eq!(msg_block, word[row].as_slice());
        }
    }

    #[test]
    fn test_qabase_inverse_rate_4_commit() {
        let num_vars = 6;
        let poly_size = 1usize << num_vars;
        let num_rows = 4usize;
        let inverse_rate = 4usize;
        let num_queries = 16usize;
        let mut rng = ChaCha8Rng::from_seed([9u8; 32]);

        let param = setup::<Fr, Blake2s>(
            poly_size,
            1,
            &mut rng,
            Some(num_rows),
            Some(inverse_rate),
            Some(num_queries),
        );
        let (pp, _) = trim::<Fr, Blake2s>(&param, poly_size, 1);
        let word = (0..num_rows)
            .map(|_| (0..poly_size).map(|_| Fr::random(&mut rng)).collect::<Vec<_>>())
            .collect::<Vec<Vec<Fr>>>();
        let mut transcript = TestTranscript::new(());
        let comm = commit_and_write::<Fr, Blake2s>(&pp, &word, &mut transcript);

        assert_eq!(comm.codeword.len(), num_rows);
        assert!(comm.qa_witnesses.is_empty(), "commitment should be codeword-only");
        for row in 0..num_rows {
            assert_eq!(comm.codeword[row].len(), inverse_rate * poly_size);
            let expected_witness = qa_encode_with_witness(&word[row], &pp.qa_params);
            for i in 0..(inverse_rate - 1) {
                let parity_block = qa_codeword_block(&comm.codeword[row], poly_size, qa_parity_block_index(i));
                assert_eq!(parity_block, expected_witness.parity_blocks[i].as_slice());
            }
        }
    }

    #[test]
    fn test_qabase_basefold_aux_commitments() {
        let num_vars = 6;
        let poly_size = 1usize << num_vars;
        let num_rows = 4usize;
        let inverse_rate = 2usize;
        let num_queries = 8usize;
        let mut rng = ChaCha8Rng::from_seed([13u8; 32]);

        let param = setup::<Fr, Blake2s>(
            poly_size,
            1,
            &mut rng,
            Some(num_rows),
            Some(inverse_rate),
            Some(num_queries),
        );
        let (pp, vp) = trim::<Fr, Blake2s>(&param, poly_size, 1);
        let word = (0..num_rows)
            .map(|_| (0..poly_size).map(|_| Fr::random(&mut rng)).collect::<Vec<_>>())
            .collect::<Vec<Vec<Fr>>>();
        let mut prover_transcript = TestTranscript::new(());
        let comm = commit_and_write::<Fr, Blake2s>(&pp, &word, &mut prover_transcript);
        let (folded_witness, aux_data) =
            commit_folded_witness_with_basefold::<Fr, Blake2s>(&pp, &word, &mut prover_transcript)
                .unwrap();
        assert_eq!(folded_witness.row_challenges.len(), num_rows);
        assert_eq!(aux_data.commitments.u_block_commitments.len(), inverse_rate);
        assert_eq!(aux_data.polys.len(), 2 * inverse_rate);

        let proof = prover_transcript.into_proof();
        let mut verifier_transcript = TestTranscript::from_proof((), proof.as_slice());
        let (verifier_row_challenges, verifier_aux_commitments) =
            read_aux_commitments_from_transcript::<Fr, Blake2s>(&vp, &comm, &mut verifier_transcript)
                .unwrap();
        assert_eq!(verifier_row_challenges, folded_witness.row_challenges);
        assert_eq!(verifier_aux_commitments.u_prime_commitments.len(), inverse_rate - 1);
    }

    #[test]
    fn test_qabase_scaling_relation_sumcheck() {
        let num_vars = 6;
        let n = 1usize << num_vars;
        let inverse_rate = 4usize;
        let mut rng = ChaCha8Rng::from_seed([23u8; 32]);
        let qa_params = QAParams::<Fr>::new_random(n, inverse_rate, &mut rng);
        let msg = (0..n).map(|_| Fr::random(&mut rng)).collect::<Vec<_>>();
        let witness = qa_encode_with_witness(&msg, &qa_params);
        let mut prover_transcript = TestTranscript::new(());
        let prover_proof = prove_scaling_relation_sumcheck::<Fr, Blake2s>(
            &qa_params,
            &witness.msg_wht,
            &witness.scaled_wht_blocks,
            &mut prover_transcript,
        )
        .unwrap();
        let proof = prover_transcript.into_proof();
        let mut verifier_transcript = TestTranscript::from_proof((), proof.as_slice());
        let (ok, verifier_proof) =
            verify_scaling_relation_sumcheck::<Fr, Blake2s>(&qa_params, &mut verifier_transcript).unwrap();
        assert!(ok);
        assert_eq!(prover_proof.alpha, verifier_proof.alpha);
    }

    #[test]
    fn test_qabase_batched_wht_relations_sumcheck() {
        let num_vars = 6;
        let n = 1usize << num_vars;
        let num_relations = 5usize;
        let mut rng = ChaCha8Rng::from_seed([61u8; 32]);
        let mut inputs = Vec::with_capacity(num_relations);
        let mut outputs = Vec::with_capacity(num_relations);
        for _ in 0..num_relations {
            let input = (0..n).map(|_| Fr::random(&mut rng)).collect::<Vec<_>>();
            let mut output = input.clone();
            wht(&mut output);
            inputs.push(input);
            outputs.push(output);
        }
        let mut prover_transcript = TestTranscript::new(());
        let prover_proof = prove_batched_wht_relations_sumcheck::<Fr, Blake2s>(
            &inputs,
            &outputs,
            &mut prover_transcript,
        )
        .unwrap();
        let proof = prover_transcript.into_proof();
        let mut verifier_transcript = TestTranscript::from_proof((), proof.as_slice());
        let (ok, verifier_proof) = verify_batched_wht_relations_sumcheck::<Fr, Blake2s>(
            num_vars,
            num_relations,
            &mut verifier_transcript,
        )
        .unwrap();
        assert!(ok);
        assert_eq!(prover_proof.eta, verifier_proof.eta);
    }

    #[test]
    fn test_qabase_basefold_batch_open_isolated() {
        type Pcs = Basefold<Fr, Blake2s, QABaseFoldConfig>;

        let num_vars = 6;
        let poly_size = 1usize << num_vars;
        let batch_size = 1usize;
        let mut rng = ChaCha8Rng::from_seed([41u8; 32]);
        let param = Pcs::setup(poly_size, batch_size, &mut rng).unwrap();
        let (pp, vp) = Pcs::trim(&param, poly_size, batch_size).unwrap();
        let evals_vec = (0..poly_size).map(|_| Fr::random(&mut rng)).collect::<Vec<_>>();
        let poly = MultilinearPolynomial::new(evals_vec.clone());
        let point_0 = (0..num_vars).map(|_| Fr::random(&mut rng)).collect::<Vec<_>>();
        let point_1 = (0..num_vars).map(|_| Fr::random(&mut rng)).collect::<Vec<_>>();
        let value_0 = eval_mle_from_evals(&evals_vec, &point_0);
        let value_1 = eval_mle_from_evals(&evals_vec, &point_1);
        let points = vec![point_0, point_1];
        let eval_claims = vec![Evaluation::new(0, 0, value_0), Evaluation::new(0, 1, value_1)];
        let mut prover_transcript = TestTranscript::new(());
        let comms = Pcs::batch_commit_and_write(
            &pp,
            std::slice::from_ref(&poly),
            &mut prover_transcript,
        )
        .unwrap();
        Pcs::batch_open(
            &pp,
            std::slice::from_ref(&poly),
            comms.iter(),
            &points,
            &eval_claims,
            &mut prover_transcript,
        )
        .unwrap();
        let proof = prover_transcript.into_proof();
        let mut verifier_transcript = TestTranscript::from_proof((), proof.as_slice());
        let verifier_comms = Pcs::read_commitments(&vp, 1, &mut verifier_transcript).unwrap();
        Pcs::batch_verify(
            &vp,
            verifier_comms.iter(),
            &points,
            &eval_claims,
            &mut verifier_transcript,
        )
        .unwrap();
    }

    #[test]
    fn test_qabase_open_scaffold_global_batch_batched_wht() {
        let num_vars = 6;
        let poly_size = 1usize << num_vars;
        let num_rows = 4usize;
        let inverse_rate = 4usize;
        let num_queries = 8usize;
        let mut rng = ChaCha8Rng::from_seed([67u8; 32]);

        let param = setup::<Fr, Blake2s>(
            poly_size,
            1,
            &mut rng,
            Some(num_rows),
            Some(inverse_rate),
            Some(num_queries),
        );
        let (pp, vp) = trim::<Fr, Blake2s>(&param, poly_size, 1);
        let word = (0..num_rows)
            .map(|_| (0..poly_size).map(|_| Fr::random(&mut rng)).collect::<Vec<_>>())
            .collect::<Vec<Vec<Fr>>>();
        let mut prover_transcript = TestTranscript::new(());
        let comm = commit_and_write::<Fr, Blake2s>(&pp, &word, &mut prover_transcript);
        let prover_output = prove_qabase_open_scaffold_global_batch_batched_wht::<Fr, Blake2s>(
            &pp,
            &word,
            &comm,
            &mut prover_transcript,
        )
        .unwrap();

        let proof = prover_transcript.into_proof();
        let mut verifier_transcript = TestTranscript::from_proof((), proof.as_slice());
        let (ok, verifier_output) = verify_qabase_open_scaffold_global_batch_batched_wht::<Fr, Blake2s>(
            &vp,
            &comm,
            &mut verifier_transcript,
        )
        .unwrap();
        assert!(ok);
        assert_eq!(prover_output.folded_witness.row_challenges, verifier_output.row_challenges);
        assert_eq!(prover_output.item1.query_indices, verifier_output.item1.query_indices);
        assert_eq!(prover_output.item1.folded_values, verifier_output.item1.folded_values);
        assert_eq!(prover_output.encoding_relations.batched_wht.gammas.len(), inverse_rate);

        let direct_folded_msg = linear_combine_rows(&word, &prover_output.folded_witness.row_challenges);
        assert_eq!(direct_folded_msg, prover_output.folded_witness.folded_msg);
        let direct_witness = qa_encode_with_witness(&prover_output.folded_witness.folded_msg, &pp.qa_params);
        assert_eq!(direct_witness.codeword, prover_output.folded_witness.folded_qa_witness.codeword);
    }

    #[test]
    fn test_qabase_open_full_global_batch_batched_wht() {
        let num_vars = 6;
        let poly_size = 1usize << num_vars;
        let num_rows = 4usize;
        let inverse_rate = 4usize;
        let num_queries = 8usize;
        let mut rng = ChaCha8Rng::from_seed([87u8; 32]);

        let param = setup::<Fr, Blake2s>(
            poly_size,
            1,
            &mut rng,
            Some(num_rows),
            Some(inverse_rate),
            Some(num_queries),
        );
        let (pp, vp) = trim::<Fr, Blake2s>(&param, poly_size, 1);

        let word = (0..num_rows)
            .map(|_| {
                (0..poly_size)
                    .map(|_| Fr::random(&mut rng))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<Vec<Fr>>>();

        let log_rows = log2_power_of_two(num_rows);
        let full_point = (0..(log_rows + num_vars))
            .map(|_| Fr::random(&mut rng))
            .collect::<Vec<_>>();
        let (z_left, z_right) =
            qabase_split_evaluation_point::<Fr>(&full_point, num_rows, num_vars);

        let row_weights = qabase_row_weights_from_z_left::<Fr>(num_rows, &z_left);
        let eval_msg = linear_combine_rows(&word, &row_weights);
        let claimed_value = eval_mle_from_evals::<Fr>(&eval_msg, &z_right);

        let mut prover_transcript = TestTranscript::new(());
        let comm = commit_and_write::<Fr, Blake2s>(&pp, &word, &mut prover_transcript);

        let prover_output = prove_qabase_open_full_global_batch_batched_wht::<Fr, Blake2s>(
            &pp,
            &word,
            &comm,
            z_left.clone(),
            z_right.clone(),
            claimed_value,
            &mut prover_transcript,
        )
        .unwrap();

        let proof = prover_transcript.into_proof();
        let mut verifier_transcript = TestTranscript::from_proof((), proof.as_slice());

        let (ok, verifier_output) = verify_qabase_open_full_global_batch_batched_wht::<Fr, Blake2s>(
            &vp,
            &comm,
            z_left,
            z_right,
            claimed_value,
            &mut verifier_transcript,
        )
        .unwrap();

        assert!(ok);
        assert!(prover_output.ok_eval_value);
        assert_eq!(
            prover_output.folded_witness.row_challenges,
            verifier_output.row_challenges
        );
        assert_eq!(
            prover_output.item1.query_indices,
            verifier_output.item1.query_indices
        );
        assert_eq!(
            prover_output.item1.folded_values,
            verifier_output.item1.folded_values
        );
        assert_eq!(
            prover_output.item1.opened_columns,
            verifier_output.item1.opened_columns
        );
        assert_eq!(
            prover_output.eval_item1.eval_folded_values,
            verifier_output.eval_item1.eval_folded_values
        );
    }

}
