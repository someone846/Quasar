#![allow(clippy::too_many_arguments)]

use crate::{
    pcs::multilinear::{
        Basefold, BasefoldCommitment, BasefoldExtParams, BasefoldParams, BasefoldProverParams,
        BasefoldVerifierParams,
    },
    pcs::{Evaluation, Point, PolynomialCommitmentScheme},
    poly::multilinear::MultilinearPolynomial,
    util::{
        arithmetic::{Field, PrimeField},
        hash::{Hash, Output, Update},
        transcript::{TranscriptRead, TranscriptWrite},
        Deserialize, DeserializeOwned, Serialize,
    },
    Error,
};

use rayon::prelude::*;
use std::{marker::PhantomData, slice, sync::Arc, time::Instant};

use crate::piop::sum_check::{
    classic::{ClassicSumCheck, EvaluationsProver},
    SumCheck as _, VirtualPolynomial,
};
use crate::util::expression::{Expression, Query, Rotation};

pub type CommitmentChunk<H> = Output<H>;

const QABASE_BASEFOLD_RATE: usize = 1;

// =============================================================================
// Security parameter selection
// =============================================================================

#[derive(Clone, Debug)]
pub struct QABaseSecurityConfig {
    pub total_log_size: usize,
    pub log_rows: usize,
    pub inverse_rate: usize,
    pub field_bits: usize,
    pub security_bits: usize,
    pub distance_failure_bits: usize,
}

impl QABaseSecurityConfig {
    pub fn row_log_size(&self) -> usize {
        assert!(self.total_log_size >= self.log_rows);
        self.total_log_size - self.log_rows
    }

    pub fn num_rows(&self) -> usize {
        1usize << self.log_rows
    }

    pub fn row_size(&self) -> usize {
        1usize << self.row_log_size()
    }

    pub fn distance(&self) -> f64 {
        qabase_distance_lower_bound(
            self.row_log_size(),
            self.inverse_rate,
            self.field_bits,
            self.distance_failure_bits,
        )
    }

    pub fn num_queries(&self) -> usize {
        qabase_queries_from_distance(self.distance(), self.security_bits)
    }
}

pub fn qabase_gp(delta: f64, field_bits: usize) -> f64 {
    assert!(delta > 0.0 && delta < 1.0);
    let bits = field_bits as f64;
    1.0 - delta + (delta * delta.log2() + (1.0 - delta) * (1.0 - delta).log2()) / bits
}

fn log2_add(a: f64, b: f64) -> f64 {
    let m = a.max(b);
    if !m.is_finite() {
        return m;
    }
    m + ((2.0f64).powf(a - m) + (2.0f64).powf(b - m)).log2()
}

pub fn qabase_distance_failure_log2(
    delta: f64,
    row_log_size: usize,
    inverse_rate: usize,
    field_bits: usize,
) -> f64 {
    let c = inverse_rate;
    assert!(c >= 2);

    let log_n = row_log_size as f64;
    let log_p = field_bits as f64;
    let log_p_minus_one = field_bits as f64;
    let eps = qabase_gp(delta, field_bits) - (1.0 + log_n / log_p) / (c as f64);

    if eps <= 0.0 {
        return f64::INFINITY;
    }

    let denom_log = if log_p * (c as f64) * eps < 60.0 {
        let x = log_p * (c as f64) * eps;
        (-(-std::f64::consts::LN_2 * x).exp_m1()).log2()
    } else {
        0.0
    };

    let log_term1_a = ((c * (c - 1)) as f64 / 2.0).log2() + log_n - 2.0 * log_p;
    let threshold1 = (((c - 1) as f64) / ((c as f64) * delta)).ceil();
    let log_term1_b = -log_p * threshold1 * (c as f64) * eps - denom_log - log_p_minus_one;
    let log_bound1 = log2_add(log_term1_a, log_term1_b);

    let log_term2_a = (c as f64).log2() + log_n - log_p;
    let threshold2 = (1.0 / delta).ceil();
    let log_term2_b = -log_p * threshold2 * (c as f64) * eps - denom_log - log_p_minus_one;
    let log_bound2 = log2_add(log_term2_a, log_term2_b);

    log_bound1.min(log_bound2)
}

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

pub fn qabase_queries_from_distance(delta: f64, security_bits: usize) -> usize {
    assert!(delta > 0.0 && delta < 1.0);
    let denom = -(1.0 - delta / 3.0).log2();
    ((security_bits as f64) / denom).ceil() as usize
}

pub fn setup_from_security_config<F, H>(
    cfg: &QABaseSecurityConfig,
    rng: impl rand_chacha::rand_core::RngCore,
) -> QABaseParams<F>
where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
{
    let delta = cfg.distance();
    let num_queries = qabase_queries_from_distance(delta, cfg.security_bits);
    println!(
        "Quasar security config: total=2^{}, row=2^{}, rows={}, c={}, delta={:.6}, queries={}",
        cfg.total_log_size,
        cfg.row_log_size(),
        cfg.num_rows(),
        cfg.inverse_rate,
        delta,
        num_queries,
    );
    setup::<F, H>(
        cfg.row_size(),
        1,
        rng,
        Some(cfg.num_rows()),
        Some(cfg.inverse_rate),
        Some(num_queries),
    )
}

// =============================================================================
// BaseFold configuration
// =============================================================================

#[derive(Clone, Copy, Debug)]
pub struct QABaseFoldConfig;

impl BasefoldExtParams for QABaseFoldConfig {
    fn get_reps() -> usize {
        249
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

// =============================================================================
// QA encoding
// =============================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: DeserializeOwned"))]
pub struct QAParams<F: PrimeField> {
    pub inverse_rate: usize,
    pub e: Vec<Vec<F>>,
}

impl<F: PrimeField> QAParams<F> {
    pub fn new_random(
        msg_len: usize,
        inverse_rate: usize,
        rng: &mut impl rand_chacha::rand_core::RngCore,
    ) -> Self {
        assert!(msg_len.is_power_of_two());
        assert!(inverse_rate >= 2 && inverse_rate.is_power_of_two());
        let e = (0..inverse_rate - 1)
            .map(|_| {
                (0..msg_len)
                    .map(|_| F::random(&mut *rng))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        Self { inverse_rate, e }
    }
}

pub fn wht<F: Field>(x: &mut [F]) {
    let len = x.len();
    assert!(len.is_power_of_two());
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

pub fn wht_parallel<F>(x: &mut [F])
where
    F: Field + Send + Sync,
{
    let len = x.len();
    assert!(len.is_power_of_two());

    let num_threads = rayon::current_num_threads();
    let mut step = 1usize;
    while 2 * step <= len {
        let chunk_len = 2 * step;
        let num_chunks = len / chunk_len;

        if num_chunks >= 4 * num_threads {
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

/// Commitment-only encoder.  It never constructs the old `QAWitness`.
pub fn qa_encode_codeword_only<F>(msg: &[F], params: &QAParams<F>) -> Vec<F>
where
    F: PrimeField,
{
    let n = msg.len();
    let rho = params.inverse_rate;
    assert!(n.is_power_of_two());
    assert_eq!(params.e.len(), rho - 1);

    let mut middle = msg.to_vec();
    wht(&mut middle);

    let mut codeword = Vec::with_capacity(rho * n);
    codeword.extend_from_slice(msg);

    for i in 0..rho - 1 {
        let mut block = if i + 1 == rho - 1 {
            core::mem::take(&mut middle)
        } else {
            middle.clone()
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

/// Build one endpoint-only GKR instance.
///
/// Returned endpoint polynomials own the input and parity blocks.  The only
/// non-endpoint witness retained is `middle = WHT(input)`.  There are no
/// scaled blocks and no concatenated codeword copy.
fn qa_encode_endpoint_instance<F>(
    input: Vec<F>,
    params: &QAParams<F>,
) -> (Vec<MultilinearPolynomial<F>>, Vec<F>)
where
    F: PrimeField + Send + Sync,
{
    let n = input.len();
    let rho = params.inverse_rate;
    assert!(n.is_power_of_two());
    assert_eq!(params.e.len(), rho - 1);

    let mut middle = input.clone();
    wht_parallel(&mut middle);

    let parity_blocks = params
        .e
        .par_iter()
        .map(|coeffs| {
            let mut parity = middle.clone();
            parity
                .par_iter_mut()
                .zip(coeffs.par_iter())
                .for_each(|(x, e)| *x *= *e);
            wht_parallel(&mut parity);
            parity
        })
        .collect::<Vec<_>>();

    let mut blocks = Vec::with_capacity(rho);
    blocks.push(MultilinearPolynomial::new(input));
    blocks.extend(parity_blocks.into_iter().map(MultilinearPolynomial::new));
    debug_assert_eq!(blocks.len(), rho);
    (blocks, middle)
}

// =============================================================================
// Merkle helpers
// =============================================================================

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

/// Column Merkle tree without a temporary `Vec<F>` for every leaf.
///
/// The previous implementation allocated one short vector per column.  At
/// total_k=28, c=2 this meant 2^23 heap allocations.  Streaming the row values
/// directly into the hasher preserves the exact leaf hash and Merkle root.
pub fn merkelize_long<H, F>(codeword: &impl QACodewordRows<F>) -> Vec<Vec<Output<H>>>
where
    H: Hash,
    F: PrimeField,
{
    assert!(codeword.num_rows() > 0);
    let num_cols = codeword.num_cols();
    assert!(num_cols.is_power_of_two());
    for row_index in 0..codeword.num_rows() {
        assert_eq!(codeword.row(row_index).len(), num_cols);
    }

    let leaves = (0..num_cols)
        .into_par_iter()
        .map(|col| {
            let mut hasher = H::new();
            for row_index in 0..codeword.num_rows() {
                hasher.update_field_element(&codeword.row(row_index)[col]);
            }
            hasher.finalize_fixed()
        })
        .collect::<Vec<_>>();

    merkelize_from_leaves::<H>(leaves)
}

/// Builds all upper Merkle layers from externally computed leaf digests.
///
/// The hybrid CUDA backend hashes encoded columns on the GPU, transfers only
/// these fixed-size digests, and reuses this CPU reduction so the root, paths,
/// and proof format remain byte-for-byte compatible with the default backend.
pub fn merkelize_from_leaves<H>(leaves: Vec<Output<H>>) -> Vec<Vec<Output<H>>>
where
    H: Hash,
{
    assert!(!leaves.is_empty());
    assert!(leaves.len().is_power_of_two());
    let mut tree = vec![leaves];
    while tree.last().expect("tree has leaves").len() > 1 {
        let prev = tree.last().expect("tree has previous layer");
        let next = prev
            .par_chunks_exact(2)
            .map(|pair| hash_hash_pair::<H>(&pair[0], &pair[1]))
            .collect::<Vec<_>>();
        tree.push(next);
    }
    tree
}

fn write_merkle_path<H, F>(
    tree: &[Vec<Output<H>>],
    index: usize,
    transcript: &mut impl TranscriptWrite<CommitmentChunk<H>, F>,
) -> Result<(), Error>
where
    H: Hash,
    F: PrimeField,
{
    if tree.is_empty() || index >= tree[0].len() {
        return Err(Error::InvalidPcsOpen(
            "invalid Quasar Merkle path request".to_string(),
        ));
    }
    let mut idx = index;
    for nodes in tree.iter().take(tree.len() - 1) {
        let sibling_idx = if idx % 2 == 0 { idx + 1 } else { idx - 1 };
        transcript.write_commitment(&nodes[sibling_idx])?;
        idx >>= 1;
    }
    Ok(())
}

fn read_merkle_path<H, F>(
    num_leaves: usize,
    transcript: &mut impl TranscriptRead<CommitmentChunk<H>, F>,
) -> Result<Vec<Output<H>>, Error>
where
    H: Hash,
    F: PrimeField,
{
    if !num_leaves.is_power_of_two() {
        return Err(Error::InvalidPcsOpen(
            "invalid Quasar Merkle leaf count".to_string(),
        ));
    }
    let depth = num_leaves.trailing_zeros() as usize;
    (0..depth)
        .map(|_| transcript.read_commitment())
        .collect::<Result<Vec<_>, _>>()
}

fn authenticate_merkle_path<H>(leaf: &Output<H>, path: &[Output<H>], index: usize) -> Output<H>
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

fn verify_merkle_path_with_leaf<H, F>(
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
    &authenticate_merkle_path::<H>(&leaf, path, index) == root
}

fn field_challenge_to_index<F>(challenge: &F, num_cols: usize) -> usize
where
    F: PrimeField,
{
    assert!(num_cols > 0);
    let repr = challenge.to_repr();
    let bytes = repr.as_ref();
    let mut acc = 0usize;
    let take = core::cmp::min(bytes.len(), core::mem::size_of::<usize>());
    for (i, byte) in bytes.iter().take(take).enumerate() {
        acc |= (*byte as usize) << (8 * i);
    }
    acc % num_cols
}

// =============================================================================
// Parameters, setup, trim and matrix commitment
// =============================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: DeserializeOwned"))]
pub struct QABaseParams<F: PrimeField> {
    pub qa_params: QAParams<F>,
    pub num_vars: usize,
    pub num_rows: usize,
    pub inverse_rate: usize,
    pub num_queries: usize,
    pub basefold_params: BasefoldParams<F>,
}

#[derive(Clone, Debug)]
pub struct QABaseProverParams<F: PrimeField, H: Hash> {
    pub qa_params: Arc<QAParams<F>>,
    pub num_vars: usize,
    pub num_rows: usize,
    pub inverse_rate: usize,
    pub num_queries: usize,
    pub basefold_prover_param: BasefoldProverParams<F>,
    pub e_polys: Vec<MultilinearPolynomial<F>>,
    pub e_commitments: Arc<Vec<BasefoldCommitment<F, H>>>,
}

#[derive(Clone, Debug)]
pub struct QABaseVerifierParams<F: PrimeField, H: Hash> {
    pub qa_params: Arc<QAParams<F>>,
    pub num_vars: usize,
    pub num_rows: usize,
    pub inverse_rate: usize,
    pub num_queries: usize,
    pub basefold_verifier_param: BasefoldVerifierParams<F>,
    pub e_commitments: Arc<Vec<BasefoldCommitment<F, H>>>,
}

/// Row-oriented access to an encoded QA matrix.
///
/// The default CPU commitment stores `Vec<Vec<F>>`.  CUDA implementations can
/// keep one contiguous pinned allocation and implement this trait without a
/// second host-side copy.
pub trait QACodewordRows<F>: Send + Sync {
    fn num_rows(&self) -> usize;
    fn num_cols(&self) -> usize;
    fn row(&self, row_index: usize) -> &[F];
}

/// Column-oriented access used after Fiat--Shamir selects Merkle queries.
/// Device-resident implementations can gather and transfer only a requested
/// column instead of materializing the complete encoded matrix on the host.
pub trait QACodewordColumns<F>: Send + Sync {
    fn row_count(&self) -> usize;
    fn column_count(&self) -> usize;
    fn read_column(&self, column_index: usize) -> Result<Vec<F>, String>;
}

impl<F> QACodewordRows<F> for Vec<Vec<F>>
where
    F: Send + Sync,
{
    fn num_rows(&self) -> usize {
        self.len()
    }

    fn num_cols(&self) -> usize {
        self.first().map_or(0, Vec::len)
    }

    fn row(&self, row_index: usize) -> &[F] {
        &self[row_index]
    }
}

impl<F> QACodewordRows<F> for [Vec<F>]
where
    F: Send + Sync,
{
    fn num_rows(&self) -> usize {
        self.len()
    }

    fn num_cols(&self) -> usize {
        self.first().map_or(0, Vec::len)
    }

    fn row(&self, row_index: usize) -> &[F] {
        &self[row_index]
    }
}

impl<F> QACodewordColumns<F> for Vec<Vec<F>>
where
    F: Copy + Send + Sync,
{
    fn row_count(&self) -> usize {
        self.len()
    }

    fn column_count(&self) -> usize {
        self.first().map_or(0, Vec::len)
    }

    fn read_column(&self, column_index: usize) -> Result<Vec<F>, String> {
        if column_index >= self.first().map_or(0, Vec::len) {
            return Err("Quasar column index is out of bounds".to_owned());
        }
        Ok(self.iter().map(|row| row[column_index]).collect())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: Serialize, C: Serialize",
    deserialize = "F: DeserializeOwned, C: DeserializeOwned"
))]
pub struct QABaseCommitment<F, H, C = Vec<Vec<F>>>
where
    F: PrimeField,
    H: Hash,
{
    pub codeword: C,
    pub codeword_tree: Vec<Vec<Output<H>>>,
    #[serde(skip)]
    field_marker: PhantomData<F>,
}

impl<F, H, C> QABaseCommitment<F, H, C>
where
    F: PrimeField,
    H: Hash,
{
    pub fn into_codeword(self) -> C {
        self.codeword
    }
}

impl<F, H, C> AsRef<[Output<H>]> for QABaseCommitment<F, H, C>
where
    F: PrimeField,
    H: Hash,
{
    fn as_ref(&self) -> &[Output<H>] {
        let root = &self.codeword_tree[self.codeword_tree.len() - 1][0];
        slice::from_ref(root)
    }
}

impl<F, H, C> AsRef<Output<H>> for QABaseCommitment<F, H, C>
where
    F: PrimeField,
    H: Hash,
{
    fn as_ref(&self) -> &Output<H> {
        &self.codeword_tree[self.codeword_tree.len() - 1][0]
    }
}

pub fn setup<F, H>(
    poly_size: usize,
    _batch_size: usize,
    mut rng: impl rand_chacha::rand_core::RngCore,
    num_rows: Option<usize>,
    inverse_rate: Option<usize>,
    num_queries: Option<usize>,
) -> QABaseParams<F>
where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
{
    assert!(poly_size.is_power_of_two());
    type Pcs<F, H> = Basefold<F, H, QABaseFoldConfig>;

    let num_queries = num_queries.unwrap_or(504);
    let num_rows = num_rows.unwrap_or(64);
    let inverse_rate = inverse_rate.unwrap_or(2);
    assert!(num_rows.is_power_of_two());
    assert!(inverse_rate >= 2 && inverse_rate.is_power_of_two());

    let num_vars = poly_size.trailing_zeros() as usize;
    let qa_params = QAParams::<F>::new_random(poly_size, inverse_rate, &mut rng);
    let basefold_params =
        Pcs::<F, H>::setup(poly_size, 1, &mut rng).expect("BaseFold setup failed");

    QABaseParams {
        qa_params,
        num_vars,
        num_rows,
        inverse_rate,
        num_queries,
        basefold_params,
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
        Pcs::<F, H>::trim(&param.basefold_params, poly_size, batch_size)
            .expect("BaseFold trim failed");

    let e_polys = param
        .qa_params
        .e
        .iter()
        .cloned()
        .map(MultilinearPolynomial::new)
        .collect::<Vec<_>>();

    let now = Instant::now();
    let e_commitments = Arc::new(
        Pcs::<F, H>::batch_commit(&basefold_pp, e_polys.iter())
            .expect("failed to commit public QA coefficient polynomials"),
    );
    println!(
        "Quasar indexing E commitments {:?}, blocks {}",
        now.elapsed(),
        e_commitments.len(),
    );

    let qa_params = Arc::new(param.qa_params.clone());
    (
        QABaseProverParams {
            qa_params: qa_params.clone(),
            num_vars: param.num_vars,
            num_rows: param.num_rows,
            inverse_rate: param.inverse_rate,
            num_queries: param.num_queries,
            basefold_prover_param: basefold_pp,
            e_polys,
            e_commitments: e_commitments.clone(),
        },
        QABaseVerifierParams {
            qa_params,
            num_vars: param.num_vars,
            num_rows: param.num_rows,
            inverse_rate: param.inverse_rate,
            num_queries: param.num_queries,
            basefold_verifier_param: basefold_vp,
            e_commitments,
        },
    )
}

pub fn commit_and_write<F, H>(
    pp: &QABaseProverParams<F, H>,
    word: &(impl QACodewordRows<F> + ?Sized),
    transcript: &mut impl TranscriptWrite<CommitmentChunk<H>, F>,
) -> QABaseCommitment<F, H>
where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
{
    assert_eq!(word.num_rows(), pp.num_rows);
    assert_eq!(word.num_cols(), 1usize << pp.num_vars);
    for i in 0..word.num_rows() {
        assert_eq!(word.row(i).len(), word.num_cols());
    }

    let now = Instant::now();
    let codeword = (0..word.num_rows())
        .into_par_iter()
        .map(|i| qa_encode_codeword_only(word.row(i), &pp.qa_params))
        .collect::<Vec<_>>();
    println!("degree {}, QA encode {:?}", pp.num_vars, now.elapsed(),);

    commit_encoded_and_write(pp, codeword, transcript)
}

/// Finish a Quasar commitment from an already encoded row matrix.
///
/// This is the shared CPU/CUDA boundary: both backends use the same column
/// hashing, Merkle tree, transcript write, opening logic, and proof format.
pub fn commit_encoded_and_write<F, H, C>(
    pp: &QABaseProverParams<F, H>,
    codeword: C,
    transcript: &mut impl TranscriptWrite<CommitmentChunk<H>, F>,
) -> QABaseCommitment<F, H, C>
where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
    C: QACodewordRows<F> + QACodewordColumns<F>,
{
    assert_eq!(QACodewordRows::num_rows(&codeword), pp.num_rows);
    assert_eq!(
        QACodewordRows::num_cols(&codeword),
        pp.inverse_rate * (1usize << pp.num_vars),
    );

    let now = Instant::now();
    let codeword_tree = merkelize_long::<H, F>(&codeword);
    println!(
        "degree {}, QA column Merkle {:?}",
        pp.num_vars,
        now.elapsed(),
    );

    commit_tree_and_write(pp, codeword, codeword_tree, transcript)
}

/// Finishes a Quasar commitment from an encoded-storage handle and a complete
/// Merkle tree. A hybrid backend uses this entry point after computing leaf
/// hashes outside Rust while retaining the standard CPU upper-tree reduction.
pub fn commit_tree_and_write<F, H, C>(
    pp: &QABaseProverParams<F, H>,
    codeword: C,
    codeword_tree: Vec<Vec<Output<H>>>,
    transcript: &mut impl TranscriptWrite<CommitmentChunk<H>, F>,
) -> QABaseCommitment<F, H, C>
where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
    C: QACodewordColumns<F>,
{
    assert_eq!(codeword.row_count(), pp.num_rows);
    assert_eq!(
        codeword.column_count(),
        pp.inverse_rate * (1usize << pp.num_vars),
    );
    assert!(!codeword_tree.is_empty());
    assert_eq!(codeword_tree[0].len(), codeword.column_count());
    assert_eq!(codeword_tree.last().expect("tree has a root").len(), 1);

    transcript
        .write_commitment(&codeword_tree[codeword_tree.len() - 1][0])
        .expect("failed to write Quasar root");

    QABaseCommitment {
        codeword,
        codeword_tree,
        field_marker: PhantomData,
    }
}

// =============================================================================
// Generic MLE helpers
// =============================================================================

fn log2_power_of_two(x: usize) -> usize {
    assert!(x.is_power_of_two());
    x.trailing_zeros() as usize
}

pub fn equality_mle_eval_at_index<F>(index: usize, num_vars: usize, point: &[F]) -> F
where
    F: PrimeField,
{
    assert_eq!(point.len(), num_vars);
    assert!(index < (1usize << num_vars));
    let mut acc = F::ONE;
    for (i, r) in point.iter().enumerate() {
        acc *= if ((index >> i) & 1) == 1 {
            *r
        } else {
            F::ONE - *r
        };
    }
    acc
}

pub fn hadamard_tensor_mle_eval<F>(a: &[F], b: &[F]) -> F
where
    F: PrimeField,
{
    assert_eq!(a.len(), b.len());
    let two = F::from(2u64);
    a.iter()
        .zip(b.iter())
        .fold(F::ONE, |acc, (x, y)| acc * (F::ONE - two * *x * *y))
}

pub fn hadamard_tensor_mle_evals_on_hypercube<F>(gamma: &[F]) -> Vec<F>
where
    F: PrimeField,
{
    let two = F::from(2u64);
    let mut evals = vec![F::ONE];
    for &g in gamma {
        let factor = F::ONE - two * g;
        let old_len = evals.len();
        evals.resize(old_len << 1, F::ZERO);
        for i in 0..old_len {
            evals[old_len + i] = evals[i] * factor;
        }
    }
    evals
}

pub fn eval_mle_from_evals<F>(evals: &[F], point: &[F]) -> F
where
    F: PrimeField,
{
    assert!(evals.len().is_power_of_two());
    assert_eq!(evals.len(), 1usize << point.len());
    let poly = MultilinearPolynomial::new(evals.to_vec());
    poly.evaluate(point)
}

fn eval_poly<F: PrimeField>(poly: &MultilinearPolynomial<F>, point: &[F]) -> F {
    poly.evaluate(point)
}

pub fn qabase_row_weights_from_z_left<F>(num_rows: usize, z_left: &[F]) -> Vec<F>
where
    F: PrimeField,
{
    let log_rows = log2_power_of_two(num_rows);
    assert_eq!(z_left.len(), log_rows);
    (0..num_rows)
        .map(|i| equality_mle_eval_at_index(i, log_rows, z_left))
        .collect()
}

fn fold_opened_column_pair<F>(
    opened_column: &[F],
    proximity_weights: &[F],
    evaluation_weights: &[F],
) -> (F, F)
where
    F: PrimeField,
{
    assert_eq!(opened_column.len(), proximity_weights.len());
    assert_eq!(opened_column.len(), evaluation_weights.len());
    let mut proximity = F::ZERO;
    let mut evaluation = F::ZERO;
    for ((value, r), z) in opened_column
        .iter()
        .zip(proximity_weights.iter())
        .zip(evaluation_weights.iter())
    {
        proximity += *r * *value;
        evaluation += *z * *value;
    }
    (proximity, evaluation)
}

/// One matrix scan computes both switched rows.
fn linear_combine_rows_pair_parallel<F>(
    rows: &(impl QACodewordRows<F> + ?Sized),
    weights_a: &[F],
    weights_b: &[F],
) -> (Vec<F>, Vec<F>)
where
    F: PrimeField + Send + Sync,
{
    let num_rows = rows.num_rows();
    assert!(num_rows > 0);
    assert_eq!(num_rows, weights_a.len());
    assert_eq!(num_rows, weights_b.len());
    let row_len = rows.num_cols();
    for i in 0..num_rows {
        assert_eq!(rows.row(i).len(), row_len);
    }

    let mut out_a = vec![F::ZERO; row_len];
    let mut out_b = vec![F::ZERO; row_len];
    out_a
        .par_iter_mut()
        .zip(out_b.par_iter_mut())
        .enumerate()
        .for_each(|(j, (a, b))| {
            let mut acc_a = F::ZERO;
            let mut acc_b = F::ZERO;
            for i in 0..num_rows {
                let value = rows.row(i)[j];
                acc_a += weights_a[i] * value;
                acc_b += weights_b[i] * value;
            }
            *a = acc_a;
            *b = acc_b;
        });
    (out_a, out_b)
}

/// Correct row-major variable split.
///
/// `eval_mle_from_evals` folds adjacent entries first, hence its first
/// coordinates address the low bits.  For row-major flattening
/// `flat[row * row_size + col]`, low bits are column variables.
pub fn qabase_split_evaluation_point<F>(
    point: &[F],
    num_rows: usize,
    num_vars: usize,
) -> (Vec<F>, Vec<F>)
where
    F: PrimeField,
{
    let log_rows = log2_power_of_two(num_rows);
    assert_eq!(point.len(), num_vars + log_rows);
    let z_right = point[..num_vars].to_vec();
    let z_left = point[num_vars..].to_vec();
    (z_left, z_right)
}

// =============================================================================
// Endpoint-only BaseFold data and opening accumulator
// =============================================================================

#[derive(Clone, Debug)]
pub struct QABaseEndpointCommitments<F, H>
where
    F: PrimeField,
    H: Hash,
{
    pub num_instances: usize,
    pub block_commitments: Vec<BasefoldCommitment<F, H>>,
}

pub struct QABaseEndpointProverData<F, H>
where
    F: PrimeField,
    H: Hash,
{
    pub commitments: QABaseEndpointCommitments<F, H>,
    pub polys: Vec<MultilinearPolynomial<F>>,
    /// `WHT(input_b)`, moved into the outer sumcheck and then released.
    pub middle_evals: Vec<Vec<F>>,
}

#[derive(Clone, Debug)]
pub struct QABaseTwoLayerGkrLayout {
    pub rho: usize,
    pub num_instances: usize,
    pub block_starts: Vec<usize>,
    pub e_start: usize,
}

impl QABaseTwoLayerGkrLayout {
    pub fn block_poly_index(&self, instance: usize, block: usize) -> usize {
        assert!(instance < self.num_instances && block < self.rho);
        self.block_starts[instance] + block
    }

    pub fn e_poly_index(&self, i: usize) -> usize {
        assert!(i + 1 < self.rho);
        self.e_start + i
    }
}

fn commit_endpoint_instances_with_basefold<F, H>(
    pp: &QABaseProverParams<F, H>,
    inputs: Vec<Vec<F>>,
    transcript: &mut impl TranscriptWrite<CommitmentChunk<H>, F>,
) -> Result<QABaseEndpointProverData<F, H>, Error>
where
    F: PrimeField + Serialize + DeserializeOwned + Send + Sync,
    H: Hash,
{
    type Pcs<F, H> = Basefold<F, H, QABaseFoldConfig>;

    let num_instances = inputs.len();
    if num_instances == 0 {
        return Err(Error::InvalidPcsOpen(
            "Quasar needs at least one endpoint instance".to_string(),
        ));
    }

    let encoded = inputs
        .into_par_iter()
        .map(|input| qa_encode_endpoint_instance(input, &pp.qa_params))
        .collect::<Vec<_>>();

    let mut polys = Vec::with_capacity(num_instances * pp.inverse_rate);
    let mut middle_evals = Vec::with_capacity(num_instances);
    for (blocks, middle) in encoded {
        if blocks.len() != pp.inverse_rate {
            return Err(Error::InvalidPcsOpen(
                "wrong Quasar endpoint block count".to_string(),
            ));
        }
        polys.extend(blocks);
        middle_evals.push(middle);
    }

    let commitments =
        Pcs::<F, H>::batch_commit_and_write(&pp.basefold_prover_param, polys.iter(), transcript)?;

    if commitments.len() != polys.len() {
        return Err(Error::InvalidPcsOpen(
            "wrong BaseFold endpoint commitment count".to_string(),
        ));
    }

    Ok(QABaseEndpointProverData {
        commitments: QABaseEndpointCommitments {
            num_instances,
            block_commitments: commitments,
        },
        polys,
        middle_evals,
    })
}

fn read_endpoint_block_commitments<F, H>(
    vp: &QABaseVerifierParams<F, H>,
    num_instances: usize,
    transcript: &mut impl TranscriptRead<CommitmentChunk<H>, F>,
) -> Result<QABaseEndpointCommitments<F, H>, Error>
where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
{
    type Pcs<F, H> = Basefold<F, H, QABaseFoldConfig>;
    let count = num_instances * vp.inverse_rate;
    let commitments =
        Pcs::<F, H>::read_commitments(&vp.basefold_verifier_param, count, transcript)?;
    if commitments.len() != count {
        return Err(Error::InvalidPcsOpen(
            "wrong endpoint commitment count".to_string(),
        ));
    }
    Ok(QABaseEndpointCommitments {
        num_instances,
        block_commitments: commitments,
    })
}

pub struct QABaseProverOpeningAccumulator<'a, F, H>
where
    F: PrimeField,
    H: Hash,
{
    pub polys: Vec<&'a MultilinearPolynomial<F>>,
    pub comms: Vec<&'a BasefoldCommitment<F, H>>,
    pub points: Vec<Point<F, MultilinearPolynomial<F>>>,
    pub evals: Vec<Evaluation<F>>,
}

pub struct QABaseVerifierOpeningAccumulator<'a, F, H>
where
    F: PrimeField,
    H: Hash,
{
    pub comms: Vec<&'a BasefoldCommitment<F, H>>,
    pub points: Vec<Point<F, MultilinearPolynomial<F>>>,
    pub evals: Vec<Evaluation<F>>,
}

fn build_two_layer_gkr_prover_accumulator<'a, F, H>(
    endpoint_data: &'a QABaseEndpointProverData<F, H>,
    pp: &'a QABaseProverParams<F, H>,
) -> (
    QABaseProverOpeningAccumulator<'a, F, H>,
    QABaseTwoLayerGkrLayout,
)
where
    F: PrimeField,
    H: Hash,
{
    let rho = pp.inverse_rate;
    let k = endpoint_data.commitments.num_instances;
    assert_eq!(endpoint_data.polys.len(), k * rho);

    let mut polys = Vec::with_capacity(k * rho + rho - 1);
    polys.extend(endpoint_data.polys.iter());
    polys.extend(pp.e_polys.iter());

    let mut comms = Vec::with_capacity(k * rho + rho - 1);
    comms.extend(endpoint_data.commitments.block_commitments.iter());
    comms.extend(pp.e_commitments.iter());

    let layout = QABaseTwoLayerGkrLayout {
        rho,
        num_instances: k,
        block_starts: (0..k).map(|b| b * rho).collect(),
        e_start: k * rho,
    };

    (
        QABaseProverOpeningAccumulator {
            polys,
            comms,
            points: Vec::new(),
            evals: Vec::new(),
        },
        layout,
    )
}

fn build_two_layer_gkr_verifier_accumulator<'a, F, H>(
    endpoints: &'a QABaseEndpointCommitments<F, H>,
    vp: &'a QABaseVerifierParams<F, H>,
) -> (
    QABaseVerifierOpeningAccumulator<'a, F, H>,
    QABaseTwoLayerGkrLayout,
)
where
    F: PrimeField,
    H: Hash,
{
    let rho = vp.inverse_rate;
    let k = endpoints.num_instances;

    let mut comms = Vec::with_capacity(k * rho + rho - 1);
    comms.extend(endpoints.block_commitments.iter());
    comms.extend(vp.e_commitments.iter());

    let layout = QABaseTwoLayerGkrLayout {
        rho,
        num_instances: k,
        block_starts: (0..k).map(|b| b * rho).collect(),
        e_start: k * rho,
    };

    (
        QABaseVerifierOpeningAccumulator {
            comms,
            points: Vec::new(),
            evals: Vec::new(),
        },
        layout,
    )
}

/// Intern identical opening points so BaseFold can merge claims at the same
/// point instead of receiving one duplicate point entry per polynomial.
fn push_basefold_opening_claim<F>(
    points: &mut Vec<Point<F, MultilinearPolynomial<F>>>,
    evals: &mut Vec<Evaluation<F>>,
    poly_index: usize,
    point: Vec<F>,
    value: F,
) where
    F: PrimeField,
{
    let point_index = match points.iter().position(|existing| existing == &point) {
        Some(index) => index,
        None => {
            points.push(point);
            points.len() - 1
        }
    };
    evals.push(Evaluation::new(poly_index, point_index, value));
}

// =============================================================================
// Two-layer GKR certification
// =============================================================================

#[derive(Clone, Debug)]
pub struct QABaseTwoLayerGkrProof<F: PrimeField> {
    pub num_instances: usize,
    pub block_batch_challenge: F,
    pub instance_batch_challenge: F,
    pub gammas: Vec<Vec<Vec<F>>>,
    pub output_evals_at_gammas: Vec<Vec<F>>,
    pub claimed_output_batch: F,
    pub outer_point: Vec<F>,
    pub outer_terminal_eval: F,
    pub e_evals_at_outer_point: Vec<F>,
    pub instance_weights_at_outer_point: Vec<F>,
    pub inner_point: Vec<F>,
    pub input_evals_at_inner_point: Vec<F>,
    pub inner_terminal_eval: F,
}

fn powers<F: PrimeField>(base: F, count: usize) -> Vec<F> {
    let mut out = Vec::with_capacity(count);
    let mut power = F::ONE;
    for _ in 0..count {
        out.push(power);
        power *= base;
    }
    out
}

fn qabase_gkr_instance_weights<F: PrimeField>(
    gammas: &[Vec<Vec<F>>],
    block_powers: &[F],
    instance_powers: &[F],
    outer_point: &[F],
    e_evals: &[F],
) -> Vec<F> {
    gammas
        .iter()
        .enumerate()
        .map(|(b, per_instance)| {
            per_instance
                .iter()
                .enumerate()
                .fold(F::ZERO, |acc, (i, gamma)| {
                    acc + instance_powers[b]
                        * block_powers[i]
                        * hadamard_tensor_mle_eval(gamma, outer_point)
                        * e_evals[i]
                })
        })
        .collect()
}

fn prove_qabase_two_layer_gkr_batch<F, H>(
    pp: &QABaseProverParams<F, H>,
    endpoint_data: &mut QABaseEndpointProverData<F, H>,
    layout: &QABaseTwoLayerGkrLayout,
    transcript: &mut impl TranscriptWrite<CommitmentChunk<H>, F>,
) -> Result<QABaseTwoLayerGkrProof<F>, Error>
where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
{
    type Sc<F> = ClassicSumCheck<EvaluationsProver<F>>;

    let k = layout.num_instances;
    let rho = layout.rho;
    let num_parity = rho - 1;
    let num_vars = pp.num_vars;
    if k == 0 || endpoint_data.middle_evals.len() != k {
        return Err(Error::InvalidPcsOpen(
            "invalid two-layer GKR instance count".to_string(),
        ));
    }

    let block_batch_challenge = transcript.squeeze_challenges(1)[0];
    let instance_batch_challenge = if k > 1 {
        transcript.squeeze_challenges(1)[0]
    } else {
        F::ONE
    };
    let block_powers = powers(block_batch_challenge, num_parity);
    let instance_powers = powers(instance_batch_challenge, k);

    let mut gammas = Vec::with_capacity(k);
    for _ in 0..k {
        let mut per_instance = Vec::with_capacity(num_parity);
        for _ in 0..num_parity {
            per_instance.push(transcript.squeeze_challenges(num_vars));
        }
        gammas.push(per_instance);
    }

    let mut output_evals_at_gammas = Vec::with_capacity(k);
    let mut claimed_output_batch = F::ZERO;
    for b in 0..k {
        let mut per_instance = Vec::with_capacity(num_parity);
        for i in 0..num_parity {
            let output_poly = &endpoint_data.polys[layout.block_poly_index(b, i + 1)];
            let value = eval_poly(output_poly, &gammas[b][i]);
            transcript.write_field_element(&value)?;
            claimed_output_batch += instance_powers[b] * block_powers[i] * value;
            per_instance.push(value);
        }
        output_evals_at_gammas.push(per_instance);
    }

    let v_start = 0usize;
    let h_start = k;
    let e_start = h_start + k * num_parity;
    let mut outer_owned_polys = Vec::with_capacity(e_start);

    // Move the only non-endpoint witness into the sumcheck.  It is released
    // immediately after the outer sumcheck.
    for middle in endpoint_data.middle_evals.iter_mut() {
        outer_owned_polys.push(MultilinearPolynomial::new(core::mem::take(middle)));
    }

    for b in 0..k {
        for i in 0..num_parity {
            let weight = instance_powers[b] * block_powers[i];
            let mut h = hadamard_tensor_mle_evals_on_hypercube(&gammas[b][i]);
            h.par_iter_mut().for_each(|value| *value *= weight);
            outer_owned_polys.push(MultilinearPolynomial::new(h));
        }
    }

    // The preprocessed E_i polynomials are borrowed directly; they are not
    // cloned into the online sumcheck witness.
    let mut outer_poly_refs = outer_owned_polys.iter().collect::<Vec<_>>();
    outer_poly_refs.extend(pp.e_polys.iter());

    let make_term = |b: usize, i: usize| -> Expression<F> {
        let v: Expression<F> =
            Expression::<F>::Polynomial(Query::new(v_start + b, Rotation::cur()));
        let h: Expression<F> =
            Expression::<F>::Polynomial(Query::new(h_start + b * num_parity + i, Rotation::cur()));
        let e: Expression<F> =
            Expression::<F>::Polynomial(Query::new(e_start + i, Rotation::cur()));
        v * h * e
    };

    let mut outer_expression = make_term(0, 0);
    for b in 0..k {
        for i in 0..num_parity {
            if b != 0 || i != 0 {
                outer_expression = outer_expression + make_term(b, i);
            }
        }
    }

    let no_challenges: Vec<F> = Vec::new();
    let no_ys: Vec<Vec<F>> = Vec::new();
    let outer_virtual_poly =
        VirtualPolynomial::new(&outer_expression, outer_poly_refs, &no_challenges, &no_ys);
    let (outer_point, outer_terminal_evals) = Sc::<F>::prove(
        &(),
        num_vars,
        outer_virtual_poly,
        claimed_output_batch,
        transcript,
    )?;

    if outer_terminal_evals.len() != e_start + num_parity {
        return Err(Error::InvalidSumcheck(
            "wrong Quasar outer terminal vector".to_string(),
        ));
    }

    let e_evals_at_outer_point = outer_terminal_evals[e_start..].to_vec();
    for value in &e_evals_at_outer_point {
        transcript.write_field_element(value)?;
    }

    let mut outer_terminal_eval = F::ZERO;
    for b in 0..k {
        let v_eval = outer_terminal_evals[v_start + b];
        for i in 0..num_parity {
            outer_terminal_eval += v_eval
                * outer_terminal_evals[h_start + b * num_parity + i]
                * outer_terminal_evals[e_start + i];
        }
    }

    let instance_weights_at_outer_point = qabase_gkr_instance_weights(
        &gammas,
        &block_powers,
        &instance_powers,
        &outer_point,
        &e_evals_at_outer_point,
    );

    drop(outer_terminal_evals);
    drop(outer_owned_polys);
    drop(outer_expression);

    let h_outer = MultilinearPolynomial::new(hadamard_tensor_mle_evals_on_hypercube(&outer_point));
    let n = 1usize << num_vars;
    let mut combined_input_evals = vec![F::ZERO; n];
    for b in 0..k {
        let input = &endpoint_data.polys[layout.block_poly_index(b, 0)].evals;
        let weight = instance_weights_at_outer_point[b];
        combined_input_evals
            .par_iter_mut()
            .zip(input.par_iter())
            .for_each(|(out, value)| *out += weight * *value);
    }
    let combined_input = MultilinearPolynomial::new(combined_input_evals);

    let inner_h: Expression<F> = Expression::<F>::Polynomial(Query::new(0, Rotation::cur()));
    let inner_input: Expression<F> = Expression::<F>::Polynomial(Query::new(1, Rotation::cur()));
    let inner_expression: Expression<F> = inner_h * inner_input;
    let inner_polys = vec![h_outer, combined_input];
    let inner_virtual_poly =
        VirtualPolynomial::new(&inner_expression, &inner_polys, &no_challenges, &no_ys);
    let (inner_point, inner_terminal_evals) = Sc::<F>::prove(
        &(),
        num_vars,
        inner_virtual_poly,
        outer_terminal_eval,
        transcript,
    )?;
    if inner_terminal_evals.len() != 2 {
        return Err(Error::InvalidSumcheck(
            "wrong Quasar inner terminal vector".to_string(),
        ));
    }

    let mut input_evals_at_inner_point = Vec::with_capacity(k);
    let mut combined_input_eval = F::ZERO;
    for b in 0..k {
        let value = eval_poly(
            &endpoint_data.polys[layout.block_poly_index(b, 0)],
            &inner_point,
        );
        transcript.write_field_element(&value)?;
        combined_input_eval += instance_weights_at_outer_point[b] * value;
        input_evals_at_inner_point.push(value);
    }

    let expected_h = hadamard_tensor_mle_eval(&outer_point, &inner_point);
    debug_assert_eq!(inner_terminal_evals[0], expected_h);
    debug_assert_eq!(inner_terminal_evals[1], combined_input_eval);
    let inner_terminal_eval = inner_terminal_evals[0] * inner_terminal_evals[1];

    Ok(QABaseTwoLayerGkrProof {
        num_instances: k,
        block_batch_challenge,
        instance_batch_challenge,
        gammas,
        output_evals_at_gammas,
        claimed_output_batch,
        outer_point,
        outer_terminal_eval,
        e_evals_at_outer_point,
        instance_weights_at_outer_point,
        inner_point,
        input_evals_at_inner_point,
        inner_terminal_eval,
    })
}

fn verify_qabase_two_layer_gkr_batch<F, H>(
    vp: &QABaseVerifierParams<F, H>,
    num_instances: usize,
    transcript: &mut impl TranscriptRead<CommitmentChunk<H>, F>,
) -> Result<(bool, QABaseTwoLayerGkrProof<F>), Error>
where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
{
    type Sc<F> = ClassicSumCheck<EvaluationsProver<F>>;

    let k = num_instances;
    let rho = vp.inverse_rate;
    let num_parity = rho - 1;
    let num_vars = vp.num_vars;

    let block_batch_challenge = transcript.squeeze_challenges(1)[0];
    let instance_batch_challenge = if k > 1 {
        transcript.squeeze_challenges(1)[0]
    } else {
        F::ONE
    };
    let block_powers = powers(block_batch_challenge, num_parity);
    let instance_powers = powers(instance_batch_challenge, k);

    let mut gammas = Vec::with_capacity(k);
    for _ in 0..k {
        let mut per_instance = Vec::with_capacity(num_parity);
        for _ in 0..num_parity {
            per_instance.push(transcript.squeeze_challenges(num_vars));
        }
        gammas.push(per_instance);
    }

    let mut output_evals_at_gammas = Vec::with_capacity(k);
    let mut claimed_output_batch = F::ZERO;
    for b in 0..k {
        let mut per_instance = Vec::with_capacity(num_parity);
        for i in 0..num_parity {
            let value = transcript.read_field_element()?;
            claimed_output_batch += instance_powers[b] * block_powers[i] * value;
            per_instance.push(value);
        }
        output_evals_at_gammas.push(per_instance);
    }

    let (outer_terminal_eval, outer_point) =
        Sc::<F>::verify(&(), num_vars, 3usize, claimed_output_batch, transcript)?;

    let mut e_evals_at_outer_point = Vec::with_capacity(num_parity);
    for _ in 0..num_parity {
        e_evals_at_outer_point.push(transcript.read_field_element()?);
    }

    let instance_weights_at_outer_point = qabase_gkr_instance_weights(
        &gammas,
        &block_powers,
        &instance_powers,
        &outer_point,
        &e_evals_at_outer_point,
    );

    let (inner_terminal_eval, inner_point) =
        Sc::<F>::verify(&(), num_vars, 2usize, outer_terminal_eval, transcript)?;

    let mut input_evals_at_inner_point = Vec::with_capacity(k);
    let mut combined_input_eval = F::ZERO;
    for b in 0..k {
        let value = transcript.read_field_element()?;
        combined_input_eval += instance_weights_at_outer_point[b] * value;
        input_evals_at_inner_point.push(value);
    }

    let expected_inner_terminal =
        hadamard_tensor_mle_eval(&outer_point, &inner_point) * combined_input_eval;
    let ok = inner_terminal_eval == expected_inner_terminal;

    Ok((
        ok,
        QABaseTwoLayerGkrProof {
            num_instances: k,
            block_batch_challenge,
            instance_batch_challenge,
            gammas,
            output_evals_at_gammas,
            claimed_output_batch,
            outer_point,
            outer_terminal_eval,
            e_evals_at_outer_point,
            instance_weights_at_outer_point,
            inner_point,
            input_evals_at_inner_point,
            inner_terminal_eval,
        },
    ))
}

fn collect_two_layer_gkr_claims_prover<F, H>(
    acc: &mut QABaseProverOpeningAccumulator<'_, F, H>,
    layout: &QABaseTwoLayerGkrLayout,
    proof: &QABaseTwoLayerGkrProof<F>,
) where
    F: PrimeField,
    H: Hash,
{
    for b in 0..layout.num_instances {
        for i in 0..layout.rho - 1 {
            push_basefold_opening_claim(
                &mut acc.points,
                &mut acc.evals,
                layout.block_poly_index(b, i + 1),
                proof.gammas[b][i].clone(),
                proof.output_evals_at_gammas[b][i],
            );
        }
    }
    for b in 0..layout.num_instances {
        push_basefold_opening_claim(
            &mut acc.points,
            &mut acc.evals,
            layout.block_poly_index(b, 0),
            proof.inner_point.clone(),
            proof.input_evals_at_inner_point[b],
        );
    }
    for i in 0..layout.rho - 1 {
        push_basefold_opening_claim(
            &mut acc.points,
            &mut acc.evals,
            layout.e_poly_index(i),
            proof.outer_point.clone(),
            proof.e_evals_at_outer_point[i],
        );
    }
}

fn collect_two_layer_gkr_claims_verifier<F, H>(
    acc: &mut QABaseVerifierOpeningAccumulator<'_, F, H>,
    layout: &QABaseTwoLayerGkrLayout,
    proof: &QABaseTwoLayerGkrProof<F>,
) where
    F: PrimeField,
    H: Hash,
{
    for b in 0..layout.num_instances {
        for i in 0..layout.rho - 1 {
            push_basefold_opening_claim(
                &mut acc.points,
                &mut acc.evals,
                layout.block_poly_index(b, i + 1),
                proof.gammas[b][i].clone(),
                proof.output_evals_at_gammas[b][i],
            );
        }
    }
    for b in 0..layout.num_instances {
        push_basefold_opening_claim(
            &mut acc.points,
            &mut acc.evals,
            layout.block_poly_index(b, 0),
            proof.inner_point.clone(),
            proof.input_evals_at_inner_point[b],
        );
    }
    for i in 0..layout.rho - 1 {
        push_basefold_opening_claim(
            &mut acc.points,
            &mut acc.evals,
            layout.e_poly_index(i),
            proof.outer_point.clone(),
            proof.e_evals_at_outer_point[i],
        );
    }
}

// =============================================================================
// One merged, block-aware sampled-column consistency sumcheck
// =============================================================================

fn qabase_weighted_sum<F: PrimeField>(values: &[F], tau: F) -> F {
    let mut acc = F::ZERO;
    let mut power = F::ONE;
    for value in values {
        acc += power * *value;
        power *= tau;
    }
    acc
}

fn selector_block_evals<F: PrimeField>(
    rho: usize,
    block_len: usize,
    query_indices: &[usize],
    tau: F,
) -> Vec<Vec<F>> {
    let mut blocks = (0..rho)
        .map(|_| vec![F::ZERO; block_len])
        .collect::<Vec<_>>();
    let mut power = F::ONE;
    for index in query_indices {
        let block = *index / block_len;
        let local = *index % block_len;
        blocks[block][local] += power;
        power *= tau;
    }
    blocks
}

fn selector_block_eval_at_point<F: PrimeField>(
    block_index: usize,
    block_len: usize,
    query_indices: &[usize],
    tau: F,
    point: &[F],
) -> F {
    let num_vars = log2_power_of_two(block_len);
    let mut acc = F::ZERO;
    let mut power = F::ONE;
    for index in query_indices {
        let block = *index / block_len;
        let local = *index % block_len;
        if block == block_index {
            acc += power * equality_mle_eval_at_index(local, num_vars, point);
        }
        power *= tau;
    }
    acc
}

#[derive(Clone, Debug)]
pub struct QABaseMergedColumnConsistencyOutput<F: PrimeField> {
    pub query_indices: Vec<usize>,
    pub tau: F,
    pub instance_batch_challenge: F,
    pub sc_point: Vec<F>,
    pub proximity_block_evals: Vec<F>,
    pub evaluation_block_evals: Vec<F>,
}

fn prove_merged_column_consistency<F, H>(
    acc: &mut QABaseProverOpeningAccumulator<'_, F, H>,
    pp: &QABaseProverParams<F, H>,
    comm: &QABaseCommitment<F, H, impl QACodewordColumns<F>>,
    endpoint_data: &QABaseEndpointProverData<F, H>,
    layout: &QABaseTwoLayerGkrLayout,
    proximity_weights: &[F],
    evaluation_weights: &[F],
    transcript: &mut impl TranscriptWrite<CommitmentChunk<H>, F>,
) -> Result<QABaseMergedColumnConsistencyOutput<F>, Error>
where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
{
    type Sc<F> = ClassicSumCheck<EvaluationsProver<F>>;
    if layout.num_instances != 2 {
        return Err(Error::InvalidPcsOpen(
            "full Quasar opening expects two instances".to_string(),
        ));
    }

    let rho = layout.rho;
    let block_len = 1usize << pp.num_vars;
    let num_cols = rho * block_len;
    let query_challenges = transcript.squeeze_challenges(pp.num_queries);
    let query_indices = query_challenges
        .iter()
        .map(|challenge| field_challenge_to_index(challenge, num_cols))
        .collect::<Vec<_>>();

    let mut proximity_values = Vec::with_capacity(query_indices.len());
    let mut evaluation_values = Vec::with_capacity(query_indices.len());

    for &full_index in &query_indices {
        write_merkle_path::<H, F>(&comm.codeword_tree, full_index, transcript)?;
        let opened_column = comm
            .codeword
            .read_column(full_index)
            .map_err(Error::InvalidPcsOpen)?;
        transcript.write_field_elements(&opened_column)?;
        let (proximity, evaluation) =
            fold_opened_column_pair(&opened_column, proximity_weights, evaluation_weights);
        proximity_values.push(proximity);
        evaluation_values.push(evaluation);
    }

    let tau = transcript.squeeze_challenges(1)[0];
    let instance_batch_challenge = transcript.squeeze_challenges(1)[0];
    let claimed_sum = qabase_weighted_sum(&proximity_values, tau)
        + instance_batch_challenge * qabase_weighted_sum(&evaluation_values, tau);

    let selector_polys = selector_block_evals(rho, block_len, &query_indices, tau)
        .into_iter()
        .map(MultilinearPolynomial::new)
        .collect::<Vec<_>>();

    // Borrow the endpoint polynomials directly.  The old implementation
    // materialized a second cN-sized vector for u + zeta*q.
    let mut poly_refs = Vec::with_capacity(3 * rho);
    for block in 0..rho {
        poly_refs.push(&selector_polys[block]);
        poly_refs.push(&endpoint_data.polys[layout.block_poly_index(0, block)]);
        poly_refs.push(&endpoint_data.polys[layout.block_poly_index(1, block)]);
    }

    let make_term = |block: usize| -> Expression<F> {
        let h: Expression<F> = Expression::<F>::Polynomial(Query::new(3 * block, Rotation::cur()));
        let u: Expression<F> =
            Expression::<F>::Polynomial(Query::new(3 * block + 1, Rotation::cur()));
        let q: Expression<F> =
            Expression::<F>::Polynomial(Query::new(3 * block + 2, Rotation::cur()));
        h * (u + q * instance_batch_challenge)
    };
    let mut expression = make_term(0);
    for block in 1..rho {
        expression = expression + make_term(block);
    }

    let challenges: Vec<F> = Vec::new();
    let ys: Vec<Vec<F>> = Vec::new();
    let virtual_poly = VirtualPolynomial::new(&expression, poly_refs, &challenges, &ys);
    let (sc_point, terminal_evals) =
        Sc::<F>::prove(&(), pp.num_vars, virtual_poly, claimed_sum, transcript)?;
    if terminal_evals.len() != 3 * rho {
        return Err(Error::InvalidSumcheck(
            "wrong merged selector terminal vector".to_string(),
        ));
    }

    let mut proximity_block_evals = Vec::with_capacity(rho);
    let mut evaluation_block_evals = Vec::with_capacity(rho);
    for block in 0..rho {
        // These values are already the sumcheck terminal evaluations; do not
        // evaluate the same endpoint polynomials a second time.
        let u = terminal_evals[3 * block + 1];
        let q = terminal_evals[3 * block + 2];
        transcript.write_field_element(&u)?;
        transcript.write_field_element(&q)?;
        proximity_block_evals.push(u);
        evaluation_block_evals.push(q);

        push_basefold_opening_claim(
            &mut acc.points,
            &mut acc.evals,
            layout.block_poly_index(0, block),
            sc_point.clone(),
            u,
        );
        push_basefold_opening_claim(
            &mut acc.points,
            &mut acc.evals,
            layout.block_poly_index(1, block),
            sc_point.clone(),
            q,
        );
    }

    Ok(QABaseMergedColumnConsistencyOutput {
        query_indices,
        tau,
        instance_batch_challenge,
        sc_point,
        proximity_block_evals,
        evaluation_block_evals,
    })
}

fn verify_merged_column_consistency<F, H>(
    acc: &mut QABaseVerifierOpeningAccumulator<'_, F, H>,
    vp: &QABaseVerifierParams<F, H>,
    comm: &QABaseCommitment<F, H, impl Send + Sync>,
    layout: &QABaseTwoLayerGkrLayout,
    proximity_weights: &[F],
    evaluation_weights: &[F],
    transcript: &mut impl TranscriptRead<CommitmentChunk<H>, F>,
) -> Result<(bool, Option<QABaseMergedColumnConsistencyOutput<F>>), Error>
where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
{
    type Sc<F> = ClassicSumCheck<EvaluationsProver<F>>;

    let rho = layout.rho;
    let block_len = 1usize << vp.num_vars;
    let num_cols = rho * block_len;
    let root: &Output<H> = comm.as_ref();

    let query_challenges = transcript.squeeze_challenges(vp.num_queries);
    let query_indices = query_challenges
        .iter()
        .map(|challenge| field_challenge_to_index(challenge, num_cols))
        .collect::<Vec<_>>();

    let mut proximity_values = Vec::with_capacity(query_indices.len());
    let mut evaluation_values = Vec::with_capacity(query_indices.len());
    for &full_index in &query_indices {
        let path = read_merkle_path::<H, F>(num_cols, transcript)?;
        let opened_column = transcript.read_field_elements(vp.num_rows)?;
        if !verify_merkle_path_with_leaf::<H, F>(root, &opened_column, &path, full_index) {
            return Ok((false, None));
        }
        let (proximity, evaluation) =
            fold_opened_column_pair(&opened_column, proximity_weights, evaluation_weights);
        proximity_values.push(proximity);
        evaluation_values.push(evaluation);
    }

    let tau = transcript.squeeze_challenges(1)[0];
    let instance_batch_challenge = transcript.squeeze_challenges(1)[0];
    let claimed_sum = qabase_weighted_sum(&proximity_values, tau)
        + instance_batch_challenge * qabase_weighted_sum(&evaluation_values, tau);

    let (terminal_eval, sc_point) =
        Sc::<F>::verify(&(), vp.num_vars, 2usize, claimed_sum, transcript)?;

    let mut proximity_block_evals = Vec::with_capacity(rho);
    let mut evaluation_block_evals = Vec::with_capacity(rho);
    let mut expected_terminal = F::ZERO;
    for block in 0..rho {
        let u = transcript.read_field_element()?;
        let q = transcript.read_field_element()?;
        proximity_block_evals.push(u);
        evaluation_block_evals.push(q);

        let selector_eval =
            selector_block_eval_at_point(block, block_len, &query_indices, tau, &sc_point);
        expected_terminal += selector_eval * (u + instance_batch_challenge * q);

        push_basefold_opening_claim(
            &mut acc.points,
            &mut acc.evals,
            layout.block_poly_index(0, block),
            sc_point.clone(),
            u,
        );
        push_basefold_opening_claim(
            &mut acc.points,
            &mut acc.evals,
            layout.block_poly_index(1, block),
            sc_point.clone(),
            q,
        );
    }

    let ok = terminal_eval == expected_terminal;
    Ok((
        ok,
        Some(QABaseMergedColumnConsistencyOutput {
            query_indices,
            tau,
            instance_batch_challenge,
            sc_point,
            proximity_block_evals,
            evaluation_block_evals,
        }),
    ))
}

// =============================================================================
// Full opening
// =============================================================================

#[derive(Clone, Debug)]
pub struct QABaseFullOpenTwoLayerGkrProverOutput {
    pub endpoint_commitment_count: usize,
    pub opening_claim_count: usize,
    pub unique_opening_point_count: usize,
    pub query_indices: Vec<usize>,
    pub ok_eval_value: bool,
}

#[derive(Clone, Debug)]
pub struct QABaseFullOpenTwoLayerGkrVerifierOutput {
    pub endpoint_commitment_count: usize,
    pub opening_claim_count: usize,
    pub unique_opening_point_count: usize,
    pub query_indices: Vec<usize>,
}

impl QABaseFullOpenTwoLayerGkrVerifierOutput {
    fn rejected() -> Self {
        Self {
            endpoint_commitment_count: 0,
            opening_claim_count: 0,
            unique_opening_point_count: 0,
            query_indices: Vec::new(),
        }
    }
}

pub fn prove_qabase_open_full_two_layer_gkr<F, H>(
    pp: &QABaseProverParams<F, H>,
    word: &(impl QACodewordRows<F> + ?Sized),
    comm: &QABaseCommitment<F, H, impl QACodewordColumns<F>>,
    z_left: Vec<F>,
    z_right: Vec<F>,
    claimed_value: F,
    transcript: &mut impl TranscriptWrite<CommitmentChunk<H>, F>,
) -> Result<QABaseFullOpenTwoLayerGkrProverOutput, Error>
where
    F: PrimeField + Serialize + DeserializeOwned + Send + Sync,
    H: Hash,
{
    type Pcs<F, H> = Basefold<F, H, QABaseFoldConfig>;
    let now = Instant::now();

    if word.num_rows() != pp.num_rows
        || word.num_cols() != (1usize << pp.num_vars)
        || (0..word.num_rows()).any(|i| word.row(i).len() != word.num_cols())
        || z_left.len() != log2_power_of_two(pp.num_rows)
        || z_right.len() != pp.num_vars
    {
        return Err(Error::InvalidPcsParam(
            "invalid Quasar full-opening dimensions".to_string(),
        ));
    }

    let row_challenges = transcript.squeeze_challenges(pp.num_rows);
    let row_weights = qabase_row_weights_from_z_left(pp.num_rows, &z_left);

    let (proximity_input, evaluation_input) =
        linear_combine_rows_pair_parallel(word, &row_challenges, &row_weights);

    let mut endpoint_data = commit_endpoint_instances_with_basefold(
        pp,
        vec![proximity_input, evaluation_input],
        transcript,
    )?;
    let endpoint_commitment_count = endpoint_data.commitments.block_commitments.len();

    // Run GKR before constructing the borrowed BaseFold accumulator, because
    // GKR moves `middle_evals` out of endpoint_data to release that memory.
    let gkr_layout = QABaseTwoLayerGkrLayout {
        rho: pp.inverse_rate,
        num_instances: endpoint_data.commitments.num_instances,
        block_starts: (0..endpoint_data.commitments.num_instances)
            .map(|b| b * pp.inverse_rate)
            .collect(),
        e_start: endpoint_data.commitments.num_instances * pp.inverse_rate,
    };
    let gkr = prove_qabase_two_layer_gkr_batch(pp, &mut endpoint_data, &gkr_layout, transcript)?;

    let (mut opening_acc, layout) = build_two_layer_gkr_prover_accumulator(&endpoint_data, pp);
    collect_two_layer_gkr_claims_prover(&mut opening_acc, &layout, &gkr);

    let column_consistency = prove_merged_column_consistency(
        &mut opening_acc,
        pp,
        comm,
        &endpoint_data,
        &layout,
        &row_challenges,
        &row_weights,
        transcript,
    )?;

    let actual_value = eval_poly(
        &endpoint_data.polys[layout.block_poly_index(1, 0)],
        &z_right,
    );
    if actual_value != claimed_value {
        return Err(Error::InvalidPcsOpen(
            "claimed Quasar evaluation does not match the witness".to_string(),
        ));
    }
    push_basefold_opening_claim(
        &mut opening_acc.points,
        &mut opening_acc.evals,
        layout.block_poly_index(1, 0),
        z_right,
        claimed_value,
    );

    let opening_claim_count = opening_acc.evals.len();
    let unique_opening_point_count = opening_acc.points.len();

    // `middle_evals` have already been moved and released by GKR.  The full
    // endpoint witness is not returned, so only the BaseFold prover state stays
    // live for the final opening.
    Pcs::<F, H>::batch_open(
        &pp.basefold_prover_param,
        opening_acc.polys.iter().copied(),
        opening_acc.comms.iter().copied(),
        &opening_acc.points,
        &opening_acc.evals,
        transcript,
    )?;

    println!(
        "Quasar two-layer GKR prove {:?}, endpoints {}, claims {}, unique points {}",
        now.elapsed(),
        endpoint_commitment_count,
        opening_claim_count,
        unique_opening_point_count,
    );

    Ok(QABaseFullOpenTwoLayerGkrProverOutput {
        endpoint_commitment_count,
        opening_claim_count,
        unique_opening_point_count,
        query_indices: column_consistency.query_indices,
        ok_eval_value: true,
    })
}

pub fn verify_qabase_open_full_two_layer_gkr<F, H>(
    vp: &QABaseVerifierParams<F, H>,
    comm: &QABaseCommitment<F, H, impl Send + Sync>,
    z_left: Vec<F>,
    z_right: Vec<F>,
    claimed_value: F,
    transcript: &mut impl TranscriptRead<CommitmentChunk<H>, F>,
) -> Result<(bool, QABaseFullOpenTwoLayerGkrVerifierOutput), Error>
where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
{
    type Pcs<F, H> = Basefold<F, H, QABaseFoldConfig>;
    let now = Instant::now();

    if z_left.len() != log2_power_of_two(vp.num_rows) || z_right.len() != vp.num_vars {
        return Err(Error::InvalidPcsParam(
            "invalid Quasar evaluation point dimensions".to_string(),
        ));
    }

    let root_from_transcript = transcript.read_commitment()?;
    let committed_root: &Output<H> = comm.as_ref();
    if &root_from_transcript != committed_root {
        return Ok((false, QABaseFullOpenTwoLayerGkrVerifierOutput::rejected()));
    }

    let row_challenges = transcript.squeeze_challenges(vp.num_rows);
    let row_weights = qabase_row_weights_from_z_left(vp.num_rows, &z_left);

    let endpoint_commitments = read_endpoint_block_commitments(vp, 2, transcript)?;
    let endpoint_commitment_count = endpoint_commitments.block_commitments.len();
    let (mut opening_acc, layout) =
        build_two_layer_gkr_verifier_accumulator(&endpoint_commitments, vp);

    let (ok_gkr, gkr) = verify_qabase_two_layer_gkr_batch(vp, 2, transcript)?;
    if !ok_gkr {
        return Ok((false, QABaseFullOpenTwoLayerGkrVerifierOutput::rejected()));
    }
    collect_two_layer_gkr_claims_verifier(&mut opening_acc, &layout, &gkr);

    let (ok_columns, column_consistency) = verify_merged_column_consistency(
        &mut opening_acc,
        vp,
        comm,
        &layout,
        &row_challenges,
        &row_weights,
        transcript,
    )?;
    let Some(column_consistency) = column_consistency else {
        return Ok((false, QABaseFullOpenTwoLayerGkrVerifierOutput::rejected()));
    };
    if !ok_columns {
        return Ok((false, QABaseFullOpenTwoLayerGkrVerifierOutput::rejected()));
    }

    push_basefold_opening_claim(
        &mut opening_acc.points,
        &mut opening_acc.evals,
        layout.block_poly_index(1, 0),
        z_right,
        claimed_value,
    );

    let opening_claim_count = opening_acc.evals.len();
    let unique_opening_point_count = opening_acc.points.len();
    Pcs::<F, H>::batch_verify(
        &vp.basefold_verifier_param,
        opening_acc.comms.iter().copied(),
        &opening_acc.points,
        &opening_acc.evals,
        transcript,
    )?;

    println!(
        "Quasar two-layer GKR verify {:?}, endpoints {}, claims {}, unique points {}",
        now.elapsed(),
        endpoint_commitment_count,
        opening_claim_count,
        unique_opening_point_count,
    );

    Ok((
        true,
        QABaseFullOpenTwoLayerGkrVerifierOutput {
            endpoint_commitment_count,
            opening_claim_count,
            unique_opening_point_count,
            query_indices: column_consistency.query_indices,
        },
    ))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod test {
    use super::*;
    use crate::util::{
        hash::Blake2s,
        transcript::{Blake2sTranscript, InMemoryTranscript, TranscriptWrite},
    };
    use halo2_curves::bn256::Fr;
    use rand_chacha::{rand_core::SeedableRng, ChaCha8Rng};
    use std::io::Cursor;

    type TestTranscript = Blake2sTranscript<Cursor<Vec<u8>>>;

    fn random_matrix(rows: usize, cols: usize, rng: &mut ChaCha8Rng) -> Vec<Vec<Fr>> {
        (0..rows)
            .map(|_| (0..cols).map(|_| Fr::random(&mut *rng)).collect())
            .collect()
    }

    #[test]
    fn test_commit_rate_2_and_4() {
        for inverse_rate in [2usize, 4usize] {
            let poly_size = 1usize << 6;
            let num_rows = 4;
            let mut rng = ChaCha8Rng::from_seed([9u8; 32]);
            let param = setup::<Fr, Blake2s>(
                poly_size,
                1,
                &mut rng,
                Some(num_rows),
                Some(inverse_rate),
                Some(8),
            );
            let (pp, _) = trim::<Fr, Blake2s>(&param, poly_size, 1);
            let word = random_matrix(num_rows, poly_size, &mut rng);
            let mut transcript = TestTranscript::new(());
            let comm = commit_and_write(&pp, &word, &mut transcript);
            assert_eq!(comm.codeword.len(), num_rows);
            assert!(comm
                .codeword
                .iter()
                .all(|row| row.len() == inverse_rate * poly_size));
        }
    }

    #[test]
    fn test_row_major_point_split_matches_direct_mle() {
        let num_rows = 4usize;
        let row_size = 8usize;
        let num_vars = 3usize;
        let mut rng = ChaCha8Rng::from_seed([17u8; 32]);
        let word = random_matrix(num_rows, row_size, &mut rng);
        let full_point = (0..5).map(|_| Fr::random(&mut rng)).collect::<Vec<_>>();

        let flat = word.iter().flatten().copied().collect::<Vec<_>>();
        let direct = eval_mle_from_evals(&flat, &full_point);
        let (z_left, z_right) = qabase_split_evaluation_point(&full_point, num_rows, num_vars);
        let weights = qabase_row_weights_from_z_left(num_rows, &z_left);
        let zero = vec![Fr::ZERO; num_rows];
        let (eval_msg, _) = linear_combine_rows_pair_parallel(&word, &weights, &zero);
        let decomposed = eval_mle_from_evals(&eval_msg, &z_right);
        assert_eq!(direct, decomposed);
    }

    #[test]
    fn test_full_two_layer_gkr_rate_2_and_4() {
        for inverse_rate in [2usize, 4usize] {
            let num_vars = 6usize;
            let poly_size = 1usize << num_vars;
            let num_rows = 4usize;
            let mut rng = ChaCha8Rng::from_seed([71u8; 32]);
            let param = setup::<Fr, Blake2s>(
                poly_size,
                1,
                &mut rng,
                Some(num_rows),
                Some(inverse_rate),
                Some(8),
            );
            let (pp, vp) = trim::<Fr, Blake2s>(&param, poly_size, 1);
            let word = random_matrix(num_rows, poly_size, &mut rng);

            let full_point = (0..(num_vars + 2))
                .map(|_| Fr::random(&mut rng))
                .collect::<Vec<_>>();
            let (z_left, z_right) = qabase_split_evaluation_point(&full_point, num_rows, num_vars);
            let flat = word.iter().flatten().copied().collect::<Vec<_>>();
            let claimed_value = eval_mle_from_evals(&flat, &full_point);

            let mut prover_transcript = TestTranscript::new(());
            let comm = commit_and_write(&pp, &word, &mut prover_transcript);
            let prover_output = prove_qabase_open_full_two_layer_gkr(
                &pp,
                &word,
                &comm,
                z_left.clone(),
                z_right.clone(),
                claimed_value,
                &mut prover_transcript,
            )
            .unwrap();
            assert_eq!(prover_output.endpoint_commitment_count, 2 * inverse_rate,);
            assert!(prover_output.unique_opening_point_count < prover_output.opening_claim_count);

            let proof = prover_transcript.into_proof();
            let mut verifier_transcript = TestTranscript::from_proof((), proof.as_slice());
            let (ok, verifier_output) = verify_qabase_open_full_two_layer_gkr(
                &vp,
                &comm,
                z_left,
                z_right,
                claimed_value,
                &mut verifier_transcript,
            )
            .unwrap();
            assert!(ok);
            assert_eq!(prover_output.query_indices, verifier_output.query_indices,);
        }
    }

    #[test]
    fn test_wrong_claimed_value_is_rejected_by_prover() {
        let num_vars = 5usize;
        let poly_size = 1usize << num_vars;
        let num_rows = 4usize;
        let mut rng = ChaCha8Rng::from_seed([81u8; 32]);
        let param = setup::<Fr, Blake2s>(poly_size, 1, &mut rng, Some(num_rows), Some(2), Some(4));
        let (pp, _) = trim::<Fr, Blake2s>(&param, poly_size, 1);
        let word = random_matrix(num_rows, poly_size, &mut rng);
        let point = (0..(num_vars + 2))
            .map(|_| Fr::random(&mut rng))
            .collect::<Vec<_>>();
        let (z_left, z_right) = qabase_split_evaluation_point(&point, num_rows, num_vars);

        let mut transcript = TestTranscript::new(());
        let comm = commit_and_write(&pp, &word, &mut transcript);
        let result = prove_qabase_open_full_two_layer_gkr(
            &pp,
            &word,
            &comm,
            z_left,
            z_right,
            Fr::random(&mut rng),
            &mut transcript,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_root_mismatch_rejects_without_panic() {
        let num_vars = 5usize;
        let poly_size = 1usize << num_vars;
        let num_rows = 4usize;
        let mut rng = ChaCha8Rng::from_seed([91u8; 32]);
        let param = setup::<Fr, Blake2s>(poly_size, 1, &mut rng, Some(num_rows), Some(2), Some(4));
        let (pp, vp) = trim::<Fr, Blake2s>(&param, poly_size, 1);
        let word = random_matrix(num_rows, poly_size, &mut rng);
        let point = (0..(num_vars + 2))
            .map(|_| Fr::random(&mut rng))
            .collect::<Vec<_>>();
        let (z_left, z_right) = qabase_split_evaluation_point(&point, num_rows, num_vars);
        let flat = word.iter().flatten().copied().collect::<Vec<_>>();
        let value = eval_mle_from_evals(&flat, &point);

        let mut prover_transcript = TestTranscript::new(());
        let comm = commit_and_write(&pp, &word, &mut prover_transcript);
        prove_qabase_open_full_two_layer_gkr(
            &pp,
            &word,
            &comm,
            z_left.clone(),
            z_right.clone(),
            value,
            &mut prover_transcript,
        )
        .unwrap();
        let mut proof = prover_transcript.into_proof();
        proof[0] ^= 1;

        let mut verifier_transcript = TestTranscript::from_proof((), proof.as_slice());
        let (ok, _) = verify_qabase_open_full_two_layer_gkr(
            &vp,
            &comm,
            z_left,
            z_right,
            value,
            &mut verifier_transcript,
        )
        .unwrap();
        assert!(!ok);
    }
}
