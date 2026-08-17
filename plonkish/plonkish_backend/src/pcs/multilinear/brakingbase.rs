//! BrakingBase: a Brakedown commitment compiled with sumcheck, SPARK, and
//! BaseFold openings.
//!
//! This module follows Protocols 1--5 and Appendices B--C of
//! Nair--Sharma--Thankey, "BrakingBase" (ASIACRYPT 2025).  The committed
//! polynomial is arranged in `O(log n)` rows, each row is encoded with the
//! systematic rate-1/2 Brakedown code, and encoded columns are Merkle
//! committed.  The opening protocol certifies column linear combinations and
//! code membership with sumcheck.  Code membership uses the sparse
//! parity-check matrix and a SPARK offline-memory-check argument.  All terminal
//! polynomial claims are reduced to one batched BaseFold opening.

use crate::{
    pcs::{
        multilinear::{
            validate_input, Basefold, BasefoldCommitment, BasefoldExtParams, BasefoldParams,
            BasefoldProverParams, BasefoldVerifierParams,
        },
        Evaluation, Point, PolynomialCommitmentScheme,
    },
    piop::sum_check::{
        classic::{ClassicSumCheck, EvaluationsProver},
        SumCheck as _, VirtualPolynomial,
    },
    poly::{multilinear::MultilinearPolynomial, Polynomial},
    util::{
        arithmetic::{inner_product, PrimeField},
        code::{Brakedown, BrakedownSpec, LinearCodes},
        expression::{Expression, Query, Rotation},
        hash::{Hash, Output},
        parallel::{num_threads, parallelize, parallelize_iter},
        transcript::{FieldTranscript, TranscriptRead, TranscriptWrite},
        Deserialize, DeserializeOwned, Itertools, Serialize,
    },
    Error,
};
use rand::RngCore;
use rayon::prelude::*;
use std::{marker::PhantomData, mem::size_of, slice};

type SumCheck<F> = ClassicSumCheck<EvaluationsProver<F>>;
type Bf<F, H, V> = Basefold<F, H, V>;

const BASE_CODE_THRESHOLD: usize = 20;
const ROW_ADDR: usize = 0;
const COL_ADDR: usize = 1;
const MATRIX_VAL: usize = 2;
const ROW_READ_TS: usize = 3;
const ROW_AUDIT_TS: usize = 4;
const COL_READ_TS: usize = 5;
const COL_AUDIT_TS: usize = 6;
const NUM_SPARK_STATIC_POLYS: usize = 7;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: DeserializeOwned"))]
struct SparseEntry<F> {
    row: usize,
    col: usize,
    value: F,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: DeserializeOwned"))]
struct SparkProverData<F: PrimeField> {
    entries: Vec<SparseEntry<F>>,
    row_addr: Vec<usize>,
    col_addr: Vec<usize>,
    static_polys: Vec<MultilinearPolynomial<F>>,
    num_ops: usize,
    num_mem_cells: usize,
    aux_len: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: DeserializeOwned"))]
pub struct BrakingBaseProverParam<F: PrimeField, H: Hash> {
    num_vars: usize,
    num_rows: usize,
    row_len: usize,
    codeword_len: usize,
    num_queries: usize,
    brakedown: Brakedown<F>,
    spark: SparkProverData<F>,
    basefold: BasefoldProverParams<F>,
    spark_comms: Vec<BasefoldCommitment<F, H>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: DeserializeOwned"))]
pub struct BrakingBaseVerifierParam<F: PrimeField, H: Hash> {
    num_vars: usize,
    num_rows: usize,
    row_len: usize,
    codeword_len: usize,
    num_queries: usize,
    num_ops: usize,
    num_mem_cells: usize,
    aux_len: usize,
    basefold: BasefoldVerifierParams<F>,
    spark_comms: Vec<BasefoldCommitment<F, H>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: DeserializeOwned"))]
pub struct BrakingBaseParams<F: PrimeField, H: Hash> {
    prover: BrakingBaseProverParam<F, H>,
    verifier: BrakingBaseVerifierParam<F, H>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: DeserializeOwned"))]
pub struct BrakingBaseCommitment<F: PrimeField, H: Hash> {
    rows: Vec<F>,
    tree: Vec<Vec<Output<H>>>,
    root: Output<H>,
}

impl<F: PrimeField, H: Hash> Default for BrakingBaseCommitment<F, H> {
    fn default() -> Self {
        Self {
            rows: Vec::new(),
            tree: Vec::new(),
            root: Output::<H>::default(),
        }
    }
}

impl<F: PrimeField, H: Hash> BrakingBaseCommitment<F, H> {
    fn from_root(root: Output<H>) -> Self {
        Self {
            rows: Vec::new(),
            tree: Vec::new(),
            root,
        }
    }

    pub fn root(&self) -> &Output<H> {
        &self.root
    }
}

impl<F: PrimeField, H: Hash> BrakingBaseProverParam<F, H> {
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    pub fn row_len(&self) -> usize {
        self.row_len
    }

    pub fn codeword_len(&self) -> usize {
        self.codeword_len
    }

    pub fn num_queries(&self) -> usize {
        self.num_queries
    }

    pub fn spark_num_ops(&self) -> usize {
        self.spark.num_ops
    }

    pub fn spark_aux_len(&self) -> usize {
        self.spark.aux_len
    }
}

impl<F: PrimeField, H: Hash> BrakingBaseVerifierParam<F, H> {
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    pub fn row_len(&self) -> usize {
        self.row_len
    }

    pub fn codeword_len(&self) -> usize {
        self.codeword_len
    }

    pub fn num_queries(&self) -> usize {
        self.num_queries
    }

    pub fn spark_num_ops(&self) -> usize {
        self.num_ops
    }

    pub fn spark_aux_len(&self) -> usize {
        self.aux_len
    }
}

impl<F: PrimeField, H: Hash> AsRef<[Output<H>]> for BrakingBaseCommitment<F, H> {
    fn as_ref(&self) -> &[Output<H>] {
        slice::from_ref(&self.root)
    }
}

#[derive(Debug)]
pub struct MultilinearBrakingBase<F: PrimeField, H: Hash, S: BrakedownSpec, V: BasefoldExtParams>(
    PhantomData<(F, H, S, V)>,
);

impl<F: PrimeField, H: Hash, S: BrakedownSpec, V: BasefoldExtParams> Clone
    for MultilinearBrakingBase<F, H, S, V>
{
    fn clone(&self) -> Self {
        Self(PhantomData)
    }
}

fn pad_poly<F: PrimeField>(mut values: Vec<F>, len: usize) -> MultilinearPolynomial<F> {
    assert!(len.is_power_of_two() && values.len() <= len);
    values.resize(len, F::ZERO);
    MultilinearPolynomial::new(values)
}

fn brakingbase_num_rows(num_vars: usize) -> usize {
    num_vars.next_power_of_two()
}

/// Build a sparse parity-check matrix for the exact encoder implemented by
/// `util::code::Brakedown`, including its constant-size Reed--Solomon base
/// code.  Entries represent a matrix with codeword coordinates as rows and
/// parity constraints as columns.
fn parity_check_entries<F: PrimeField>(code: &Brakedown<F>) -> Vec<SparseEntry<F>> {
    let a = code.a_matrices();
    let b = code.b_matrices();
    assert!(!a.is_empty() && a.len() == b.len());

    let mut entries = Vec::new();
    let mut constraint = 0usize;
    let mut input_offset = 0usize;

    // Stored recursive A-images.  The final A-image is fed directly into the
    // small Reed--Solomon base code and is not itself stored.
    for matrix in &a[..a.len() - 1] {
        let dim = matrix.dimension();
        let output_offset = input_offset + dim.n;
        for (input_row, cells) in matrix.rows().enumerate() {
            for &(output_col, coeff) in cells {
                entries.push(SparseEntry {
                    row: input_offset + input_row,
                    col: constraint + output_col,
                    value: coeff,
                });
            }
        }
        for output_col in 0..dim.m {
            entries.push(SparseEntry {
                row: output_offset + output_col,
                col: constraint + output_col,
                value: -F::ONE,
            });
        }
        constraint += dim.m;
        input_offset += dim.n;
    }

    // Dense, constant-size Reed--Solomon base relation.
    let a_last = a.last().unwrap();
    let b_last = b.last().unwrap();
    let a_dim = a_last.dimension();
    let b_dim = b_last.dimension();
    let base_offset = input_offset + a_dim.n;
    for output in 0..b_dim.n {
        let x = F::from((output + 1) as u64);
        let mut powers = vec![F::ONE; a_dim.m];
        for i in 1..powers.len() {
            powers[i] = powers[i - 1] * x;
        }
        for (input_row, cells) in a_last.rows().enumerate() {
            let coeff = cells
                .iter()
                .fold(F::ZERO, |acc, (j, value)| acc + *value * powers[*j]);
            if coeff != F::ZERO {
                entries.push(SparseEntry {
                    row: input_offset + input_row,
                    col: constraint + output,
                    value: coeff,
                });
            }
        }
        entries.push(SparseEntry {
            row: base_offset + output,
            col: constraint + output,
            value: -F::ONE,
        });
    }
    constraint += b_dim.n;

    // B-images are appended from the deepest recursion level back outwards.
    let mut output_offset = base_offset + b_dim.n;
    let mut encoded_input_offset = input_offset + a_dim.n + a_dim.m;
    for (a_matrix, b_matrix) in a.iter().rev().zip(b.iter().rev()) {
        let a_dim = a_matrix.dimension();
        let b_dim = b_matrix.dimension();
        encoded_input_offset -= a_dim.m;
        for (input_row, cells) in b_matrix.rows().enumerate() {
            for &(output_col, coeff) in cells {
                entries.push(SparseEntry {
                    row: encoded_input_offset + input_row,
                    col: constraint + output_col,
                    value: coeff,
                });
            }
        }
        for output_col in 0..b_dim.m {
            entries.push(SparseEntry {
                row: output_offset + output_col,
                col: constraint + output_col,
                value: -F::ONE,
            });
        }
        constraint += b_dim.m;
        output_offset += b_dim.m;
    }

    assert_eq!(constraint, code.codeword_len() - code.row_len());
    assert_eq!(output_offset, code.codeword_len());
    entries
}

fn spark_preprocess<F: PrimeField>(
    entries: Vec<SparseEntry<F>>,
    matrix_rows: usize,
    matrix_cols: usize,
    minimum_aux_len: usize,
) -> SparkProverData<F> {
    assert!(matrix_rows.is_power_of_two() && matrix_cols.is_power_of_two());
    let num_mem_cells = matrix_rows.max(matrix_cols);
    let num_ops = entries.len().max(2).next_power_of_two();
    let aux_len = num_ops
        .max(num_mem_cells)
        .max(minimum_aux_len)
        .next_power_of_two();

    let mut row_addr = vec![0usize; num_ops];
    let mut col_addr = vec![0usize; num_ops];
    let mut values = vec![F::ZERO; num_ops];
    for (i, entry) in entries.iter().enumerate() {
        row_addr[i] = entry.row;
        col_addr[i] = entry.col;
        values[i] = entry.value;
    }

    fn timestamps(addresses: &[usize], cells: usize) -> (Vec<usize>, Vec<usize>) {
        let mut audit = vec![0usize; cells];
        let mut read = vec![0usize; addresses.len()];
        for (i, &address) in addresses.iter().enumerate() {
            assert!(address < cells);
            read[i] = audit[address];
            audit[address] += 1;
        }
        (read, audit)
    }

    let (row_read, row_audit) = timestamps(&row_addr, num_mem_cells);
    let (col_read, col_audit) = timestamps(&col_addr, num_mem_cells);
    let to_field = |values: &[usize]| values.iter().map(|&v| F::from(v as u64)).collect_vec();

    let static_polys = vec![
        pad_poly(to_field(&row_addr), aux_len),
        pad_poly(to_field(&col_addr), aux_len),
        pad_poly(values, aux_len),
        pad_poly(to_field(&row_read), aux_len),
        pad_poly(to_field(&row_audit), aux_len),
        pad_poly(to_field(&col_read), aux_len),
        pad_poly(to_field(&col_audit), aux_len),
    ];

    SparkProverData {
        entries,
        row_addr,
        col_addr,
        static_polys,
        num_ops,
        num_mem_cells,
        aux_len,
    }
}

#[derive(Clone)]
struct OpeningClaim<F> {
    poly: usize,
    point: Vec<F>,
    value: F,
}

fn extend_point<F: PrimeField>(point: &[F], num_vars: usize) -> Vec<F> {
    assert!(point.len() <= num_vars);
    point
        .iter()
        .copied()
        .chain(std::iter::repeat(F::ZERO))
        .take(num_vars)
        .collect()
}

fn eq_index<F: PrimeField>(point: &[F], index: usize) -> F {
    point.iter().enumerate().fold(F::ONE, |acc, (i, x)| {
        acc * if ((index >> i) & 1) == 1 {
            *x
        } else {
            F::ONE - x
        }
    })
}

fn identity_eval<F: PrimeField>(point: &[F]) -> F {
    let mut power = F::ONE;
    point.iter().fold(F::ZERO, |acc, x| {
        let out = acc + power * x;
        power = power.double();
        out
    })
}

fn challenge_index<F: PrimeField>(transcript: &mut impl FieldTranscript<F>, cap: usize) -> usize {
    let challenge = transcript.squeeze_challenge();
    let mut bytes = [0u8; size_of::<u32>()];
    bytes.copy_from_slice(&challenge.to_repr().as_ref()[..size_of::<u32>()]);
    u32::from_le_bytes(bytes) as usize % cap
}

fn merkelize_columns<F: PrimeField, H: Hash>(
    rows: &[F],
    num_rows: usize,
    codeword_len: usize,
) -> Vec<Vec<Output<H>>> {
    let leaf_count = codeword_len.next_power_of_two();
    let mut leaves = vec![Output::<H>::default(); leaf_count];
    parallelize(&mut leaves[..codeword_len], |(leaves, start)| {
        for (leaf, column) in leaves.iter_mut().zip(start..) {
            let mut hasher = H::new();
            for row in 0..num_rows {
                hasher.update_field_element(&rows[row * codeword_len + column]);
            }
            hasher.finalize_into_reset(leaf);
        }
    });

    // Padding leaves are hashes of an empty column, rather than raw zero hash
    // bytes, so every leaf has an unambiguous domain interpretation.
    for leaf in &mut leaves[codeword_len..] {
        let mut hasher = H::new();
        hasher.finalize_into_reset(leaf);
    }

    let mut tree = vec![leaves];
    while tree.last().unwrap().len() > 1 {
        let previous = tree.last().unwrap();
        let mut next = vec![Output::<H>::default(); previous.len() / 2];
        next.par_iter_mut().enumerate().for_each(|(i, out)| {
            let mut hasher = H::new();
            hasher.update(&previous[2 * i]);
            hasher.update(&previous[2 * i + 1]);
            hasher.finalize_into_reset(out);
        });
        tree.push(next);
    }
    tree
}

fn write_column_opening<F: PrimeField, H: Hash>(
    comm: &BrakingBaseCommitment<F, H>,
    num_rows: usize,
    codeword_len: usize,
    column: usize,
    transcript: &mut impl TranscriptWrite<Output<H>, F>,
) -> Result<(), Error> {
    for row in 0..num_rows {
        transcript.write_field_element(&comm.rows[row * codeword_len + column])?;
    }
    let mut index = column;
    for level in &comm.tree[..comm.tree.len() - 1] {
        transcript.write_commitment(&level[index ^ 1])?;
        index >>= 1;
    }
    Ok(())
}

fn read_and_verify_column<F: PrimeField, H: Hash>(
    root: &Output<H>,
    num_rows: usize,
    codeword_len: usize,
    column: usize,
    transcript: &mut impl TranscriptRead<Output<H>, F>,
) -> Result<Vec<F>, Error> {
    let values = transcript.read_field_elements(num_rows)?;
    let mut hasher = H::new();
    for value in &values {
        hasher.update_field_element(value);
    }
    let mut digest = hasher.finalize_fixed_reset();
    let depth = codeword_len.next_power_of_two().ilog2() as usize;
    let path = transcript.read_commitments(depth)?;
    for (level, sibling) in path.iter().enumerate() {
        if ((column >> level) & 1) == 0 {
            hasher.update(&digest);
            hasher.update(sibling);
        } else {
            hasher.update(sibling);
            hasher.update(&digest);
        }
        digest = hasher.finalize_fixed_reset();
    }
    if &digest != root {
        return Err(Error::InvalidPcsOpen(
            "BrakingBase column Merkle path failed".into(),
        ));
    }
    Ok(values)
}

fn product_tree<F: PrimeField>(leaves: Vec<F>) -> Vec<Vec<F>> {
    assert!(leaves.len().is_power_of_two() && !leaves.is_empty());
    let mut levels = vec![leaves];
    while levels.last().unwrap().len() > 1 {
        let next = levels
            .last()
            .unwrap()
            .chunks_exact(2)
            .map(|pair| pair[0] * pair[1])
            .collect_vec();
        levels.push(next);
    }
    levels
}

/// Batched GKR-style product proof.  The prover work across all layers is
/// linear in the leaf count, while proof size and verifier work are quadratic
/// in its logarithm.
fn prove_products<F: PrimeField, H: Hash>(
    leaf_vectors: Vec<Vec<F>>,
    transcript: &mut impl TranscriptWrite<Output<H>, F>,
) -> Result<(Vec<F>, Vec<F>, Vec<F>), Error> {
    assert!(!leaf_vectors.is_empty());
    let len = leaf_vectors[0].len();
    assert!(len.is_power_of_two());
    assert!(leaf_vectors.iter().all(|v| v.len() == len));
    let trees = leaf_vectors.into_iter().map(product_tree).collect_vec();
    let roots = trees
        .iter()
        .map(|tree| tree.last().unwrap()[0])
        .collect_vec();
    transcript.write_field_elements(&roots)?;

    let mut claims = roots.clone();
    let mut point = Vec::new();
    let num_layers = len.ilog2() as usize;
    for step in 0..num_layers {
        let child_level = num_layers - step - 1;
        let pairs = trees
            .iter()
            .map(|tree| {
                let child = &tree[child_level];
                let left = child.iter().step_by(2).copied().collect_vec();
                let right = child.iter().skip(1).step_by(2).copied().collect_vec();
                (
                    MultilinearPolynomial::new(left),
                    MultilinearPolynomial::new(right),
                )
            })
            .collect_vec();

        let coeffs = transcript.squeeze_challenges(claims.len());
        let combined_claim = inner_product(&claims, &coeffs);
        let eval_point = if point.is_empty() {
            Vec::new()
        } else {
            let expression = pairs
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    Expression::<F>::eq_xy(0)
                        * Expression::<F>::Polynomial(Query::new(2 * i, Rotation::cur()))
                        * Expression::<F>::Polynomial(Query::new(2 * i + 1, Rotation::cur()))
                        * coeffs[i]
                })
                .sum();
            let polys = pairs.iter().flat_map(|(l, r)| [l, r]).collect_vec();
            let ys = [point.clone()];
            let virtual_poly = VirtualPolynomial::new(&expression, polys, &[], &ys);
            SumCheck::<F>::prove(&(), point.len(), virtual_poly, combined_claim, transcript)?.0
        };

        let mut left_evals = Vec::with_capacity(pairs.len());
        let mut right_evals = Vec::with_capacity(pairs.len());
        for (left, right) in &pairs {
            left_evals.push(if eval_point.is_empty() {
                left.evals()[0]
            } else {
                left.evaluate(&eval_point)
            });
            right_evals.push(if eval_point.is_empty() {
                right.evals()[0]
            } else {
                right.evaluate(&eval_point)
            });
        }
        transcript.write_field_elements(&left_evals)?;
        transcript.write_field_elements(&right_evals)?;

        let branch = transcript.squeeze_challenge();
        claims = left_evals
            .iter()
            .zip(&right_evals)
            .map(|(l, r)| *l + branch * (*r - l))
            .collect();
        point = std::iter::once(branch).chain(eval_point).collect();
    }
    Ok((roots, claims, point))
}

fn verify_products<F: PrimeField, H: Hash>(
    num_products: usize,
    len: usize,
    transcript: &mut impl TranscriptRead<Output<H>, F>,
) -> Result<(Vec<F>, Vec<F>, Vec<F>), Error> {
    let roots = transcript.read_field_elements(num_products)?;
    let mut claims = roots.clone();
    let mut point = Vec::new();
    let num_layers = len.ilog2() as usize;
    for _ in 0..num_layers {
        let coeffs = transcript.squeeze_challenges(num_products);
        let combined_claim = inner_product(&claims, &coeffs);
        let (terminal, eval_point) = if point.is_empty() {
            (combined_claim, Vec::new())
        } else {
            SumCheck::<F>::verify(&(), point.len(), 3, combined_claim, transcript)?
        };
        let left = transcript.read_field_elements(num_products)?;
        let right = transcript.read_field_elements(num_products)?;
        let eq = if point.is_empty() {
            F::ONE
        } else {
            crate::piop::sum_check::eq_xy_eval(&eval_point, &point)
        };
        let expected = coeffs.iter().enumerate().fold(F::ZERO, |acc, (i, coeff)| {
            acc + *coeff * left[i] * right[i] * eq
        });
        if terminal != expected {
            return Err(Error::InvalidPcsOpen("SPARK product layer failed".into()));
        }
        let branch = transcript.squeeze_challenge();
        claims = left
            .iter()
            .zip(&right)
            .map(|(l, r)| *l + branch * (*r - l))
            .collect();
        point = std::iter::once(branch).chain(eval_point).collect();
    }
    Ok((roots, claims, point))
}

fn hash_memory_tuple<F: PrimeField>(address: F, value: F, timestamp: F, gamma: F, tau: F) -> F {
    address + gamma * value + gamma.square() * timestamp - tau
}

fn prove_triple_product<F: PrimeField, H: Hash>(
    a: &MultilinearPolynomial<F>,
    b: &MultilinearPolynomial<F>,
    c: &MultilinearPolynomial<F>,
    sum: F,
    transcript: &mut impl TranscriptWrite<Output<H>, F>,
) -> Result<(Vec<F>, [F; 3]), Error> {
    let expression = Expression::<F>::Polynomial(Query::new(0, Rotation::cur()))
        * Expression::<F>::Polynomial(Query::new(1, Rotation::cur()))
        * Expression::<F>::Polynomial(Query::new(2, Rotation::cur()));
    let virtual_poly = VirtualPolynomial::new(&expression, [a, b, c], &[], &[]);
    let (point, _) = SumCheck::<F>::prove(&(), a.num_vars(), virtual_poly, sum, transcript)?;
    let evals = [a.evaluate(&point), b.evaluate(&point), c.evaluate(&point)];
    transcript.write_field_elements(&evals)?;
    Ok((point, evals))
}

fn verify_triple_product<F: PrimeField, H: Hash>(
    num_vars: usize,
    sum: F,
    transcript: &mut impl TranscriptRead<Output<H>, F>,
) -> Result<(Vec<F>, [F; 3]), Error> {
    let (terminal, point) = SumCheck::<F>::verify(&(), num_vars, 3, sum, transcript)?;
    let values = transcript.read_field_elements(3)?;
    let evals = [values[0], values[1], values[2]];
    if terminal != evals[0] * evals[1] * evals[2] {
        return Err(Error::InvalidPcsOpen(
            "SPARK sparse dot-product failed".into(),
        ));
    }
    Ok((point, evals))
}

struct SparkProverOutput<F: PrimeField, H: Hash> {
    dynamic_polys: Vec<MultilinearPolynomial<F>>,
    dynamic_comms: Vec<BasefoldCommitment<F, H>>,
    claims: Vec<OpeningClaim<F>>,
}

fn spark_prove<F, H, V>(
    pp: &BrakingBaseProverParam<F, H>,
    x: &[F],
    y: &[F],
    claimed_eval: F,
    dynamic_offset: usize,
    static_offset: usize,
    transcript: &mut impl TranscriptWrite<Output<H>, F>,
) -> Result<SparkProverOutput<F, H>, Error>
where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
    V: BasefoldExtParams,
{
    let spark = &pp.spark;
    let mem_vars = spark.num_mem_cells.ilog2() as usize;
    let ops_vars = spark.num_ops.ilog2() as usize;
    let aux_vars = spark.aux_len.ilog2() as usize;
    let x_ext = extend_point(x, mem_vars);
    let y_ext = extend_point(y, mem_vars);
    let mem_x = MultilinearPolynomial::eq_xy(&x_ext).into_evals();
    let mem_y = MultilinearPolynomial::eq_xy(&y_ext).into_evals();

    let deref_row = spark
        .row_addr
        .iter()
        .map(|&address| mem_x[address])
        .collect_vec();
    let deref_col = spark
        .col_addr
        .iter()
        .map(|&address| mem_y[address])
        .collect_vec();
    let dynamic_polys = vec![
        pad_poly(deref_row.clone(), spark.aux_len),
        pad_poly(deref_col.clone(), spark.aux_len),
    ];
    let dynamic_comms =
        Bf::<F, H, V>::batch_commit_and_write(&pp.basefold, &dynamic_polys, transcript)?;

    let gamma = transcript.squeeze_challenge();
    let tau = transcript.squeeze_challenge();

    let values = &spark.static_polys[MATRIX_VAL].evals()[..spark.num_ops];
    let sparse_eval = (0..spark.num_ops).fold(F::ZERO, |acc, i| {
        acc + values[i] * deref_row[i] * deref_col[i]
    });
    if sparse_eval != claimed_eval {
        return Err(Error::InvalidPcsOpen(
            "SPARK witness does not match claimed matrix evaluation".into(),
        ));
    }

    let val_poly = MultilinearPolynomial::new(values.to_vec());
    let deref_row_poly = MultilinearPolynomial::new(deref_row.clone());
    let deref_col_poly = MultilinearPolynomial::new(deref_col.clone());
    let (dot_point, dot_evals) = prove_triple_product::<F, H>(
        &val_poly,
        &deref_row_poly,
        &deref_col_poly,
        claimed_eval,
        transcript,
    )?;
    let mut claims = vec![
        OpeningClaim {
            poly: static_offset + MATRIX_VAL,
            point: extend_point(&dot_point, aux_vars),
            value: dot_evals[0],
        },
        OpeningClaim {
            poly: dynamic_offset,
            point: extend_point(&dot_point, aux_vars),
            value: dot_evals[1],
        },
        OpeningClaim {
            poly: dynamic_offset + 1,
            point: extend_point(&dot_point, aux_vars),
            value: dot_evals[2],
        },
    ];

    let row_addr = &spark.static_polys[ROW_ADDR].evals()[..spark.num_ops];
    let col_addr = &spark.static_polys[COL_ADDR].evals()[..spark.num_ops];
    let row_ts = &spark.static_polys[ROW_READ_TS].evals()[..spark.num_ops];
    let col_ts = &spark.static_polys[COL_READ_TS].evals()[..spark.num_ops];
    let row_audit = &spark.static_polys[ROW_AUDIT_TS].evals()[..spark.num_mem_cells];
    let col_audit = &spark.static_polys[COL_AUDIT_TS].evals()[..spark.num_mem_cells];

    let row_init = (0..spark.num_mem_cells)
        .map(|i| hash_memory_tuple(F::from(i as u64), mem_x[i], F::ZERO, gamma, tau))
        .collect_vec();
    let row_audit_hash = (0..spark.num_mem_cells)
        .map(|i| hash_memory_tuple(F::from(i as u64), mem_x[i], row_audit[i], gamma, tau))
        .collect_vec();
    let col_init = (0..spark.num_mem_cells)
        .map(|i| hash_memory_tuple(F::from(i as u64), mem_y[i], F::ZERO, gamma, tau))
        .collect_vec();
    let col_audit_hash = (0..spark.num_mem_cells)
        .map(|i| hash_memory_tuple(F::from(i as u64), mem_y[i], col_audit[i], gamma, tau))
        .collect_vec();

    let row_read = (0..spark.num_ops)
        .map(|i| hash_memory_tuple(row_addr[i], deref_row[i], row_ts[i], gamma, tau))
        .collect_vec();
    let row_write = (0..spark.num_ops)
        .map(|i| hash_memory_tuple(row_addr[i], deref_row[i], row_ts[i] + F::ONE, gamma, tau))
        .collect_vec();
    let col_read = (0..spark.num_ops)
        .map(|i| hash_memory_tuple(col_addr[i], deref_col[i], col_ts[i], gamma, tau))
        .collect_vec();
    let col_write = (0..spark.num_ops)
        .map(|i| hash_memory_tuple(col_addr[i], deref_col[i], col_ts[i] + F::ONE, gamma, tau))
        .collect_vec();

    let (ops_roots, ops_leaf_claims, ops_point) =
        prove_products::<F, H>(vec![row_read, row_write, col_read, col_write], transcript)?;
    let (mem_roots, mem_leaf_claims, mem_point) = prove_products::<F, H>(
        vec![row_init, row_audit_hash, col_init, col_audit_hash],
        transcript,
    )?;
    if mem_roots[0] * ops_roots[1] != ops_roots[0] * mem_roots[1]
        || mem_roots[2] * ops_roots[3] != ops_roots[2] * mem_roots[3]
    {
        return Err(Error::InvalidPcsOpen(
            "SPARK memory multiset check failed".into(),
        ));
    }

    let ops_aux_point = extend_point(&ops_point, aux_vars);
    let ops_values = [
        spark.static_polys[ROW_ADDR].evaluate(&ops_aux_point),
        spark.static_polys[ROW_READ_TS].evaluate(&ops_aux_point),
        spark.static_polys[COL_ADDR].evaluate(&ops_aux_point),
        spark.static_polys[COL_READ_TS].evaluate(&ops_aux_point),
        dynamic_polys[0].evaluate(&ops_aux_point),
        dynamic_polys[1].evaluate(&ops_aux_point),
    ];
    transcript.write_field_elements(&ops_values)?;
    let expected_ops = [
        hash_memory_tuple(ops_values[0], ops_values[4], ops_values[1], gamma, tau),
        hash_memory_tuple(
            ops_values[0],
            ops_values[4],
            ops_values[1] + F::ONE,
            gamma,
            tau,
        ),
        hash_memory_tuple(ops_values[2], ops_values[5], ops_values[3], gamma, tau),
        hash_memory_tuple(
            ops_values[2],
            ops_values[5],
            ops_values[3] + F::ONE,
            gamma,
            tau,
        ),
    ];
    if ops_leaf_claims != expected_ops {
        return Err(Error::InvalidPcsOpen(
            "SPARK operation hash opening failed".into(),
        ));
    }
    for (poly, value) in [
        (static_offset + ROW_ADDR, ops_values[0]),
        (static_offset + ROW_READ_TS, ops_values[1]),
        (static_offset + COL_ADDR, ops_values[2]),
        (static_offset + COL_READ_TS, ops_values[3]),
        (dynamic_offset, ops_values[4]),
        (dynamic_offset + 1, ops_values[5]),
    ] {
        claims.push(OpeningClaim {
            poly,
            point: ops_aux_point.clone(),
            value,
        });
    }

    let mem_aux_point = extend_point(&mem_point, aux_vars);
    let audit_values = [
        spark.static_polys[ROW_AUDIT_TS].evaluate(&mem_aux_point),
        spark.static_polys[COL_AUDIT_TS].evaluate(&mem_aux_point),
    ];
    transcript.write_field_elements(&audit_values)?;
    let address = identity_eval(&mem_point);
    let row_value = crate::piop::sum_check::eq_xy_eval(&mem_point, &x_ext);
    let col_value = crate::piop::sum_check::eq_xy_eval(&mem_point, &y_ext);
    let expected_mem = [
        hash_memory_tuple(address, row_value, F::ZERO, gamma, tau),
        hash_memory_tuple(address, row_value, audit_values[0], gamma, tau),
        hash_memory_tuple(address, col_value, F::ZERO, gamma, tau),
        hash_memory_tuple(address, col_value, audit_values[1], gamma, tau),
    ];
    if mem_leaf_claims != expected_mem {
        return Err(Error::InvalidPcsOpen(
            "SPARK memory hash opening failed".into(),
        ));
    }
    claims.push(OpeningClaim {
        poly: static_offset + ROW_AUDIT_TS,
        point: mem_aux_point.clone(),
        value: audit_values[0],
    });
    claims.push(OpeningClaim {
        poly: static_offset + COL_AUDIT_TS,
        point: mem_aux_point,
        value: audit_values[1],
    });

    Ok(SparkProverOutput {
        dynamic_polys,
        dynamic_comms,
        claims,
    })
}

fn spark_verify<F, H, V>(
    vp: &BrakingBaseVerifierParam<F, H>,
    x: &[F],
    y: &[F],
    claimed_eval: F,
    dynamic_offset: usize,
    static_offset: usize,
    transcript: &mut impl TranscriptRead<Output<H>, F>,
) -> Result<(Vec<BasefoldCommitment<F, H>>, Vec<OpeningClaim<F>>), Error>
where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
    V: BasefoldExtParams,
{
    let mem_vars = vp.num_mem_cells.ilog2() as usize;
    let ops_vars = vp.num_ops.ilog2() as usize;
    let aux_vars = vp.aux_len.ilog2() as usize;
    let x_ext = extend_point(x, mem_vars);
    let y_ext = extend_point(y, mem_vars);
    let dynamic_comms = Bf::<F, H, V>::read_commitments(&vp.basefold, 2, transcript)?;
    let gamma = transcript.squeeze_challenge();
    let tau = transcript.squeeze_challenge();

    let (dot_point, dot_evals) = verify_triple_product::<F, H>(ops_vars, claimed_eval, transcript)?;
    let mut claims = vec![
        OpeningClaim {
            poly: static_offset + MATRIX_VAL,
            point: extend_point(&dot_point, aux_vars),
            value: dot_evals[0],
        },
        OpeningClaim {
            poly: dynamic_offset,
            point: extend_point(&dot_point, aux_vars),
            value: dot_evals[1],
        },
        OpeningClaim {
            poly: dynamic_offset + 1,
            point: extend_point(&dot_point, aux_vars),
            value: dot_evals[2],
        },
    ];

    let (ops_roots, ops_leaf_claims, ops_point) =
        verify_products::<F, H>(4, vp.num_ops, transcript)?;
    let (mem_roots, mem_leaf_claims, mem_point) =
        verify_products::<F, H>(4, vp.num_mem_cells, transcript)?;
    if mem_roots[0] * ops_roots[1] != ops_roots[0] * mem_roots[1]
        || mem_roots[2] * ops_roots[3] != ops_roots[2] * mem_roots[3]
    {
        return Err(Error::InvalidPcsOpen(
            "SPARK memory multiset check failed".into(),
        ));
    }

    let ops_values = transcript.read_field_elements(6)?;
    let expected_ops = [
        hash_memory_tuple(ops_values[0], ops_values[4], ops_values[1], gamma, tau),
        hash_memory_tuple(
            ops_values[0],
            ops_values[4],
            ops_values[1] + F::ONE,
            gamma,
            tau,
        ),
        hash_memory_tuple(ops_values[2], ops_values[5], ops_values[3], gamma, tau),
        hash_memory_tuple(
            ops_values[2],
            ops_values[5],
            ops_values[3] + F::ONE,
            gamma,
            tau,
        ),
    ];
    if ops_leaf_claims != expected_ops {
        return Err(Error::InvalidPcsOpen(
            "SPARK operation hash opening failed".into(),
        ));
    }
    let ops_aux_point = extend_point(&ops_point, aux_vars);
    for (poly, value) in [
        (static_offset + ROW_ADDR, ops_values[0]),
        (static_offset + ROW_READ_TS, ops_values[1]),
        (static_offset + COL_ADDR, ops_values[2]),
        (static_offset + COL_READ_TS, ops_values[3]),
        (dynamic_offset, ops_values[4]),
        (dynamic_offset + 1, ops_values[5]),
    ] {
        claims.push(OpeningClaim {
            poly,
            point: ops_aux_point.clone(),
            value,
        });
    }

    let audit_values = transcript.read_field_elements(2)?;
    let address = identity_eval(&mem_point);
    let row_value = crate::piop::sum_check::eq_xy_eval(&mem_point, &x_ext);
    let col_value = crate::piop::sum_check::eq_xy_eval(&mem_point, &y_ext);
    let expected_mem = [
        hash_memory_tuple(address, row_value, F::ZERO, gamma, tau),
        hash_memory_tuple(address, row_value, audit_values[0], gamma, tau),
        hash_memory_tuple(address, col_value, F::ZERO, gamma, tau),
        hash_memory_tuple(address, col_value, audit_values[1], gamma, tau),
    ];
    if mem_leaf_claims != expected_mem {
        return Err(Error::InvalidPcsOpen(
            "SPARK memory hash opening failed".into(),
        ));
    }
    let mem_aux_point = extend_point(&mem_point, aux_vars);
    claims.push(OpeningClaim {
        poly: static_offset + ROW_AUDIT_TS,
        point: mem_aux_point.clone(),
        value: audit_values[0],
    });
    claims.push(OpeningClaim {
        poly: static_offset + COL_AUDIT_TS,
        point: mem_aux_point,
        value: audit_values[1],
    });

    Ok((dynamic_comms, claims))
}

fn prove_two_factor<F: PrimeField, H: Hash>(
    selector: &MultilinearPolynomial<F>,
    p: &MultilinearPolynomial<F>,
    q: &MultilinearPolynomial<F>,
    eta: F,
    sum: F,
    transcript: &mut impl TranscriptWrite<Output<H>, F>,
) -> Result<(Vec<F>, [F; 3]), Error> {
    let expression = Expression::<F>::Polynomial(Query::new(0, Rotation::cur()))
        * (Expression::<F>::Polynomial(Query::new(1, Rotation::cur()))
            + Expression::<F>::Polynomial(Query::new(2, Rotation::cur())) * eta);
    let virtual_poly = VirtualPolynomial::new(&expression, [selector, p, q], &[], &[]);
    let (point, _) = SumCheck::<F>::prove(&(), selector.num_vars(), virtual_poly, sum, transcript)?;
    let evals = [
        selector.evaluate(&point),
        p.evaluate(&point),
        q.evaluate(&point),
    ];
    transcript.write_field_elements(&evals[1..])?;
    Ok((point, evals))
}

fn verify_two_factor<F: PrimeField, H: Hash>(
    num_vars: usize,
    eta: F,
    sum: F,
    selector_eval: impl FnOnce(&[F]) -> F,
    transcript: &mut impl TranscriptRead<Output<H>, F>,
) -> Result<(Vec<F>, [F; 3]), Error> {
    let (terminal, point) = SumCheck::<F>::verify(&(), num_vars, 2, sum, transcript)?;
    let pq = transcript.read_field_elements(2)?;
    let evals = [selector_eval(&point), pq[0], pq[1]];
    if terminal != evals[0] * (evals[1] + eta * evals[2]) {
        return Err(Error::InvalidPcsOpen(
            "BrakingBase sumcheck terminal failed".into(),
        ));
    }
    Ok((point, evals))
}

fn prove_code_decomposition<F: PrimeField, H: Hash>(
    aux_polys: &[MultilinearPolynomial<F>],
    code_point: &[F],
    p_eval: F,
    q_eval: F,
    aux_vars: usize,
    transcript: &mut impl TranscriptWrite<Output<H>, F>,
) -> Result<Vec<OpeningClaim<F>>, Error> {
    assert_eq!(aux_polys.len(), 4);
    let (&branch, base_point) = code_point.split_last().unwrap();
    let point = extend_point(base_point, aux_vars);
    let values = aux_polys
        .iter()
        .map(|poly| poly.evaluate(&point))
        .collect_vec();
    transcript.write_field_elements(&values)?;
    if p_eval != values[0] + branch * (values[1] - values[0])
        || q_eval != values[2] + branch * (values[3] - values[2])
    {
        return Err(Error::InvalidPcsOpen(
            "codeword split evaluation mismatch".into(),
        ));
    }
    Ok(values
        .into_iter()
        .enumerate()
        .map(|(poly, value)| OpeningClaim {
            poly,
            point: point.clone(),
            value,
        })
        .collect())
}

fn verify_code_decomposition<F: PrimeField, H: Hash>(
    code_point: &[F],
    p_eval: F,
    q_eval: F,
    aux_vars: usize,
    transcript: &mut impl TranscriptRead<Output<H>, F>,
) -> Result<Vec<OpeningClaim<F>>, Error> {
    let (&branch, base_point) = code_point.split_last().unwrap();
    let point = extend_point(base_point, aux_vars);
    let values = transcript.read_field_elements(4)?;
    if p_eval != values[0] + branch * (values[1] - values[0])
        || q_eval != values[2] + branch * (values[3] - values[2])
    {
        return Err(Error::InvalidPcsOpen(
            "codeword split evaluation mismatch".into(),
        ));
    }
    Ok(values
        .into_iter()
        .enumerate()
        .map(|(poly, value)| OpeningClaim {
            poly,
            point: point.clone(),
            value,
        })
        .collect())
}

fn combine_rows<F: PrimeField>(
    evals: &[F],
    num_rows: usize,
    row_len: usize,
    coefficients: &[F],
) -> Vec<F> {
    assert_eq!(evals.len(), num_rows * row_len);
    assert_eq!(coefficients.len(), num_rows);
    let mut out = vec![F::ZERO; row_len];
    out.par_iter_mut().enumerate().for_each(|(column, value)| {
        *value = (0..num_rows).fold(F::ZERO, |acc, row| {
            acc + coefficients[row] * evals[row * row_len + column]
        });
    });
    out
}

fn parity_selector<F: PrimeField>(
    spark: &SparkProverData<F>,
    codeword_len: usize,
    point: &[F],
) -> MultilinearPolynomial<F> {
    let mut values = vec![F::ZERO; codeword_len];
    for entry in &spark.entries {
        values[entry.row] += entry.value * eq_index(point, entry.col);
    }
    MultilinearPolynomial::new(values)
}

fn batch_open_all<F, H, V>(
    pp: &BasefoldProverParams<F>,
    polys: &[MultilinearPolynomial<F>],
    comms: &[BasefoldCommitment<F, H>],
    claims: &[OpeningClaim<F>],
    transcript: &mut impl TranscriptWrite<Output<H>, F>,
) -> Result<(), Error>
where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
    V: BasefoldExtParams,
{
    let points = claims.iter().map(|claim| claim.point.clone()).collect_vec();
    let evals = claims
        .iter()
        .enumerate()
        .map(|(point, claim)| Evaluation::new(claim.poly, point, claim.value))
        .collect_vec();
    Bf::<F, H, V>::batch_open(pp, polys, comms, &points, &evals, transcript)
}

fn batch_verify_all<F, H, V>(
    vp: &BasefoldVerifierParams<F>,
    comms: &[BasefoldCommitment<F, H>],
    claims: &[OpeningClaim<F>],
    transcript: &mut impl TranscriptRead<Output<H>, F>,
) -> Result<(), Error>
where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
    V: BasefoldExtParams,
{
    let points = claims.iter().map(|claim| claim.point.clone()).collect_vec();
    let evals = claims
        .iter()
        .enumerate()
        .map(|(point, claim)| Evaluation::new(claim.poly, point, claim.value))
        .collect_vec();
    Bf::<F, H, V>::batch_verify(vp, comms, &points, &evals, transcript)
}

impl<F, H, S, V> MultilinearBrakingBase<F, H, S, V>
where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
    S: BrakedownSpec,
    V: BasefoldExtParams,
{
    /// Generate BrakingBase parameters with an explicit matrix row count.
    ///
    /// Both dimensions must be powers of two because they index multilinear
    /// polynomials over Boolean hypercubes.  Consequently `num_rows` must be a
    /// power-of-two divisor of `poly_size`.
    pub fn setup_with_num_rows(
        poly_size: usize,
        num_rows: usize,
        mut rng: impl RngCore,
    ) -> Result<BrakingBaseParams<F, H>, Error> {
        if !poly_size.is_power_of_two() {
            return Err(Error::InvalidPcsParam(
                "BrakingBase polynomial size must be a power of two".into(),
            ));
        }
        if !num_rows.is_power_of_two() || num_rows > poly_size || poly_size % num_rows != 0 {
            return Err(Error::InvalidPcsParam(
                "BrakingBase row count must be a power-of-two divisor of the polynomial size"
                    .into(),
            ));
        }
        let num_vars = poly_size.ilog2() as usize;
        let row_len = poly_size / num_rows;
        if row_len <= BASE_CODE_THRESHOLD {
            return Err(Error::InvalidPcsParam(
                "BrakingBase row length is below the Brakedown base threshold".into(),
            ));
        }
        let brakedown = Brakedown::new_with_row_len::<S>(row_len, BASE_CODE_THRESHOLD, &mut rng);
        if brakedown.num_proximity_testing() != 1 {
            return Err(Error::InvalidPcsParam(
                "BrakingBase requires a field/parameter set with one random-row proximity test"
                    .into(),
            ));
        }
        let codeword_len = brakedown.codeword_len();
        if codeword_len != 2 * row_len {
            return Err(Error::InvalidPcsParam(format!(
                "BrakingBase currently requires an exact rate-1/2 Brakedown code (got {codeword_len}/{row_len})"
            )));
        }
        let entries = parity_check_entries(&brakedown);
        let spark = spark_preprocess(entries, codeword_len, row_len, row_len);
        let bf_param = Bf::<F, H, V>::setup(spark.aux_len, NUM_SPARK_STATIC_POLYS + 6, &mut rng)?;
        let (bf_pp, bf_vp) =
            Bf::<F, H, V>::trim(&bf_param, spark.aux_len, NUM_SPARK_STATIC_POLYS + 6)?;
        let spark_comms = Bf::<F, H, V>::batch_commit(&bf_pp, &spark.static_polys)?;
        let spark_verifier_comms = spark_comms
            .iter()
            .map(BasefoldCommitment::verifier_only)
            .collect();
        let num_queries = brakedown.num_column_opening();

        let prover = BrakingBaseProverParam {
            num_vars,
            num_rows,
            row_len,
            codeword_len,
            num_queries,
            brakedown,
            spark: spark.clone(),
            basefold: bf_pp,
            spark_comms: spark_comms.clone(),
        };
        let verifier = BrakingBaseVerifierParam {
            num_vars,
            num_rows,
            row_len,
            codeword_len,
            num_queries,
            num_ops: spark.num_ops,
            num_mem_cells: spark.num_mem_cells,
            aux_len: spark.aux_len,
            basefold: bf_vp,
            spark_comms: spark_verifier_comms,
        };
        Ok(BrakingBaseParams { prover, verifier })
    }
}

impl<F, H, S, V> PolynomialCommitmentScheme<F> for MultilinearBrakingBase<F, H, S, V>
where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
    S: BrakedownSpec,
    V: BasefoldExtParams,
{
    type Param = BrakingBaseParams<F, H>;
    type ProverParam = BrakingBaseProverParam<F, H>;
    type VerifierParam = BrakingBaseVerifierParam<F, H>;
    type Polynomial = MultilinearPolynomial<F>;
    type Commitment = BrakingBaseCommitment<F, H>;
    type CommitmentChunk = Output<H>;

    fn setup(poly_size: usize, _: usize, rng: impl RngCore) -> Result<Self::Param, Error> {
        let num_rows = if poly_size.is_power_of_two() {
            brakingbase_num_rows(poly_size.ilog2() as usize)
        } else {
            1
        };
        Self::setup_with_num_rows(poly_size, num_rows, rng)
    }

    fn trim(
        param: &Self::Param,
        poly_size: usize,
        _: usize,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), Error> {
        if poly_size != (1usize << param.prover.num_vars) {
            return Err(Error::InvalidPcsParam(
                "BrakingBase does not support trimming to a different size".into(),
            ));
        }
        Ok((param.prover.clone(), param.verifier.clone()))
    }

    fn commit(pp: &Self::ProverParam, poly: &Self::Polynomial) -> Result<Self::Commitment, Error> {
        validate_input("commit", pp.num_vars, [poly], None)?;
        let mut rows = vec![F::ZERO; pp.num_rows * pp.codeword_len];
        let chunk_rows = ((pp.num_rows + num_threads() - 1) / num_threads()).max(1);
        parallelize_iter(
            rows.chunks_mut(chunk_rows * pp.codeword_len)
                .zip(poly.evals().chunks(chunk_rows * pp.row_len)),
            |(out, input)| {
                for (codeword, message) in out
                    .chunks_mut(pp.codeword_len)
                    .zip(input.chunks(pp.row_len))
                {
                    codeword[..pp.row_len].copy_from_slice(message);
                    pp.brakedown.encode(codeword);
                }
            },
        );
        let tree = merkelize_columns::<F, H>(&rows, pp.num_rows, pp.codeword_len);
        let root = tree.last().unwrap()[0].clone();
        Ok(BrakingBaseCommitment { rows, tree, root })
    }

    fn batch_commit<'a>(
        pp: &Self::ProverParam,
        polys: impl IntoIterator<Item = &'a Self::Polynomial>,
    ) -> Result<Vec<Self::Commitment>, Error>
    where
        Self::Polynomial: 'a,
    {
        polys
            .into_iter()
            .map(|poly| Self::commit(pp, poly))
            .collect()
    }

    fn open(
        pp: &Self::ProverParam,
        poly: &Self::Polynomial,
        comm: &Self::Commitment,
        point: &Point<F, Self::Polynomial>,
        eval: &F,
        transcript: &mut impl TranscriptWrite<Self::CommitmentChunk, F>,
    ) -> Result<(), Error> {
        validate_input("open", pp.num_vars, [poly], [point])?;
        if comm.rows.len() != pp.num_rows * pp.codeword_len || comm.tree.is_empty() {
            return Err(Error::InvalidPcsOpen(
                "BrakingBase prover commitment lacks opening data".into(),
            ));
        }
        let log_rows = pp.num_rows.ilog2() as usize;
        let (column_point, row_point) = point.split_at(pp.num_vars - log_rows);
        let row_weights = MultilinearPolynomial::eq_xy(row_point).into_evals();

        // Protocol 4, Steps 1--3: proximity and evaluation row combinations.
        let proximity_weights = transcript.squeeze_challenges(pp.num_rows);
        let p_message = combine_rows(poly.evals(), pp.num_rows, pp.row_len, &proximity_weights);
        let q_message = combine_rows(poly.evals(), pp.num_rows, pp.row_len, &row_weights);
        let mut p_codeword = vec![F::ZERO; pp.codeword_len];
        let mut q_codeword = vec![F::ZERO; pp.codeword_len];
        p_codeword[..pp.row_len].copy_from_slice(&p_message);
        q_codeword[..pp.row_len].copy_from_slice(&q_message);
        pp.brakedown.encode(&mut p_codeword);
        pp.brakedown.encode(&mut q_codeword);

        let aux_polys = vec![
            pad_poly(p_codeword[..pp.row_len].to_vec(), pp.spark.aux_len),
            pad_poly(p_codeword[pp.row_len..].to_vec(), pp.spark.aux_len),
            pad_poly(q_codeword[..pp.row_len].to_vec(), pp.spark.aux_len),
            pad_poly(q_codeword[pp.row_len..].to_vec(), pp.spark.aux_len),
        ];
        let aux_comms =
            Bf::<F, H, V>::batch_commit_and_write(&pp.basefold, &aux_polys, transcript)?;

        // Steps 4--7: open sampled committed columns and prove the two folded
        // column relations in one randomly batched sumcheck.
        let query_indices = (0..pp.num_queries)
            .map(|_| challenge_index(transcript, pp.codeword_len))
            .collect_vec();
        for &column in &query_indices {
            write_column_opening(comm, pp.num_rows, pp.codeword_len, column, transcript)?;
        }
        let query_weights = transcript.squeeze_challenges(pp.num_queries);
        let eta_columns = transcript.squeeze_challenge();
        let mut lhs_p = F::ZERO;
        let mut lhs_q = F::ZERO;
        let mut mask = vec![F::ZERO; pp.codeword_len];
        for ((&column, weight), _) in query_indices.iter().zip(&query_weights).zip(0..) {
            let column_values = (0..pp.num_rows)
                .map(|row| comm.rows[row * pp.codeword_len + column])
                .collect_vec();
            lhs_p += *weight * inner_product(&proximity_weights, &column_values);
            lhs_q += *weight * inner_product(&row_weights, &column_values);
            mask[column] += weight;
        }
        let mask_poly = MultilinearPolynomial::new(mask);
        let p_code_poly = MultilinearPolynomial::new(p_codeword.clone());
        let q_code_poly = MultilinearPolynomial::new(q_codeword.clone());
        let (beta, consistency_evals) = prove_two_factor::<F, H>(
            &mask_poly,
            &p_code_poly,
            &q_code_poly,
            eta_columns,
            lhs_p + eta_columns * lhs_q,
            transcript,
        )?;
        let aux_vars = pp.spark.aux_len.ilog2() as usize;
        let mut claims = prove_code_decomposition::<F, H>(
            &aux_polys,
            &beta,
            consistency_evals[1],
            consistency_evals[2],
            aux_vars,
            transcript,
        )?;

        // Steps 9--13: code membership via the sparse parity-check matrix.
        let parity_point = transcript.squeeze_challenges(pp.row_len.ilog2() as usize);
        let h_selector = parity_selector(&pp.spark, pp.codeword_len, &parity_point);
        let eta_parity = transcript.squeeze_challenge();
        let p_parity = inner_product(p_codeword.iter(), h_selector.evals());
        let q_parity = inner_product(q_codeword.iter(), h_selector.evals());
        if p_parity != F::ZERO || q_parity != F::ZERO {
            return Err(Error::InvalidPcsOpen(
                "constructed Brakedown word failed parity check".into(),
            ));
        }
        let (gamma_point, parity_evals) = prove_two_factor::<F, H>(
            &h_selector,
            &p_code_poly,
            &q_code_poly,
            eta_parity,
            F::ZERO,
            transcript,
        )?;
        transcript.write_field_element(&parity_evals[0])?;
        claims.extend(prove_code_decomposition::<F, H>(
            &aux_polys,
            &gamma_point,
            parity_evals[1],
            parity_evals[2],
            aux_vars,
            transcript,
        )?);

        let dynamic_offset = aux_polys.len();
        let static_offset = dynamic_offset + 2;
        let spark_output = spark_prove::<F, H, V>(
            pp,
            &gamma_point,
            &parity_point,
            parity_evals[0],
            dynamic_offset,
            static_offset,
            transcript,
        )?;
        claims.extend(spark_output.claims);

        // The evaluation-folded message q must evaluate to the public claim at
        // the remaining (column) coordinates.
        let q_eval_point = extend_point(column_point, aux_vars);
        let q_eval = aux_polys[2].evaluate(&q_eval_point);
        if q_eval != *eval {
            return Err(Error::InvalidPcsOpen(
                "BrakingBase public evaluation mismatch".into(),
            ));
        }
        claims.push(OpeningClaim {
            poly: 2,
            point: q_eval_point,
            value: *eval,
        });

        let all_polys = aux_polys
            .iter()
            .chain(spark_output.dynamic_polys.iter())
            .chain(pp.spark.static_polys.iter())
            .cloned()
            .collect_vec();
        let all_comms = aux_comms
            .iter()
            .chain(spark_output.dynamic_comms.iter())
            .chain(pp.spark_comms.iter())
            .cloned()
            .collect_vec();
        batch_open_all::<F, H, V>(&pp.basefold, &all_polys, &all_comms, &claims, transcript)
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
        for claim in evals {
            Self::open(
                pp,
                polys[claim.poly()],
                comms[claim.poly()],
                &points[claim.point()],
                claim.value(),
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
        Ok(transcript
            .read_commitments(num_polys)?
            .into_iter()
            .map(BrakingBaseCommitment::from_root)
            .collect())
    }

    fn verify(
        vp: &Self::VerifierParam,
        comm: &Self::Commitment,
        point: &Point<F, Self::Polynomial>,
        eval: &F,
        transcript: &mut impl TranscriptRead<Self::CommitmentChunk, F>,
    ) -> Result<(), Error> {
        validate_input("verify", vp.num_vars, [], [point])?;
        let log_rows = vp.num_rows.ilog2() as usize;
        let (column_point, row_point) = point.split_at(vp.num_vars - log_rows);
        let row_weights = MultilinearPolynomial::eq_xy(row_point).into_evals();
        let proximity_weights = transcript.squeeze_challenges(vp.num_rows);
        let aux_comms = Bf::<F, H, V>::read_commitments(&vp.basefold, 4, transcript)?;

        let query_indices = (0..vp.num_queries)
            .map(|_| challenge_index(transcript, vp.codeword_len))
            .collect_vec();
        let mut columns = Vec::with_capacity(vp.num_queries);
        for &column in &query_indices {
            columns.push(read_and_verify_column::<F, H>(
                comm.root(),
                vp.num_rows,
                vp.codeword_len,
                column,
                transcript,
            )?);
        }
        let query_weights = transcript.squeeze_challenges(vp.num_queries);
        let eta_columns = transcript.squeeze_challenge();
        let mut lhs_p = F::ZERO;
        let mut lhs_q = F::ZERO;
        for (i, column) in columns.iter().enumerate() {
            lhs_p += query_weights[i] * inner_product(&proximity_weights, column);
            lhs_q += query_weights[i] * inner_product(&row_weights, column);
        }
        let mask_eval = |beta: &[F]| {
            query_indices
                .iter()
                .zip(&query_weights)
                .fold(F::ZERO, |acc, (&index, weight)| {
                    acc + *weight * eq_index(beta, index)
                })
        };
        let (beta, consistency_evals) = verify_two_factor::<F, H>(
            vp.codeword_len.ilog2() as usize,
            eta_columns,
            lhs_p + eta_columns * lhs_q,
            mask_eval,
            transcript,
        )?;
        let aux_vars = vp.aux_len.ilog2() as usize;
        let mut claims = verify_code_decomposition::<F, H>(
            &beta,
            consistency_evals[1],
            consistency_evals[2],
            aux_vars,
            transcript,
        )?;

        let parity_point = transcript.squeeze_challenges(vp.row_len.ilog2() as usize);
        let eta_parity = transcript.squeeze_challenge();
        // The parity selector evaluation is supplied as the third terminal
        // value, then certified by SPARK below.
        let (parity_terminal, gamma_point) = SumCheck::<F>::verify(
            &(),
            vp.codeword_len.ilog2() as usize,
            2,
            F::ZERO,
            transcript,
        )?;
        let pq = transcript.read_field_elements(2)?;
        let h_eval = if pq.len() == 2 {
            // Written after p/q so that the transcript order matches the
            // generic two-factor prover; SPARK binds this value immediately.
            transcript.read_field_element()?
        } else {
            unreachable!()
        };
        if parity_terminal != h_eval * (pq[0] + eta_parity * pq[1]) {
            return Err(Error::InvalidPcsOpen(
                "BrakingBase parity sumcheck terminal failed".into(),
            ));
        }
        claims.extend(verify_code_decomposition::<F, H>(
            &gamma_point,
            pq[0],
            pq[1],
            aux_vars,
            transcript,
        )?);

        let dynamic_offset = 4;
        let static_offset = 6;
        let (dynamic_comms, spark_claims) = spark_verify::<F, H, V>(
            vp,
            &gamma_point,
            &parity_point,
            h_eval,
            dynamic_offset,
            static_offset,
            transcript,
        )?;
        claims.extend(spark_claims);
        claims.push(OpeningClaim {
            poly: 2,
            point: extend_point(column_point, aux_vars),
            value: *eval,
        });

        let all_comms = aux_comms
            .iter()
            .chain(dynamic_comms.iter())
            .chain(vp.spark_comms.iter())
            .cloned()
            .collect_vec();
        batch_verify_all::<F, H, V>(&vp.basefold, &all_comms, &claims, transcript)
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
        for claim in evals {
            Self::verify(
                vp,
                comms[claim.poly()],
                &points[claim.point()],
                claim.value(),
                transcript,
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::{
        arithmetic::Field,
        hash::Blake2s,
        new_fields::Mersenne127,
        test::seeded_std_rng,
        transcript::{
            Blake2sTranscript, FieldTranscriptRead, FieldTranscriptWrite, InMemoryTranscript,
        },
    };

    #[derive(Debug)]
    struct TestBrakedown;

    impl BrakedownSpec for TestBrakedown {
        const LAMBDA: f64 = 8.0;
        const ALPHA: f64 = 0.30;
        const BETA: f64 = 0.20;
        const R: f64 = 2.0;

        fn c_n(_: usize) -> usize {
            4
        }
        fn d_n(_: usize, _: usize) -> usize {
            4
        }
        fn num_column_opening() -> usize {
            2
        }
    }

    #[derive(Debug)]
    struct TestBaseFold;

    impl BasefoldExtParams for TestBaseFold {
        fn get_reps() -> usize {
            2
        }
        fn get_rate() -> usize {
            1
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

    type F = Mersenne127;
    type Pcs = MultilinearBrakingBase<F, Blake2s, TestBrakedown, TestBaseFold>;

    #[test]
    fn parity_matrix_annihilates_encoder_output() {
        let mut rng = seeded_std_rng();
        let code =
            Brakedown::<F>::new_with_row_len::<TestBrakedown>(64, BASE_CODE_THRESHOLD, &mut rng);
        assert_eq!(code.codeword_len(), 2 * code.row_len());
        let mut word = (0..code.row_len())
            .map(|_| F::random(&mut rng))
            .collect_vec();
        word.resize(code.codeword_len(), F::ZERO);
        code.encode(&mut word);
        let entries = parity_check_entries(&code);
        let mut syndrome = vec![F::ZERO; code.codeword_len() - code.row_len()];
        for entry in entries {
            syndrome[entry.col] += word[entry.row] * entry.value;
        }
        assert!(syndrome.iter().all(|value| *value == F::ZERO));
    }

    #[test]
    fn commit_open_verify() {
        let num_vars = 10;
        let poly_size = 1usize << num_vars;
        let mut rng = seeded_std_rng();
        let params = Pcs::setup_with_num_rows(poly_size, 8, &mut rng).unwrap();
        let (pp, vp) = Pcs::trim(&params, poly_size, 1).unwrap();
        assert_eq!(pp.num_rows(), 8);
        let poly = MultilinearPolynomial::<F>::rand(num_vars, &mut rng);

        let proof = {
            let mut transcript = Blake2sTranscript::new(());
            let comm = Pcs::commit_and_write(&pp, &poly, &mut transcript).unwrap();
            let point = transcript.squeeze_challenges(num_vars);
            let eval = poly.evaluate(&point);
            transcript.write_field_element(&eval).unwrap();
            Pcs::open(&pp, &poly, &comm, &point, &eval, &mut transcript).unwrap();
            transcript.into_proof()
        };

        let mut transcript = Blake2sTranscript::from_proof((), &proof);
        let comm = Pcs::read_commitment(&vp, &mut transcript).unwrap();
        let point = transcript.squeeze_challenges(num_vars);
        let eval = transcript.read_field_element().unwrap();
        Pcs::verify(&vp, &comm, &point, &eval, &mut transcript).unwrap();
    }

    #[test]
    fn tampered_commitment_is_rejected() {
        let num_vars = 10;
        let poly_size = 1usize << num_vars;
        let mut rng = seeded_std_rng();
        let params = Pcs::setup(poly_size, 1, &mut rng).unwrap();
        let (pp, vp) = Pcs::trim(&params, poly_size, 1).unwrap();
        let poly = MultilinearPolynomial::<F>::rand(num_vars, &mut rng);

        let mut transcript = Blake2sTranscript::new(());
        let comm = Pcs::commit_and_write(&pp, &poly, &mut transcript).unwrap();
        let point = transcript.squeeze_challenges(num_vars);
        let eval = poly.evaluate(&point);
        transcript.write_field_element(&eval).unwrap();
        Pcs::open(&pp, &poly, &comm, &point, &eval, &mut transcript).unwrap();
        let mut proof = transcript.into_proof();

        // The first transcript item is the BrakingBase Merkle root.  Changing
        // it must alter every following Fiat--Shamir challenge and be rejected.
        proof[0] ^= 1;
        let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut transcript = Blake2sTranscript::from_proof((), &proof);
            let comm = Pcs::read_commitment(&vp, &mut transcript).unwrap();
            let point = transcript.squeeze_challenges(num_vars);
            let eval = transcript.read_field_element().unwrap();
            Pcs::verify(&vp, &comm, &point, &eval, &mut transcript)
        }))
        .map_or(true, |result| result.is_err());
        assert!(rejected);
    }
}
