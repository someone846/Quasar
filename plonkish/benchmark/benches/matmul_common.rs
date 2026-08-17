//! Shared non-ZK matrix-multiplication application used by the CPU and CUDA
//! benchmark drivers.  Keeping the application reduction and sumcheck here
//! ensures that both drivers benchmark exactly the same statement.

use plonkish_backend::util::{
    arithmetic::Field,
    new_fields::Mersenne127,
    transcript::{Blake2sTranscript, FieldTranscript, InMemoryTranscript},
};
use rand_chacha::{rand_core::SeedableRng, ChaCha8Rng};
use rayon::prelude::*;
use std::io::Cursor;

pub type BenchField = Mersenne127;
type SumcheckTranscript = Blake2sTranscript<Cursor<Vec<u8>>>;

#[derive(Clone, Debug)]
pub struct MatmulShape {
    pub k: usize,
    pub log_m: usize,
    pub log_n: usize,
    pub log_p: usize,
    pub m: usize,
    pub n: usize,
    pub p: usize,
}

impl MatmulShape {
    pub fn from_k(k: usize, log_m: usize) -> Self {
        // B has exactly 2^k entries. Keep a transformer-like expansion:
        // even k: p=4n; odd k: p=2n.
        assert!(k >= 4);
        let log_n = if k % 2 == 0 { (k - 2) / 2 } else { (k - 1) / 2 };
        let log_p = k - log_n;
        let m = 1usize << log_m;
        let n = 1usize << log_n;
        let p = 1usize << log_p;
        assert_eq!(n.checked_mul(p).expect("B size overflow"), 1usize << k);
        Self {
            k,
            log_m,
            log_n,
            log_p,
            m,
            n,
            p,
        }
    }

    pub fn log_a(&self) -> usize {
        self.log_m + self.log_n
    }

    pub fn log_b(&self) -> usize {
        self.log_n + self.log_p
    }

    pub fn log_c(&self) -> usize {
        self.log_m + self.log_p
    }
}

/// Dense row-major evaluation tables for a valid product.
pub struct MatmulEvals {
    pub a: Vec<BenchField>,
    pub b: Vec<BenchField>,
    pub c: Vec<BenchField>,
}

pub fn build_evals(shape: &MatmulShape, seed: [u8; 32]) -> MatmulEvals {
    let mut rng = ChaCha8Rng::from_seed(seed);
    let u = random_vec(shape.m, &mut rng);
    let v = random_vec(shape.n, &mut rng);
    let s = random_vec(shape.n, &mut rng);
    let w = random_vec(shape.p, &mut rng);
    let gamma = v
        .iter()
        .zip(s.iter())
        .fold(BenchField::ZERO, |acc, (x, y)| acc + *x * *y);

    // Rank one is used only to construct the witness. All proving code below
    // consumes the materialized dense tables.
    let a = (0..shape.m * shape.n)
        .into_par_iter()
        .map(|idx| u[idx / shape.n] * v[idx % shape.n])
        .collect();
    let b = (0..shape.n * shape.p)
        .into_par_iter()
        .map(|idx| s[idx / shape.p] * w[idx % shape.p])
        .collect();
    let c = (0..shape.m * shape.p)
        .into_par_iter()
        .map(|idx| gamma * u[idx / shape.p] * w[idx % shape.p])
        .collect();
    MatmulEvals { a, b, c }
}

#[derive(Debug)]
pub struct AppPrepared {
    pub a_y: Vec<BenchField>,
    pub b_y: Vec<BenchField>,
    pub c_eval: BenchField,
    pub rx: Vec<BenchField>,
    pub rz: Vec<BenchField>,
}

pub fn prepare_application(
    shape: &MatmulShape,
    a_evals: &[BenchField],
    b_evals: &[BenchField],
    c_evals: &[BenchField],
    mut rng: ChaCha8Rng,
) -> AppPrepared {
    assert_eq!(a_evals.len(), shape.m * shape.n);
    assert_eq!(b_evals.len(), shape.n * shape.p);
    assert_eq!(c_evals.len(), shape.m * shape.p);
    let rx = random_vec(shape.log_m, &mut rng);
    let rz = random_vec(shape.log_p, &mut rng);
    let wx = eq_weights(&rx);
    let wz = eq_weights(&rz);

    let (a_y, b_y) = rayon::join(
        || {
            (0..shape.n)
                .into_par_iter()
                .map(|j| {
                    (0..shape.m).fold(BenchField::ZERO, |acc, i| {
                        acc + wx[i] * a_evals[i * shape.n + j]
                    })
                })
                .collect::<Vec<_>>()
        },
        || {
            (0..shape.n)
                .into_par_iter()
                .map(|j| {
                    b_evals[j * shape.p..(j + 1) * shape.p]
                        .iter()
                        .zip(wz.iter())
                        .fold(BenchField::ZERO, |acc, (b, z)| acc + *b * *z)
                })
                .collect::<Vec<_>>()
        },
    );

    let c_eval = (0..shape.m)
        .into_par_iter()
        .map(|i| {
            let row_eval = c_evals[i * shape.p..(i + 1) * shape.p]
                .iter()
                .zip(wz.iter())
                .fold(BenchField::ZERO, |acc, (c, z)| acc + *c * *z);
            wx[i] * row_eval
        })
        .reduce(|| BenchField::ZERO, |a, b| a + b);

    let direct_rhs = a_y
        .iter()
        .zip(b_y.iter())
        .fold(BenchField::ZERO, |acc, (a, b)| acc + *a * *b);
    assert_eq!(direct_rhs, c_eval, "invalid matrix-multiplication witness");
    AppPrepared {
        a_y,
        b_y,
        c_eval,
        rx,
        rz,
    }
}

fn eq_weights(point: &[BenchField]) -> Vec<BenchField> {
    let mut weights = vec![BenchField::ONE];
    for r in point {
        let mut next = vec![BenchField::ZERO; 2 * weights.len()];
        for (i, old) in weights.iter().enumerate() {
            next[i] = *old * (BenchField::ONE - *r);
            next[i + weights.len()] = *old * *r;
        }
        weights = next;
    }
    weights
}

#[derive(Clone, Debug)]
pub struct ProductSumcheckProof {
    /// g_i(X) = c0 + c1 X + c2 X^2.
    pub rounds: Vec<[BenchField; 3]>,
}

#[derive(Debug)]
pub struct SumcheckProverOutput {
    pub proof: ProductSumcheckProof,
    pub ry: Vec<BenchField>,
    pub a_eval: BenchField,
    pub b_eval: BenchField,
}

pub fn prove_product_sumcheck(
    a: &[BenchField],
    b: &[BenchField],
    initial_claim: BenchField,
    mut rng: ChaCha8Rng,
) -> SumcheckProverOutput {
    assert_eq!(a.len(), b.len());
    assert!(a.len().is_power_of_two());
    let mut av = a.to_vec();
    let mut bv = b.to_vec();
    let mut current_claim = initial_claim;
    let mut rounds = Vec::with_capacity(a.len().trailing_zeros() as usize);
    let mut ry = Vec::with_capacity(rounds.capacity());
    let verifier_coin = BenchField::random(&mut rng);
    let mut transcript = SumcheckTranscript::new(());
    transcript.common_field_element(&verifier_coin).unwrap();

    while av.len() > 1 {
        let (mut c0, mut c1, mut c2) = (BenchField::ZERO, BenchField::ZERO, BenchField::ZERO);
        for pair in 0..av.len() / 2 {
            let (a0, a1) = (av[2 * pair], av[2 * pair + 1]);
            let (b0, b1) = (bv[2 * pair], bv[2 * pair + 1]);
            let (da, db) = (a1 - a0, b1 - b0);
            c0 += a0 * b0;
            c1 += a0 * db + b0 * da;
            c2 += da * db;
        }
        assert_eq!(
            c0 + (c0 + c1 + c2),
            current_claim,
            "sumcheck claim mismatch"
        );
        transcript.common_field_elements(&[c0, c1, c2]).unwrap();
        let r = transcript.squeeze_challenge();
        rounds.push([c0, c1, c2]);
        ry.push(r);
        av = av
            .chunks_exact(2)
            .map(|pair| pair[0] + (pair[1] - pair[0]) * r)
            .collect();
        bv = bv
            .chunks_exact(2)
            .map(|pair| pair[0] + (pair[1] - pair[0]) * r)
            .collect();
        current_claim = c0 + c1 * r + c2 * r * r;
    }
    assert_eq!(current_claim, av[0] * bv[0]);
    SumcheckProverOutput {
        proof: ProductSumcheckProof { rounds },
        ry,
        a_eval: av[0],
        b_eval: bv[0],
    }
}

pub fn verify_product_sumcheck(
    proof: &ProductSumcheckProof,
    initial_claim: BenchField,
    final_a: BenchField,
    final_b: BenchField,
    mut rng: ChaCha8Rng,
) -> Option<Vec<BenchField>> {
    let mut claim = initial_claim;
    let mut ry = Vec::with_capacity(proof.rounds.len());
    let verifier_coin = BenchField::random(&mut rng);
    let mut transcript = SumcheckTranscript::new(());
    transcript.common_field_element(&verifier_coin).ok()?;
    for &[c0, c1, c2] in &proof.rounds {
        if c0 + (c0 + c1 + c2) != claim {
            return None;
        }
        transcript.common_field_elements(&[c0, c1, c2]).ok()?;
        let r = transcript.squeeze_challenge();
        ry.push(r);
        claim = c0 + c1 * r + c2 * r * r;
    }
    (claim == final_a * final_b).then_some(ry)
}

pub fn application_points(
    app: &AppPrepared,
    ry: &[BenchField],
) -> (Vec<BenchField>, Vec<BenchField>, Vec<BenchField>) {
    // MLE variables are little-endian: A[x,y] -> [y,x], B[y,z] -> [z,y],
    // and C[x,z] -> [z,x].
    let mut point_a = Vec::with_capacity(ry.len() + app.rx.len());
    point_a.extend_from_slice(ry);
    point_a.extend_from_slice(&app.rx);
    let mut point_b = Vec::with_capacity(app.rz.len() + ry.len());
    point_b.extend_from_slice(&app.rz);
    point_b.extend_from_slice(ry);
    let mut point_c = Vec::with_capacity(app.rz.len() + app.rx.len());
    point_c.extend_from_slice(&app.rz);
    point_c.extend_from_slice(&app.rx);
    (point_a, point_b, point_c)
}

fn random_vec(len: usize, rng: &mut ChaCha8Rng) -> Vec<BenchField> {
    (0..len).map(|_| BenchField::random(&mut *rng)).collect()
}

pub fn seed32(base: u8, k: usize, sample: usize, domain: usize) -> [u8; 32] {
    let mut out = [base; 32];
    for (i, x) in k.to_le_bytes().iter().enumerate() {
        out[i % 32] ^= *x;
    }
    for (i, x) in sample.to_le_bytes().iter().enumerate() {
        out[(8 + i) % 32] ^= *x;
    }
    for (i, x) in domain.to_le_bytes().iter().enumerate() {
        out[(16 + i) % 32] ^= *x;
    }
    out
}

pub fn parse_exp_list(value: &str) -> Vec<usize> {
    let mut out = Vec::new();
    for part in value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
    {
        let range = part
            .split_once("..=")
            .or_else(|| part.split_once(".."))
            .or_else(|| part.split_once('-'));
        if let Some((a, b)) = range {
            let a = a.parse::<usize>().expect("invalid range start");
            let b = b.parse::<usize>().expect("invalid range end");
            assert!(a <= b);
            out.extend(a..=b);
        } else {
            out.push(part.parse::<usize>().expect("invalid exponent"));
        }
    }
    out.sort_unstable();
    out.dedup();
    assert!(!out.is_empty());
    out
}
