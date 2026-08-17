use plonkish_backend::{
    pcs::multilinear::quasar::{qa_encode_codeword_only, wht, QAParams},
    util::arithmetic::{Field, PrimeField},
};
use rand_chacha::{rand_core::RngCore, ChaCha8Rng};
use rayon::prelude::*;
use std::{
    hint::black_box,
    time::{Duration, Instant},
};

#[derive(Clone, Debug, Default)]
pub struct CpuQaTiming {
    pub allocation: Duration,
    pub first_wht: Duration,
    pub scaling_multiplications: Duration,
    pub second_wht: Duration,
    pub systematic_copy: Duration,
    pub total: Duration,
}

impl CpuQaTiming {
    pub fn wht_total(&self) -> Duration {
        self.first_wht + self.second_wht
    }

    pub fn measured_compute(&self) -> Duration {
        self.wht_total() + self.scaling_multiplications
    }
}

#[derive(Clone, Debug)]
pub struct FieldOpTiming {
    pub elements: usize,
    pub repetitions: usize,
    pub additions: Duration,
    pub subtractions: Duration,
    pub multiplications: Duration,
    pub butterflies: Duration,
}

impl FieldOpTiming {
    fn operations(&self) -> f64 {
        (self.elements * self.repetitions) as f64
    }

    pub fn add_ns_per_op(&self) -> f64 {
        self.additions.as_secs_f64() * 1e9 / self.operations()
    }

    pub fn sub_ns_per_op(&self) -> f64 {
        self.subtractions.as_secs_f64() * 1e9 / self.operations()
    }

    pub fn mul_ns_per_op(&self) -> f64 {
        self.multiplications.as_secs_f64() * 1e9 / self.operations()
    }

    /// Each loop iteration computes one addition and one subtraction.
    pub fn butterfly_ns(&self) -> f64 {
        self.butterflies.as_secs_f64() * 1e9 / self.operations()
    }
}

/// The current commitment encoder, parallelized over rows exactly as in the
/// QAPCS/Quasar commitment path. This is the baseline used for the total CPU
/// time; `qa_encode_cpu_profiled_rows` below adds barriers between stages so
/// that the WHT and multiplication wall times can be measured separately.
pub fn qa_encode_cpu_baseline_rows<F>(
    messages: &[F],
    row_len: usize,
    params: &QAParams<F>,
) -> Vec<F>
where
    F: PrimeField + Send + Sync,
{
    assert_eq!(messages.len() % row_len, 0);
    messages
        .par_chunks(row_len)
        .map(|row| qa_encode_codeword_only(row, params))
        .collect::<Vec<_>>()
        .into_iter()
        .flatten()
        .collect()
}

/// Serial QA encoding of one message into one codeword.
pub fn qa_encode_cpu_serial_single_row<F>(message: &[F], params: &QAParams<F>) -> Vec<F>
where
    F: PrimeField,
{
    qa_encode_codeword_only(message, params)
}

/// In-place Walsh--Hadamard transform with parallelism inside one row.
///
/// Every Rayon task owns a disjoint `2 * half`-element butterfly block.  The
/// `for_each` completes before the next stage starts, providing the barrier
/// required by the WHT data dependencies.
pub fn wht_parallel_single_row<F>(values: &mut [F])
where
    F: PrimeField + Send + Sync,
{
    assert!(values.len().is_power_of_two());
    let workers = rayon::current_num_threads();
    let mut half = 1;
    while half < values.len() {
        let block_len = 2 * half;
        let blocks = values.len() / block_len;
        if blocks >= 4 * workers {
            values.par_chunks_mut(block_len).for_each(|block| {
                let (left, right) = block.split_at_mut(half);
                for (a, b) in left.iter_mut().zip(right.iter_mut()) {
                    let x = *a;
                    let y = *b;
                    *a = x + y;
                    *b = x - y;
                }
            });
        } else if half >= 1024 {
            values.chunks_mut(block_len).for_each(|block| {
                let (left, right) = block.split_at_mut(half);
                left.par_iter_mut()
                    .zip(right.par_iter_mut())
                    .for_each(|(a, b)| {
                        let x = *a;
                        let y = *b;
                        *a = x + y;
                        *b = x - y;
                    });
            });
        } else {
            values.chunks_mut(block_len).for_each(|block| {
                let (left, right) = block.split_at_mut(half);
                for (a, b) in left.iter_mut().zip(right.iter_mut()) {
                    let x = *a;
                    let y = *b;
                    *a = x + y;
                    *b = x - y;
                }
            });
        }
        half *= 2;
    }
}

/// QA encoding of exactly one row, parallelized within that codeword.
///
/// This preserves the same systematic output layout as
/// `qa_encode_codeword_only`; unlike row-level parallelism, it remains useful
/// when the benchmark contains one message and one QA codeword.
pub fn qa_encode_cpu_parallel_single_row<F>(message: &[F], params: &QAParams<F>) -> Vec<F>
where
    F: PrimeField + Send + Sync,
{
    let row_len = message.len();
    assert!(row_len.is_power_of_two());
    assert_eq!(params.e.len() + 1, params.inverse_rate);
    assert!(params
        .e
        .iter()
        .all(|coefficients| coefficients.len() == row_len));

    let mut transformed = message.to_vec();
    wht_parallel_single_row(&mut transformed);

    let mut codeword = vec![F::ZERO; params.inverse_rate * row_len];
    codeword[row_len..]
        .par_chunks_mut(row_len)
        .zip(params.e.par_iter())
        .for_each(|(block, coefficients)| {
            block
                .par_iter_mut()
                .zip(transformed.par_iter())
                .zip(coefficients.par_iter())
                .for_each(|((out, value), coefficient)| *out = *value * *coefficient);
            wht_parallel_single_row(block);
        });
    codeword[..row_len].copy_from_slice(message);
    codeword
}

/// Stage-separated CPU encoder for the existing QA code.
///
/// Output layout is row-major:
/// `row_0(message || parity_0 || ...) || row_1(...) || ...`.
/// The function uses the same serial per-row WHT and Rayon row parallelism as
/// the current commitment encoder. The only deliberate change is a barrier
/// between stages, which makes the wall-clock split meaningful.
pub fn qa_encode_cpu_profiled_rows<F>(
    messages: &[F],
    row_len: usize,
    params: &QAParams<F>,
) -> (Vec<F>, CpuQaTiming)
where
    F: PrimeField + Send + Sync,
{
    assert!(row_len.is_power_of_two());
    assert_eq!(messages.len() % row_len, 0);
    assert_eq!(params.e.len() + 1, params.inverse_rate);
    assert!(params.e.iter().all(|e| e.len() == row_len));

    let rows = messages.len() / row_len;
    let c = params.inverse_rate;
    let total_start = Instant::now();

    let start = Instant::now();
    let mut middle = messages.to_vec();
    let mut codewords = vec![F::ZERO; rows * c * row_len];
    let allocation = start.elapsed();

    let start = Instant::now();
    middle.par_chunks_mut(row_len).for_each(wht::<F>);
    let first_wht = start.elapsed();

    let start = Instant::now();
    codewords
        .par_chunks_mut(c * row_len)
        .zip(middle.par_chunks(row_len))
        .for_each(|(encoded_row, transformed_row)| {
            for (parity_index, coefficients) in params.e.iter().enumerate() {
                let begin = (parity_index + 1) * row_len;
                let block = &mut encoded_row[begin..begin + row_len];
                for ((out, value), coefficient) in block
                    .iter_mut()
                    .zip(transformed_row.iter())
                    .zip(coefficients.iter())
                {
                    *out = *value * *coefficient;
                }
            }
        });
    let scaling_multiplications = start.elapsed();

    let start = Instant::now();
    codewords
        .par_chunks_mut(c * row_len)
        .for_each(|encoded_row| {
            for parity_index in 0..(c - 1) {
                let begin = (parity_index + 1) * row_len;
                wht(&mut encoded_row[begin..begin + row_len]);
            }
        });
    let second_wht = start.elapsed();

    let start = Instant::now();
    codewords
        .par_chunks_mut(c * row_len)
        .zip(messages.par_chunks(row_len))
        .for_each(|(encoded_row, message)| encoded_row[..row_len].copy_from_slice(message));
    let systematic_copy = start.elapsed();

    let timing = CpuQaTiming {
        allocation,
        first_wht,
        scaling_multiplications,
        second_wht,
        systematic_copy,
        total: total_start.elapsed(),
    };
    (codewords, timing)
}

/// Memory-throughput-aware microbenchmark for the concrete field operations.
/// It measures the same vector-shaped operations that appear in WHT and
/// scaling, rather than a latency-only dependency chain.
pub fn benchmark_field_operations<F>(
    rng: &mut ChaCha8Rng,
    elements: usize,
    repetitions: usize,
) -> FieldOpTiming
where
    F: PrimeField,
{
    assert!(elements > 0 && repetitions > 0);
    let a = (0..elements)
        .map(|_| F::random(&mut *rng))
        .collect::<Vec<_>>();
    let b = (0..elements)
        .map(|_| F::random(&mut *rng))
        .collect::<Vec<_>>();
    let mut out = vec![F::ZERO; elements];

    let start = Instant::now();
    for _ in 0..repetitions {
        for i in 0..elements {
            out[i] = black_box(a[i] + b[i]);
        }
        black_box(&out);
    }
    let additions = start.elapsed();

    let start = Instant::now();
    for _ in 0..repetitions {
        for i in 0..elements {
            out[i] = black_box(a[i] - b[i]);
        }
        black_box(&out);
    }
    let subtractions = start.elapsed();

    let start = Instant::now();
    for _ in 0..repetitions {
        for i in 0..elements {
            out[i] = black_box(a[i] * b[i]);
        }
        black_box(&out);
    }
    let multiplications = start.elapsed();

    let start = Instant::now();
    for _ in 0..repetitions {
        for i in 0..elements {
            let x = a[i];
            let y = b[i];
            out[i] = black_box(x + y);
            black_box(x - y);
        }
        black_box(&out);
    }
    let butterflies = start.elapsed();

    FieldOpTiming {
        elements,
        repetitions,
        additions,
        subtractions,
        multiplications,
        butterflies,
    }
}
