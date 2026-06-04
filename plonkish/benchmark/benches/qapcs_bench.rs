//! Benchmark for `MultilinearQAPCS`.
//!
//! Intended path:
//!
//!     benchmark/benches/qapcs_bench.rs
//!
//! This version matches the QABase benchmark output style:
//!
//! - writes CSV files under `./bench_data/qapcs`;
//! - creates the output directory automatically;
//! - appends one averaged row per `total_k`;
//! - for `samples >= 5`, averages only samples 2..samples, i.e. discards the
//!   first two warm-up samples, matching the QABase benchmark convention.
//!
//! Example:
//!
//!     RAYON_NUM_THREADS=32 cargo bench -p benchmark --bench qapcs_bench -- \
//!       --total-k 20..30 --samples 5 --threads 32
//!
//! If the benchmark target is not already configured with `harness = false`, add
//! the corresponding bench target to `benchmark/Cargo.toml`:
//!
//!     [[bench]]
//!     name = "qapcs_bench"
//!     harness = false

use plonkish_backend::{
    pcs::{
        multilinear::qapcs::{MultilinearQAPCS, QAPCSSpec, QAPCSSpecRateHalf100},
        PolynomialCommitmentScheme,
    },
    poly::{multilinear::MultilinearPolynomial, Polynomial},
    util::{
        arithmetic::{Field, PrimeField},
        hash::Blake2s,
        new_fields::Mersenne127,
        transcript::{Blake2sTranscript, InMemoryTranscript},
    },
};
use rand_chacha::{rand_core::SeedableRng, ChaCha8Rng};
use rayon::{current_num_threads, ThreadPoolBuilder};
use std::{
    env,
    fs::{create_dir_all, File, OpenOptions},
    io::{Cursor, Write},
    path::Path,
    time::{Duration, Instant},
};

type BenchField = Mersenne127;
type BenchHash = Blake2s;
type BenchSpec = QAPCSSpecRateHalf100;
type BenchPcs = MultilinearQAPCS<BenchField, BenchHash, BenchSpec>;
type BenchTranscript = Blake2sTranscript<Cursor<Vec<u8>>>;

const DEFAULT_TOTAL_K_START: usize = 20;
const DEFAULT_TOTAL_K_END: usize = 30;
const DEFAULT_SAMPLES: usize = 5;
const DEFAULT_THREADS: usize = 32;
const OUTPUT_DIR: &str = "./bench_data/qapcs";

fn main() {
    let args = Args::parse();

    env::set_var("RAYON_NUM_THREADS", args.threads.to_string());

    ensure_output_dir();
    let path = output_path(&args);
    write_header_if_new(&path);

    eprintln!("QAPCS benchmark config: {args:?}");
    eprintln!("writing csv to {path}");

    let pool = ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build()
        .expect("failed to build Rayon thread pool");

    pool.install(|| {
        eprintln!("rayon current_num_threads = {}", current_num_threads());

        for total_k in args.total_k_start..=args.total_k_end {
            let result = bench_one_total_k(total_k, &args);

            println!(
                "total_k={}, row_log={}, log_rows={}, rows={}, row_len={}, codeword_len={}, c={}, delta={:.8}, queries={}, proximity_reps={}, proof={} bytes, threads={}, commit={} ms, open={} ms, verify={} ms",
                result.total_k,
                result.row_log,
                result.log_rows,
                result.num_rows,
                result.row_len,
                result.codeword_len,
                result.inverse_rate,
                result.delta,
                result.queries,
                result.proximity_reps,
                result.proof_bytes,
                result.threads,
                result.commit_avg.as_millis(),
                result.open_avg.as_millis(),
                result.verify_avg.as_millis(),
            );

            append_result(&path, &result);
        }
    });

    eprintln!("wrote CSV results to {path}");
}

#[derive(Clone, Debug)]
struct BenchResult {
    field: &'static str,
    total_k: usize,
    row_log: usize,
    log_rows: usize,
    num_rows: usize,
    row_len: usize,
    codeword_len: usize,
    inverse_rate: usize,
    security_bits: usize,
    distance_failure_bits: usize,
    delta: f64,
    queries: usize,
    proximity_reps: usize,
    threads: usize,
    commit_avg: Duration,
    open_avg: Duration,
    verify_avg: Duration,
    proof_bytes: usize,
}

fn bench_one_total_k(total_k: usize, args: &Args) -> BenchResult {
    assert!(total_k < usize::BITS as usize, "total_k too large for usize");

    let poly_size = 1usize << total_k;

    // Use one setup/trim per total_k, then run several fresh samples under the
    // same shape. This matches how the averaged QABase benchmark is organized.
    let mut setup_rng = ChaCha8Rng::from_seed(seed_from(total_k, usize::MAX));
    let param = BenchPcs::setup(poly_size, 1, &mut setup_rng).expect("QAPCS setup failed");
    let (pp, vp) = BenchPcs::trim(&param, poly_size, 1).expect("QAPCS trim failed");

    let shape = pp.shape().clone();
    let log_rows = shape.num_rows.trailing_zeros() as usize;

    eprintln!(
        "QAPCS parameters: total=2^{}, row=2^{}, log_rows={}, rows={}, row_len={}, codeword_len={}, c={}, security={}, distance_failure={}, delta={:.8}, queries={}, proximity_reps={}, threads={}",
        total_k,
        shape.row_log_size,
        log_rows,
        shape.num_rows,
        shape.row_len,
        shape.codeword_len,
        <BenchSpec as QAPCSSpec>::inverse_rate(),
        <BenchSpec as QAPCSSpec>::security_bits(),
        <BenchSpec as QAPCSSpec>::distance_failure_bits(),
        shape.delta,
        shape.num_column_opening,
        shape.num_proximity_testing,
        current_num_threads(),
    );

    let mut commit_times = Vec::with_capacity(args.samples);
    let mut open_times = Vec::with_capacity(args.samples);
    let mut verify_times = Vec::with_capacity(args.samples);
    let mut last_proof_bytes = 0usize;

    for sample in 0..args.samples {
        let mut rng = ChaCha8Rng::from_seed(seed_from(total_k, sample));

        let evals = random_vec::<BenchField>(poly_size, &mut rng);
        let poly = MultilinearPolynomial::new(evals);

        let point = random_vec::<BenchField>(total_k, &mut rng);
        let eval = evaluate_mle_slow::<BenchField>(poly.evals(), &point);

        let now = Instant::now();
        let comm = BenchPcs::commit(&pp, &poly).expect("QAPCS commit failed");
        commit_times.push(now.elapsed());

        let mut prover_transcript = BenchTranscript::new(());
        let now = Instant::now();
        BenchPcs::open(&pp, &poly, &comm, &point, &eval, &mut prover_transcript)
            .expect("QAPCS open failed");
        open_times.push(now.elapsed());

        let proof = prover_transcript.into_proof();
        last_proof_bytes = proof.len();

        let mut verifier_transcript = BenchTranscript::from_proof((), proof.as_slice());
        let now = Instant::now();
        BenchPcs::verify(&vp, &comm, &point, &eval, &mut verifier_transcript)
            .expect("QAPCS verify failed");
        verify_times.push(now.elapsed());

        eprintln!(
            "sample {sample}: total_k={}, row_log={}, rows={}, queries={}, proof={} bytes, commit={} ms, open={} ms, verify={} ms",
            total_k,
            shape.row_log_size,
            shape.num_rows,
            shape.num_column_opening,
            last_proof_bytes,
            commit_times.last().unwrap().as_millis(),
            open_times.last().unwrap().as_millis(),
            verify_times.last().unwrap().as_millis(),
        );
    }

    BenchResult {
        field: "mersenne127",
        total_k,
        row_log: shape.row_log_size,
        log_rows,
        num_rows: shape.num_rows,
        row_len: shape.row_len,
        codeword_len: shape.codeword_len,
        inverse_rate: <BenchSpec as QAPCSSpec>::inverse_rate(),
        security_bits: <BenchSpec as QAPCSSpec>::security_bits(),
        distance_failure_bits: <BenchSpec as QAPCSSpec>::distance_failure_bits(),
        delta: shape.delta,
        queries: shape.num_column_opening,
        proximity_reps: shape.num_proximity_testing,
        threads: current_num_threads(),
        commit_avg: avg_after_warmup(&commit_times),
        open_avg: avg_after_warmup(&open_times),
        verify_avg: avg_after_warmup(&verify_times),
        proof_bytes: last_proof_bytes,
    }
}

fn random_vec<F>(len: usize, rng: &mut ChaCha8Rng) -> Vec<F>
where
    F: Field,
{
    (0..len).map(|_| F::random(&mut *rng)).collect()
}

/// Memory-light multilinear evaluation.
///
/// This avoids cloning the full evaluation vector, which matters for 2^30-scale
/// benchmarks. It uses the same little-endian Boolean-index convention as the
/// standard in-place folding algorithm:
///
///     eval(point) = sum_b evals[b] * prod_i eq(b_i, point_i).
///
/// It costs O(N log N) field operations and O(1) extra memory.
fn evaluate_mle_slow<F>(evals: &[F], point: &[F]) -> F
where
    F: PrimeField,
{
    assert!(evals.len().is_power_of_two(), "MLE eval length must be a power of two");
    assert_eq!(evals.len(), 1usize << point.len(), "point length mismatch");

    let mut acc = F::ZERO;
    for (idx, value) in evals.iter().enumerate() {
        let mut weight = F::ONE;
        for (bit, r) in point.iter().enumerate() {
            weight *= if ((idx >> bit) & 1) == 1 {
                *r
            } else {
                F::ONE - *r
            };
        }
        acc += *value * weight;
    }
    acc
}

fn avg_after_warmup(times: &[Duration]) -> Duration {
    assert!(!times.is_empty());

    // Same convention as qabase_bench: for 5 samples, report the average of the
    // last 3 samples; for fewer samples, average all available samples.
    let start = if times.len() >= 5 { 2 } else { 0 };

    let mut acc = Duration::new(0, 0);
    for t in &times[start..] {
        acc += *t;
    }

    acc / ((times.len() - start) as u32)
}

// -----------------------------------------------------------------------------
// CSV output
// -----------------------------------------------------------------------------

fn ensure_output_dir() {
    if !Path::new(OUTPUT_DIR).exists() {
        create_dir_all(OUTPUT_DIR).expect("failed to create benchmark output directory");
    }
}

fn output_path(args: &Args) -> String {
    format!(
        "{OUTPUT_DIR}/qapcs_mersenne127_ratehalf100_sec{}_df{}_th{}.csv",
        <BenchSpec as QAPCSSpec>::security_bits(),
        <BenchSpec as QAPCSSpec>::distance_failure_bits(),
        args.threads,
    )
}

fn write_header_if_new(path: &str) {
    if !Path::new(path).exists() {
        let mut f = File::create(path).expect("failed to create output csv");

        writeln!(
            &mut f,
            "field,total_k,row_log,log_rows,num_rows,row_len,codeword_len,inverse_rate,security_bits,distance_failure_bits,delta,queries,proximity_reps,threads,commit_ms,open_ms,verify_ms,proof_bytes,proof_kb"
        )
        .unwrap();
    }
}

fn append_result(path: &str, result: &BenchResult) {
    let mut f = OpenOptions::new()
        .append(true)
        .open(path)
        .expect("failed to open output csv");

    let proof_kb = result.proof_bytes as f64 / 1024.0;

    writeln!(
        &mut f,
        "{},{},{},{},{},{},{},{},{},{},{:.8},{},{},{},{},{},{},{},{:.2}",
        result.field,
        result.total_k,
        result.row_log,
        result.log_rows,
        result.num_rows,
        result.row_len,
        result.codeword_len,
        result.inverse_rate,
        result.security_bits,
        result.distance_failure_bits,
        result.delta,
        result.queries,
        result.proximity_reps,
        result.threads,
        result.commit_avg.as_millis(),
        result.open_avg.as_millis(),
        result.verify_avg.as_millis(),
        result.proof_bytes,
        proof_kb,
    )
    .unwrap();
}

#[derive(Clone, Debug)]
struct Args {
    total_k_start: usize,
    total_k_end: usize,
    samples: usize,
    threads: usize,
}

impl Args {
    fn parse() -> Self {
        let mut args = Self {
            total_k_start: DEFAULT_TOTAL_K_START,
            total_k_end: DEFAULT_TOTAL_K_END,
            samples: DEFAULT_SAMPLES,
            threads: DEFAULT_THREADS,
        };

        let mut iter = env::args().skip(1);
        while let Some(flag) = iter.next() {
            match flag.as_str() {
                "--total-k" => {
                    let value = iter.next().expect("--total-k needs a value, e.g. 20..30");
                    let (start, end) = parse_range_inclusive(&value);
                    args.total_k_start = start;
                    args.total_k_end = end;
                }
                "--samples" => {
                    let value = iter.next().expect("--samples needs a value");
                    args.samples = value.parse().expect("invalid --samples value");
                }
                "--threads" => {
                    let value = iter.next().expect("--threads needs a value");
                    args.threads = value.parse().expect("invalid --threads value");
                }
                "--bench" => {
                    // Defensive only: `--bench <name>` should normally be consumed by Cargo
                    // before the `--` separator, but ignore it if accidentally forwarded.
                    let _ = iter.next();
                }
                "--help" | "-h" => {
                    print_help_and_exit();
                }
                other => {
                    panic!("unknown argument: {other}");
                }
            }
        }

        assert!(args.total_k_start <= args.total_k_end, "invalid total-k range");
        assert!(args.samples >= 1, "samples must be positive");
        assert!(args.threads >= 1, "threads must be positive");

        args
    }
}

fn parse_range_inclusive(value: &str) -> (usize, usize) {
    if let Some((start, end)) = value.split_once("..=") {
        return (
            start.parse().expect("invalid range start"),
            end.parse().expect("invalid range end"),
        );
    }
    if let Some((start, end)) = value.split_once("..") {
        return (
            start.parse().expect("invalid range start"),
            end.parse().expect("invalid range end"),
        );
    }
    let single = value.parse().expect("invalid total-k value");
    (single, single)
}

fn seed_from(total_k: usize, sample: usize) -> [u8; 32] {
    let mut seed = [0u8; 32];
    seed[0..8].copy_from_slice(&(total_k as u64).to_le_bytes());
    seed[8..16].copy_from_slice(&(sample as u64).to_le_bytes());
    seed[16..24].copy_from_slice(&0x5141_5043_535f_4245u64.to_le_bytes()); // "QAPCS_BE"
    seed[24..32].copy_from_slice(&0x4e43_484d_4152_4b21u64.to_le_bytes()); // "NCHMARK!"
    seed
}

fn print_help_and_exit() -> ! {
    println!(
        "Usage: qapcs_bench [--total-k 20..30] [--samples 5] [--threads 32]\n\
         \n\
         Writes one averaged CSV row per total_k to ./bench_data/qapcs/.\n\
         For samples >= 5, the first two samples are discarded and the remaining samples are averaged.\n\
         Proof size is the opening transcript size in bytes."
    );
    std::process::exit(0)
}
