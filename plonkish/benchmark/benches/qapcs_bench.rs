//! Full-opening benchmark for `MultilinearQAPCS`.
//!
//! Intended path:
//!
//!     benchmark/benches/qapcs_bench.rs
//!
//! Assumptions:
//!
//! - `qapcs.rs` and `quasar.rs` are both exported from
//!   `plonkish_backend::pcs::multilinear`;
//! - QAPCS exposes `qapcs_open_full` and `qapcs_verify_full`;
//! - hash commitments are absorbed into the Fiat--Shamir transcript.
//!
//! This benchmark:
//!
//! 1. measures setup and trim separately;
//! 2. prepares the claimed MLE evaluation in O(N) time using the same
//!    row-major decomposition as QAPCS;
//! 3. measures commitment samples before opening samples;
//! 4. writes the QAPCS root into each proof transcript before sampling opening
//!    challenges, and verifies that root before running the verifier;
//! 5. uses the median commitment time and the post-warm-up mean for
//!    open/verify.

use plonkish_backend::{
    pcs::{
        multilinear::qapcs::{
            qapcs_open_full, qapcs_verify_full, MultilinearQAPCS, QAPCSSpec, QAPCSSpecRateHalf100,
        },
        PolynomialCommitmentScheme,
    },
    poly::{multilinear::MultilinearPolynomial, Polynomial},
    util::{
        arithmetic::{inner_product, Field, PrimeField},
        hash::{Blake2s, Output},
        new_fields::Mersenne127,
        transcript::{Blake2sTranscript, InMemoryTranscript, TranscriptRead, TranscriptWrite},
    },
};
use rand_chacha::{rand_core::SeedableRng, ChaCha8Rng};
use rayon::{current_num_threads, prelude::*, ThreadPoolBuilder};
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
type BenchCommitmentChunk = Output<BenchHash>;

const DEFAULT_TOTAL_K_START: usize = 20;
const DEFAULT_TOTAL_K_END: usize = 28;
const DEFAULT_SAMPLES: usize = 5;
const DEFAULT_THREADS: usize = 32;
const OUTPUT_DIR: &str = "./bench_data/qapcs";

fn main() {
    let args = Args::parse();

    // Keep libraries that inspect this environment variable consistent with the
    // dedicated pool used below.
    env::set_var("RAYON_NUM_THREADS", args.threads.to_string());

    ensure_output_dir();
    let path = output_path(&args);
    write_header_if_new(&path);

    eprintln!("QAPCS full-opening benchmark config: {args:?}");
    eprintln!("writing CSV to {path}");

    let pool = ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build()
        .expect("failed to build Rayon thread pool");

    pool.install(|| {
        eprintln!("rayon current_num_threads = {}", current_num_threads());

        for total_k in args.total_k_start..=args.total_k_end {
            let result = bench_one_total_k(total_k, &args);

            println!(
                concat!(
                    "total_k={}, row_log={}, log_rows={}, rows={}, row_len={}, ",
                    "codeword_len={}, c={}, delta={:.8}, queries={}, proximity_reps={}, ",
                    "proof={} bytes, threads={}, setup={:.3} ms, trim={:.3} ms, ",
                    "eval_prepare={:.3} ms, commit={:.3} ms, open={:.3} ms, verify={:.3} ms"
                ),
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
                duration_ms(result.setup_time),
                duration_ms(result.trim_time),
                duration_ms(result.eval_prepare_time),
                duration_ms(result.commit_time),
                duration_ms(result.open_time),
                duration_ms(result.verify_time),
            );

            append_result(&path, &result);
        }
    });

    eprintln!("wrote CSV results to {path}");
}

#[derive(Clone, Debug)]
struct BenchResult {
    field: &'static str,
    protocol: &'static str,
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
    setup_time: Duration,
    trim_time: Duration,
    eval_prepare_time: Duration,
    commit_time: Duration,
    open_time: Duration,
    verify_time: Duration,
    proof_bytes: usize,
}

fn bench_one_total_k(total_k: usize, args: &Args) -> BenchResult {
    assert!(
        total_k < usize::BITS as usize,
        "total_k is too large for usize"
    );

    let poly_size = 1usize << total_k;

    let mut setup_rng = ChaCha8Rng::from_seed(seed_from(total_k, usize::MAX));

    let setup_start = Instant::now();
    let param = BenchPcs::setup(poly_size, 1, &mut setup_rng).expect("QAPCS setup failed");
    let setup_time = setup_start.elapsed();

    let trim_start = Instant::now();
    let (pp, vp) = BenchPcs::trim(&param, poly_size, 1).expect("QAPCS trim failed");
    let trim_time = trim_start.elapsed();

    let shape = pp.shape().clone();
    let log_rows = shape.num_rows.trailing_zeros() as usize;

    eprintln!(
        concat!(
            "QAPCS parameters: total=2^{}, row=2^{}, log_rows={}, rows={}, ",
            "row_len={}, codeword_len={}, c={}, security={}, distance_failure={}, ",
            "delta={:.8}, queries={}, proximity_reps={}, threads={}"
        ),
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

    // Generate one fixed benchmark instance. Reusing it removes input-generation
    // noise from protocol timings.
    let mut rng = ChaCha8Rng::from_seed(seed_from(total_k, 0));
    let evals = random_vec::<BenchField>(poly_size, &mut rng);
    let poly = MultilinearPolynomial::new(evals);
    let point = random_vec::<BenchField>(total_k, &mut rng);

    let eval_prepare_start = Instant::now();
    let eval = prepare_row_major_mle_evaluation::<BenchField>(
        poly.evals(),
        &point,
        shape.num_rows,
        shape.row_len,
    );
    let eval_prepare_time = eval_prepare_start.elapsed();

    // Commitment-only measurements. No opening work is interleaved.
    let mut commit_times = Vec::with_capacity(args.samples);
    for sample in 0..args.samples {
        let start = Instant::now();
        let comm = BenchPcs::commit(&pp, &poly).expect("QAPCS commitment failed");
        let elapsed = start.elapsed();
        commit_times.push(elapsed);

        eprintln!(
            concat!(
                "commit-only sample {}: total_k={}, row_log={}, rows={}, ",
                "commit={:.3} ms"
            ),
            sample,
            total_k,
            shape.row_log_size,
            shape.num_rows,
            duration_ms(elapsed),
        );

        drop(comm);
    }

    // One reusable commitment for all opening/verification samples.
    let comm = BenchPcs::commit(&pp, &poly).expect("reusable QAPCS commitment failed");

    let mut open_times = Vec::with_capacity(args.samples);
    let mut verify_times = Vec::with_capacity(args.samples);
    let mut last_proof_bytes = 0usize;

    for sample in 0..args.samples {
        let mut prover_transcript = BenchTranscript::new(());

        // Bind all subsequent Fiat--Shamir challenges to the main commitment.
        <BenchTranscript as TranscriptWrite<BenchCommitmentChunk, BenchField>>::write_commitment(
            &mut prover_transcript,
            comm.root(),
        )
        .expect("failed to write QAPCS commitment root");

        let open_start = Instant::now();
        qapcs_open_full::<BenchField, BenchHash>(
            &pp,
            &poly,
            &comm,
            &point,
            &eval,
            &mut prover_transcript,
        )
        .expect("QAPCS full opening failed");
        let open_elapsed = open_start.elapsed();
        open_times.push(open_elapsed);

        let proof = prover_transcript.into_proof();
        last_proof_bytes = proof.len();

        let mut verifier_transcript = BenchTranscript::from_proof((), proof.as_slice());

        // Match the Quasar benchmark: verifier time includes reading and
        // checking the public commitment root.
        let verify_start = Instant::now();

        // Read and bind the same root before replaying verifier challenges.
        let root_from_proof = <BenchTranscript as TranscriptRead<
            BenchCommitmentChunk,
            BenchField,
        >>::read_commitment(&mut verifier_transcript)
        .expect("failed to read QAPCS commitment root");

        assert_eq!(
            &root_from_proof,
            comm.root(),
            "QAPCS commitment root mismatch"
        );

        qapcs_verify_full::<BenchField, BenchHash>(
            &vp,
            &comm,
            &point,
            &eval,
            &mut verifier_transcript,
        )
        .expect("QAPCS full verification failed");
        let verify_elapsed = verify_start.elapsed();
        verify_times.push(verify_elapsed);

        eprintln!(
            concat!(
                "open/verify sample {}: total_k={}, row_log={}, rows={}, queries={}, ",
                "proximity_reps={}, proof={} bytes, open={:.3} ms, verify={:.3} ms"
            ),
            sample,
            total_k,
            shape.row_log_size,
            shape.num_rows,
            shape.num_column_opening,
            shape.num_proximity_testing,
            last_proof_bytes,
            duration_ms(open_elapsed),
            duration_ms(verify_elapsed),
        );
    }

    BenchResult {
        field: "mersenne127",
        protocol: "qapcs_full_opening",
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
        setup_time,
        trim_time,
        eval_prepare_time,
        commit_time: median_duration(&commit_times),
        open_time: avg_after_warmup(&open_times),
        verify_time: avg_after_warmup(&verify_times),
        proof_bytes: last_proof_bytes,
    }
}

fn random_vec<F>(len: usize, rng: &mut ChaCha8Rng) -> Vec<F>
where
    F: Field,
{
    (0..len).map(|_| F::random(&mut *rng)).collect()
}

/// Evaluate a row-major MLE in O(N) field operations.
///
/// For:
///
///     flat[row * row_len + column],
///
/// the first MLE coordinates index columns and the final log2(num_rows)
/// coordinates index rows.
fn prepare_row_major_mle_evaluation<F>(
    evals: &[F],
    point: &[F],
    num_rows: usize,
    row_len: usize,
) -> F
where
    F: PrimeField + Send + Sync,
{
    assert!(num_rows.is_power_of_two());
    assert!(row_len.is_power_of_two());
    assert_eq!(evals.len(), num_rows * row_len);

    let log_rows = num_rows.trailing_zeros() as usize;
    let log_columns = row_len.trailing_zeros() as usize;
    assert_eq!(point.len(), log_columns + log_rows);

    let (column_point, row_point) = point.split_at(log_columns);
    let row_weights = MultilinearPolynomial::<F>::eq_xy(row_point).into_evals();
    let column_weights = MultilinearPolynomial::<F>::eq_xy(column_point).into_evals();

    assert_eq!(row_weights.len(), num_rows);
    assert_eq!(column_weights.len(), row_len);

    let folded_row = (0..row_len)
        .into_par_iter()
        .map(|column| {
            let mut acc = F::ZERO;
            for row in 0..num_rows {
                acc += row_weights[row] * evals[row * row_len + column];
            }
            acc
        })
        .collect::<Vec<_>>();

    inner_product(&folded_row, &column_weights)
}

fn avg_after_warmup(times: &[Duration]) -> Duration {
    assert!(!times.is_empty());

    let start = if times.len() >= 5 { 2 } else { 0 };
    let mut acc = Duration::ZERO;
    for time in &times[start..] {
        acc += *time;
    }
    acc / ((times.len() - start) as u32)
}

fn median_duration(times: &[Duration]) -> Duration {
    assert!(!times.is_empty());

    let mut sorted = times.to_vec();
    sorted.sort_unstable();

    let middle = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        sorted[middle]
    } else {
        (sorted[middle - 1] + sorted[middle]) / 2
    }
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
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
        "{}/qapcs_full_mersenne127_rho{}_sec{}_df{}_th{}.csv",
        OUTPUT_DIR,
        <BenchSpec as QAPCSSpec>::inverse_rate(),
        <BenchSpec as QAPCSSpec>::security_bits(),
        <BenchSpec as QAPCSSpec>::distance_failure_bits(),
        args.threads,
    )
}

fn write_header_if_new(path: &str) {
    if !Path::new(path).exists() {
        let mut file = File::create(path).expect("failed to create output CSV");

        writeln!(
            &mut file,
            concat!(
                "field,protocol,total_k,row_log,log_rows,num_rows,row_len,",
                "codeword_len,inverse_rate,security_bits,distance_failure_bits,",
                "delta,queries,proximity_reps,proof_bytes,proof_kb,threads,",
                "setup_ms,trim_ms,eval_prepare_ms,commit_ms,open_ms,verify_ms"
            )
        )
        .unwrap();
    }
}

fn append_result(path: &str, result: &BenchResult) {
    let mut file = OpenOptions::new()
        .append(true)
        .open(path)
        .expect("failed to open output CSV");

    let proof_kb = result.proof_bytes as f64 / 1024.0;

    writeln!(
        &mut file,
        concat!(
            "{},{},{},{},{},{},{},{},{},{},{},{:.8},{},{},{},{:.2},{},",
            "{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}"
        ),
        result.field,
        result.protocol,
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
        result.proof_bytes,
        proof_kb,
        result.threads,
        duration_ms(result.setup_time),
        duration_ms(result.trim_time),
        duration_ms(result.eval_prepare_time),
        duration_ms(result.commit_time),
        duration_ms(result.open_time),
        duration_ms(result.verify_time),
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
                    let value = iter.next().expect("--total-k needs a value, e.g. 20..=28");
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
                    // Defensive only. Cargo normally consumes this.
                    let _ = iter.next();
                }
                "--smoke" => {
                    args.total_k_start = 12;
                    args.total_k_end = 12;
                    args.samples = 1;
                    args.threads = args.threads.min(8);
                }
                "--help" | "-h" => print_help_and_exit(),
                other => panic!("unknown argument: {other}"),
            }
        }

        assert!(
            args.total_k_start <= args.total_k_end,
            "invalid total-k range"
        );
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
    seed[16..24].copy_from_slice(&0x5141_5043_535f_4245u64.to_le_bytes());
    seed[24..32].copy_from_slice(&0x4e43_484d_4152_4b21u64.to_le_bytes());
    seed
}

fn print_help_and_exit() -> ! {
    println!(
        "QAPCS full-opening benchmark\n\n\
         Smoke test:\n\
           cargo bench -p benchmark --bench qapcs_bench -- --smoke --threads 8\n\n\
         Paper-scale test:\n\
           cargo bench -p benchmark --bench qapcs_bench -- \\\n             --total-k 20..=28 --samples 5 --threads 32\n\n\
         Options:\n\
           --total-k <k|a..b|a..=b>\n\
           --samples <n>\n\
           --threads <n>\n\
           --smoke\n"
    );
    std::process::exit(0)
}
