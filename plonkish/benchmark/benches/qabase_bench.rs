use plonkish_backend::{
    pcs::multilinear::qabase::{
        commit_and_write, prove_qabase_open_full_global_batch_batched_wht,
        prove_qabase_open_scaffold_global_batch_batched_wht, setup, trim,
        verify_qabase_open_full_global_batch_batched_wht,
        verify_qabase_open_scaffold_global_batch_batched_wht,
    },
    util::{
        arithmetic::Field,
        hash::Blake2s,
        new_fields::Mersenne127,
        transcript::{Blake2sTranscript, InMemoryTranscript},
    },
};

use rand_chacha::{rand_core::SeedableRng, ChaCha8Rng};
use rayon::{current_num_threads, ThreadPoolBuilder};

use std::{
    env::args,
    fs::{create_dir_all, File, OpenOptions},
    io::{Cursor, Write},
    path::Path,
    time::{Duration, Instant},
};

type TestTranscript = Blake2sTranscript<Cursor<Vec<u8>>>;

type BenchField = Mersenne127;

const OUTPUT_DIR: &str = "./bench_data/qabase";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OpenMode {
    /// Current benchmark path: run only the existing proximity scaffold.
    /// This is used as the DP24-style merged timing path.
    Merge,

    /// Complete protocol path: run proximity branch plus evaluation branch.
    Full,
}

impl OpenMode {
    fn as_str(self) -> &'static str {
        match self {
            OpenMode::Merge => "merge",
            OpenMode::Full => "full",
        }
    }

    fn parse(value: &str) -> Self {
        match value {
            "merge" | "Merge" | "MERGE" => OpenMode::Merge,
            "full" | "Full" | "FULL" => OpenMode::Full,
            other => panic!("invalid --open-mode: {other}; expected merge or full"),
        }
    }
}

#[derive(Clone, Debug)]
struct BenchConfig {
    /// Total input-size exponents. The range is inclusive.
    total_k_values: Vec<usize>,

    /// Which opening path to benchmark.
    ///
    /// Merge keeps the old timing path and runs only the proximity scaffold.
    /// Full runs proximity plus the evaluation branch q = QAEnc(eq(z_L, .)^T M).
    open_mode: OpenMode,

    /// Default row exponent used outside the special small-size schedule.
    /// The committed matrix has 2^log_rows rows.
    log_rows: usize,

    /// If true, use the small-size schedule:
    ///   total_k = 20,21 -> 16 rows  (log_rows = 4)
    ///   total_k = 22,23 -> 32 rows  (log_rows = 5)
    ///   otherwise       -> 2^log_rows rows.
    ///
    /// Default is false, so every total_k uses 2^log_rows rows.
    small_row_schedule: bool,

    /// QA inverse rate c. The code rate is 1/c.
    inverse_rate: usize,

    /// If true, compute Merkle query count from the QA distance bound.
    auto_queries: bool,

    /// Manual Merkle query count. Used only if auto_queries = false.
    queries: usize,

    /// Field size in bits used for parameter calculation.
    field_bits: usize,

    /// Target query soundness in bits.
    security_bits: usize,

    /// Target failure probability for the random QA-code distance bound.
    distance_failure_bits: usize,

    /// Number of benchmark samples per total_k.
    samples: usize,

    /// Number of Rayon worker threads.
    threads: usize,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            total_k_values: (20..=30).collect(),
            open_mode: OpenMode::Merge,
            log_rows: 6,
            small_row_schedule: false,
            inverse_rate: 2,
            auto_queries: true,
            queries: 0,
            field_bits: 127,
            security_bits: 100,
            distance_failure_bits: 100,
            samples: 5,
            threads: 32,
        }
    }
}

impl BenchConfig {
    fn log_rows_for_total_k(&self, total_k: usize) -> usize {
        if !self.small_row_schedule {
            return self.log_rows;
        }

        match total_k {
            20 | 21 => 4, // 16 rows
            22 | 23 => 5, // 32 rows
            _ => self.log_rows,
        }
    }
}

#[derive(Clone, Debug)]
struct SecurityChoice {
    total_k: usize,
    row_k: usize,
    log_rows: usize,
    num_rows: usize,
    inverse_rate: usize,
    field_bits: usize,
    security_bits: usize,
    distance_failure_bits: usize,
    delta: f64,
    queries: usize,
}

impl SecurityChoice {
    fn new(total_k: usize, cfg: &BenchConfig) -> Self {
        let log_rows = cfg.log_rows_for_total_k(total_k);

        assert!(
            total_k >= log_rows,
            "total_k must be at least log_rows"
        );

        let row_k = total_k - log_rows;

        let delta = qabase_distance_lower_bound(
            row_k,
            cfg.inverse_rate,
            cfg.field_bits,
            cfg.distance_failure_bits,
        );

        let auto_queries = qabase_queries_from_distance(delta, cfg.security_bits);

        let queries = if cfg.auto_queries {
            auto_queries
        } else {
            cfg.queries
        };

        Self {
            total_k,
            row_k,
            log_rows,
            num_rows: 1usize << log_rows,
            inverse_rate: cfg.inverse_rate,
            field_bits: cfg.field_bits,
            security_bits: cfg.security_bits,
            distance_failure_bits: cfg.distance_failure_bits,
            delta,
            queries,
        }
    }

    fn row_size(&self) -> usize {
        1usize << self.row_k
    }
}

#[derive(Clone, Debug)]
struct BenchResult {
    field: &'static str,
    open_mode: OpenMode,
    total_k: usize,
    row_k: usize,
    log_rows: usize,
    num_rows: usize,
    inverse_rate: usize,
    field_bits: usize,
    security_bits: usize,
    distance_failure_bits: usize,
    delta: f64,
    queries: usize,
    proof_bytes: usize,
    threads: usize,
    commit_avg: Duration,
    prove_avg: Duration,
    verify_avg: Duration,
}

fn main() {
    let cfg = parse_args();

    ensure_output_dir();

    let path = output_path(&cfg);
    write_header_if_new(&path);

    println!("QABase full benchmark config: {cfg:?}");
    println!("writing csv to {path}");

    let pool = ThreadPoolBuilder::new()
        .num_threads(cfg.threads)
        .build()
        .expect("failed to build Rayon thread pool");

    pool.install(|| {
        println!("rayon current_num_threads = {}", current_num_threads());

        for total_k in cfg.total_k_values.clone() {
            let result = bench_one_total_k(total_k, &cfg);

            println!(
                "mode={}, total_k={}, row_k={}, log_rows={}, rows={}, c={}, delta={:.8}, queries={}, proof={} bytes, threads={}, commit={} ms, prove={} ms, verify={} ms",
                result.open_mode.as_str(),
                result.total_k,
                result.row_k,
                result.log_rows,
                result.num_rows,
                result.inverse_rate,
                result.delta,
                result.queries,
                result.proof_bytes,
                result.threads,
                result.commit_avg.as_millis(),
                result.prove_avg.as_millis(),
                result.verify_avg.as_millis(),
            );

            append_result(&path, &result);
        }
    });
}

// -----------------------------------------------------------------------------
// Argument parsing
// -----------------------------------------------------------------------------

fn parse_exp_list(value: &str) -> Vec<usize> {
    if let Some((start, end)) = value.split_once("..=") {
        let start = start.parse::<usize>().expect("invalid range start");
        let end = end.parse::<usize>().expect("invalid range end");
        assert!(start <= end, "range start must be <= end");
        (start..=end).collect()
    } else if let Some((start, end)) = value.split_once("..") {
        let start = start.parse::<usize>().expect("invalid range start");
        let end = end.parse::<usize>().expect("invalid range end");
        assert!(start <= end, "range start must be <= end");
        (start..=end).collect()
    } else {
        vec![value.parse::<usize>().expect("invalid exponent")]
    }
}

fn parse_args() -> BenchConfig {
    let mut cfg = BenchConfig::default();

    let argv = args().collect::<Vec<_>>();
    let mut i = 1;

    while i < argv.len() {
        match argv[i].as_str() {
            // Defensive: in normal cargo usage, --bench is consumed by cargo.
            "--bench" => {
                if i + 1 < argv.len() && !argv[i + 1].starts_with("--") {
                    i += 1;
                }
            }

            "--total-k" => {
                i += 1;
                cfg.total_k_values = parse_exp_list(&argv[i]);
            }

            "--k" => {
                i += 1;
                cfg.total_k_values = parse_exp_list(&argv[i]);
            }

            "--open-mode" => {
                i += 1;
                cfg.open_mode = OpenMode::parse(&argv[i]);
            }

            "--log-rows" => {
                i += 1;
                cfg.log_rows = argv[i].parse().expect("invalid --log-rows");
            }

            "--small-row-schedule" => {
                cfg.small_row_schedule = true;
            }

            "--fixed-log-rows" => {
                cfg.small_row_schedule = false;
            }

            "--inverse-rate" => {
                i += 1;
                cfg.inverse_rate = argv[i].parse().expect("invalid --inverse-rate");
            }

            "--auto-queries" => {
                cfg.auto_queries = true;
            }

            "--queries" => {
                i += 1;
                cfg.queries = argv[i].parse().expect("invalid --queries");
                cfg.auto_queries = false;
            }

            "--field-bits" => {
                i += 1;
                cfg.field_bits = argv[i].parse().expect("invalid --field-bits");
            }

            "--security" => {
                i += 1;
                cfg.security_bits = argv[i].parse().expect("invalid --security");
            }

            "--distance-failure" => {
                i += 1;
                cfg.distance_failure_bits = argv[i].parse().expect("invalid --distance-failure");
            }

            "--samples" => {
                i += 1;
                cfg.samples = argv[i].parse().expect("invalid --samples");
            }

            "--threads" => {
                i += 1;
                cfg.threads = argv[i].parse().expect("invalid --threads");
            }

            "--help" | "-h" => {
                print_help_and_exit();
            }

            other => {
                panic!("unknown argument: {other}");
            }
        }

        i += 1;
    }

    assert!(cfg.samples >= 1, "samples must be positive");
    assert!(cfg.inverse_rate >= 2, "inverse_rate must be at least 2");
    assert!(
        cfg.inverse_rate.is_power_of_two(),
        "current QA implementation expects inverse_rate to be a power of two"
    );
    assert!(cfg.field_bits >= 32, "field_bits is unexpectedly small");
    assert!(cfg.security_bits >= 1, "security must be positive");
    assert!(cfg.distance_failure_bits >= 1, "distance-failure must be positive");
    assert!(cfg.threads >= 1, "threads must be positive");

    cfg
}

fn print_help_and_exit() -> ! {
    eprintln!(
        "QABase full PCS benchmark over F_{{2^127-1}} (Mersenne127)\n\n\
         Recommended usage:\n\
           cargo bench -p benchmark --bench qabase_bench -- \\\n\
             --total-k 20..30 \\\n\
             --open-mode merge \\\n\
             --inverse-rate 2 \\\n\
             --field-bits 127 \\\n\
             --security 100 \\\n\
             --distance-failure 100 \\\n\
             --auto-queries \\\n\
             --samples 5 \\\n\
             --threads 32\n\n\
         Row schedule:\n\
           default: always use 2^--log-rows rows; with --log-rows=6 this is 64 rows\n\
           optional --small-row-schedule:\n\
             total_k = 20,21 -> 16 rows  (log_rows=4)\n\
             total_k = 22,23 -> 32 rows  (log_rows=5)\n\
             otherwise       -> 2^--log-rows rows\n\n\
         Options:\n\
           --total-k <k | a..b>       Total input-size exponent(s), inclusive\n\
           --k <k | a..b>             Alias for --total-k\n\
           --open-mode <merge|full>   merge: old proximity-only timing; full: complete protocol\n\
           --inverse-rate <c>         QA inverse rate c. Default: 2\n\
           --log-rows <r>             Default number of rows is 2^r. Default: 6\n\
           --small-row-schedule       Enable 16/32-row schedule for small sizes\n\
           --fixed-log-rows           Always use --log-rows. This is the default\n\
           --auto-queries             Compute queries from QA distance bound\n\
           --queries <q>              Manual Merkle query count; disables auto-queries\n\
           --field-bits <b>           Field-size bits for parameter calculation\n\
           --security <lambda>        Target query soundness bits\n\
           --distance-failure <bits>  Target QA-distance failure bits\n\
           --samples <s>              Number of samples\n\
           --threads <n>              Rayon worker threads\n"
    );

    std::process::exit(0);
}

// -----------------------------------------------------------------------------
// Security parameter calculation
// -----------------------------------------------------------------------------

fn qabase_gp(delta: f64, field_bits: usize) -> f64 {
    assert!(delta > 0.0 && delta < 1.0);

    let bits = field_bits as f64;

    1.0 - delta
        + (delta * delta.log2() + (1.0 - delta) * (1.0 - delta).log2()) / bits
}

fn log2_add(a: f64, b: f64) -> f64 {
    let m = a.max(b);

    if !m.is_finite() {
        return m;
    }

    m + ((2.0f64).powf(a - m) + (2.0f64).powf(b - m)).log2()
}

fn qabase_distance_failure_log2(
    delta: f64,
    row_log_size: usize,
    inverse_rate: usize,
    field_bits: usize,
) -> f64 {
    let c = inverse_rate;
    assert!(c >= 2, "inverse_rate must be at least 2");

    let log_n = row_log_size as f64;
    let log_p = field_bits as f64;

    // Approximation for parameter selection: p ≈ 2^field_bits.
    // For Mersenne127, p = 2^127 - 1, so log2(p - 1) is essentially 127
    // at the precision needed by this benchmark.
    let log_p_minus_one = field_bits as f64;

    let eps = qabase_gp(delta, field_bits) - (1.0 + log_n / log_p) / (c as f64);

    if eps <= 0.0 {
        return f64::INFINITY;
    }

    let denom_log = if log_p * (c as f64) * eps < 60.0 {
        (1.0 - (2.0f64).powf(-log_p * (c as f64) * eps)).log2()
    } else {
        0.0
    };

    // New Corollary 3.19 bound, first branch:
    //
    //   c(c-1)N/(2p^2)
    //     + p^{-ceil((c-1)/(c delta)) c eps}
    //       / ((1 - p^{-c eps})(p - 1)).
    let log_term1_a = ((c * (c - 1)) as f64 / 2.0).log2() + log_n - 2.0 * log_p;
    let threshold1 = (((c - 1) as f64) / ((c as f64) * delta)).ceil();
    let log_term1_b =
        -log_p * threshold1 * (c as f64) * eps - denom_log - log_p_minus_one;
    let log_bound1 = log2_add(log_term1_a, log_term1_b);

    // New Corollary 3.19 bound, second branch:
    //
    //   cN/p
    //     + p^{-ceil(1/delta) c eps}
    //       / ((1 - p^{-c eps})(p - 1)).
    let log_term2_a = (c as f64).log2() + log_n - log_p;
    let threshold2 = (1.0 / delta).ceil();
    let log_term2_b =
        -log_p * threshold2 * (c as f64) * eps - denom_log - log_p_minus_one;
    let log_bound2 = log2_add(log_term2_a, log_term2_b);

    log_bound1.min(log_bound2)
}

fn qabase_distance_lower_bound(
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

fn qabase_queries_from_distance(delta: f64, security_bits: usize) -> usize {
    assert!(delta > 0.0 && delta < 1.0);

    let denom = -(1.0 - delta / 3.0).log2();

    ((security_bits as f64) / denom).ceil() as usize
}

// -----------------------------------------------------------------------------
// Benchmark logic
// -----------------------------------------------------------------------------

fn make_random_matrix(
    row_k: usize,
    num_rows: usize,
    rng: &mut ChaCha8Rng,
) -> Vec<Vec<BenchField>> {
    let row_size = 1usize << row_k;

    (0..num_rows)
        .map(|_| {
            (0..row_size)
                .map(|_| BenchField::random(&mut *rng))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
}

fn make_random_point(num_vars: usize, rng: &mut ChaCha8Rng) -> Vec<BenchField> {
    (0..num_vars)
        .map(|_| BenchField::random(&mut *rng))
        .collect::<Vec<_>>()
}

fn split_evaluation_point(
    point: &[BenchField],
    log_rows: usize,
    row_k: usize,
) -> (Vec<BenchField>, Vec<BenchField>) {
    assert_eq!(
        point.len(),
        log_rows + row_k,
        "full evaluation point length mismatch"
    );

    let z_left = point[..log_rows].to_vec();
    let z_right = point[log_rows..].to_vec();

    (z_left, z_right)
}

fn equality_eval_at_index(
    index: usize,
    num_vars: usize,
    point: &[BenchField],
) -> BenchField {
    assert_eq!(point.len(), num_vars, "point length mismatch");
    assert!(index < (1usize << num_vars), "index out of range");

    let mut acc = BenchField::ONE;

    for i in 0..num_vars {
        acc *= if ((index >> i) & 1) == 1 {
            point[i]
        } else {
            BenchField::ONE - point[i]
        };
    }

    acc
}

fn eval_mle_from_evals_local(evals: &[BenchField], point: &[BenchField]) -> BenchField {
    assert!(
        evals.len().is_power_of_two(),
        "MLE eval vector length must be a power of two"
    );
    assert_eq!(
        evals.len(),
        1usize << point.len(),
        "point length does not match eval length"
    );

    let mut layer = evals.to_vec();
    let mut size = layer.len();

    for r in point {
        let half = size >> 1;

        for i in 0..half {
            let left = layer[2 * i];
            let right = layer[2 * i + 1];
            layer[i] = left * (BenchField::ONE - *r) + right * (*r);
        }

        size = half;
    }

    layer[0]
}

fn evaluate_matrix_at_point(
    matrix: &[Vec<BenchField>],
    z_left: &[BenchField],
    z_right: &[BenchField],
) -> BenchField {
    assert!(!matrix.is_empty(), "cannot evaluate an empty matrix");
    assert_eq!(
        matrix.len(),
        1usize << z_left.len(),
        "z_left length does not match matrix row count"
    );

    let row_len = matrix[0].len();
    assert_eq!(
        row_len,
        1usize << z_right.len(),
        "z_right length does not match matrix row length"
    );

    for row in matrix {
        assert_eq!(row.len(), row_len, "all rows must have the same length");
    }

    let mut folded_row = vec![BenchField::ZERO; row_len];

    for (row_index, row) in matrix.iter().enumerate() {
        let weight = equality_eval_at_index(row_index, z_left.len(), z_left);

        for j in 0..row_len {
            folded_row[j] += weight * row[j];
        }
    }

    eval_mle_from_evals_local(&folded_row, z_right)
}

fn avg_after_warmup(times: &[Duration]) -> Duration {
    assert!(!times.is_empty());

    let start = if times.len() >= 5 { 2 } else { 0 };

    let mut acc = Duration::new(0, 0);

    for t in &times[start..] {
        acc += *t;
    }

    acc / ((times.len() - start) as u32)
}

fn bench_one_total_k(total_k: usize, cfg: &BenchConfig) -> BenchResult {
    let choice = SecurityChoice::new(total_k, cfg);
    let row_size = choice.row_size();

    println!(
        "QABase parameters: total=2^{}, row=2^{}, log_rows={}, rows={}, c={}, field_bits={}, security={}, distance_failure={}, delta={:.8}, queries={}, threads={}",
        choice.total_k,
        choice.row_k,
        choice.log_rows,
        choice.num_rows,
        choice.inverse_rate,
        choice.field_bits,
        choice.security_bits,
        choice.distance_failure_bits,
        choice.delta,
        choice.queries,
        current_num_threads(),
    );

    let mut rng = ChaCha8Rng::from_seed([91u8; 32]);

    let param = setup::<BenchField, Blake2s>(
        row_size,
        1,
        &mut rng,
        Some(choice.num_rows),
        Some(choice.inverse_rate),
        Some(choice.queries),
    );

    let (pp, vp) = trim::<BenchField, Blake2s>(&param, row_size, 1);

    let matrix = make_random_matrix(choice.row_k, choice.num_rows, &mut rng);

    // Public evaluation point and claimed value for Full mode.
    // These are computed outside the timed prove/verify sections.
    let full_point = make_random_point(choice.total_k, &mut rng);
    let (z_left, z_right) =
        split_evaluation_point(&full_point, choice.log_rows, choice.row_k);
    let claimed_value = evaluate_matrix_at_point(&matrix, &z_left, &z_right);

    let mut commit_times = Vec::with_capacity(cfg.samples);
    let mut prove_times = Vec::with_capacity(cfg.samples);
    let mut verify_times = Vec::with_capacity(cfg.samples);

    let mut last_proof_bytes = 0usize;

    for sample_idx in 0..cfg.samples {
        let mut prover_transcript = TestTranscript::new(());

        let start = Instant::now();
        let comm = commit_and_write::<BenchField, Blake2s>(&pp, &matrix, &mut prover_transcript);
        commit_times.push(start.elapsed());

        let start = Instant::now();
        match cfg.open_mode {
            OpenMode::Merge => {
                prove_qabase_open_scaffold_global_batch_batched_wht::<BenchField, Blake2s>(
                    &pp,
                    &matrix,
                    &comm,
                    &mut prover_transcript,
                )
                .expect("QABase merge prover failed");
            }
            OpenMode::Full => {
                prove_qabase_open_full_global_batch_batched_wht::<BenchField, Blake2s>(
                    &pp,
                    &matrix,
                    &comm,
                    z_left.clone(),
                    z_right.clone(),
                    claimed_value,
                    &mut prover_transcript,
                )
                .expect("QABase full prover failed");
            }
        }
        prove_times.push(start.elapsed());

        let proof = prover_transcript.into_proof();
        last_proof_bytes = proof.len();

        let mut verifier_transcript = TestTranscript::from_proof((), proof.as_slice());

        let start = Instant::now();
        let ok = match cfg.open_mode {
            OpenMode::Merge => {
                let (ok, _verifier_output) =
                    verify_qabase_open_scaffold_global_batch_batched_wht::<BenchField, Blake2s>(
                        &vp,
                        &comm,
                        &mut verifier_transcript,
                    )
                    .expect("QABase merge verifier errored");
                ok
            }
            OpenMode::Full => {
                let (ok, _verifier_output) =
                    verify_qabase_open_full_global_batch_batched_wht::<BenchField, Blake2s>(
                        &vp,
                        &comm,
                        z_left.clone(),
                        z_right.clone(),
                        claimed_value,
                        &mut verifier_transcript,
                    )
                    .expect("QABase full verifier errored");
                ok
            }
        };
        verify_times.push(start.elapsed());

        assert!(
            ok,
            "QABase verifier rejected in {} mode",
            cfg.open_mode.as_str()
        );

        println!(
            "sample {sample_idx}: mode={}, total_k={}, row_k={}, log_rows={}, rows={}, c={}, queries={}, proof={} bytes",
            cfg.open_mode.as_str(),
            choice.total_k,
            choice.row_k,
            choice.log_rows,
            choice.num_rows,
            choice.inverse_rate,
            choice.queries,
            last_proof_bytes,
        );
    }

    BenchResult {
        field: "mersenne127",
        open_mode: cfg.open_mode,
        total_k: choice.total_k,
        row_k: choice.row_k,
        log_rows: choice.log_rows,
        num_rows: choice.num_rows,
        inverse_rate: choice.inverse_rate,
        field_bits: choice.field_bits,
        security_bits: choice.security_bits,
        distance_failure_bits: choice.distance_failure_bits,
        delta: choice.delta,
        queries: choice.queries,
        proof_bytes: last_proof_bytes,
        threads: current_num_threads(),
        commit_avg: avg_after_warmup(&commit_times),
        prove_avg: avg_after_warmup(&prove_times),
        verify_avg: avg_after_warmup(&verify_times),
    }
}

// -----------------------------------------------------------------------------
// CSV output
// -----------------------------------------------------------------------------

fn ensure_output_dir() {
    if !Path::new(OUTPUT_DIR).exists() {
        create_dir_all(OUTPUT_DIR).expect("failed to create benchmark output directory");
    }
}

fn output_path(cfg: &BenchConfig) -> String {
    let q_tag = if cfg.auto_queries {
        "auto".to_string()
    } else {
        cfg.queries.to_string()
    };

    let row_tag = if cfg.small_row_schedule {
        format!("smallrows16_32_default{}", 1usize << cfg.log_rows)
    } else {
        format!("rows{}", 1usize << cfg.log_rows)
    };

    format!(
        "{OUTPUT_DIR}/qabase_mersenne127_{}_rho{}_{}_sec{}_df{}_queries{}_th{}.csv",
        cfg.open_mode.as_str(),
        cfg.inverse_rate,
        row_tag,
        cfg.security_bits,
        cfg.distance_failure_bits,
        q_tag,
        cfg.threads,
    )
}

fn write_header_if_new(path: &str) {
    if !Path::new(path).exists() {
        let mut f = File::create(path).expect("failed to create output csv");

        writeln!(
            &mut f,
            "field,open_mode,total_k,row_k,log_rows,num_rows,inverse_rate,field_bits,security_bits,distance_failure_bits,delta,queries,proof_bytes,threads,commit_ms,prove_ms,verify_ms"
        )
        .unwrap();
    }
}

fn append_result(path: &str, result: &BenchResult) {
    let mut f = OpenOptions::new()
        .append(true)
        .open(path)
        .expect("failed to open output csv");

    writeln!(
        &mut f,
        "{},{},{},{},{},{},{},{},{},{},{:.8},{},{},{},{},{},{}",
        result.field,
        result.open_mode.as_str(),
        result.total_k,
        result.row_k,
        result.log_rows,
        result.num_rows,
        result.inverse_rate,
        result.field_bits,
        result.security_bits,
        result.distance_failure_bits,
        result.delta,
        result.queries,
        result.proof_bytes,
        result.threads,
        result.commit_avg.as_millis(),
        result.prove_avg.as_millis(),
        result.verify_avg.as_millis(),
    )
    .unwrap();
}
