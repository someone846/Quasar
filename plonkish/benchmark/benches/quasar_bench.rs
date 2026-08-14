use plonkish_backend::{
    pcs::multilinear::quasar::{
        commit_and_write, eval_mle_from_evals, prove_qabase_open_full_two_layer_gkr,
        qabase_row_weights_from_z_left, qabase_split_evaluation_point, setup, trim,
        verify_qabase_open_full_two_layer_gkr, QABaseCommitment,
    },
    util::{
        arithmetic::Field,
        hash::{Blake2s, Output},
        new_fields::Mersenne127,
        transcript::{Blake2sTranscript, InMemoryTranscript, TranscriptWrite},
    },
};

use rand_chacha::{rand_core::SeedableRng, ChaCha8Rng};
use rayon::{current_num_threads, prelude::*, ThreadPoolBuilder};

use std::{
    env::args,
    hint::black_box,
    fs::{create_dir_all, File, OpenOptions},
    io::{Cursor, Write},
    path::Path,
    time::{Duration, Instant},
};

type TestTranscript = Blake2sTranscript<Cursor<Vec<u8>>>;
type BenchField = Mersenne127;

const OUTPUT_DIR: &str = "./bench_data/quasar";

#[derive(Clone, Debug)]
struct BenchConfig {
    /// Total multilinear-polynomial size is 2^total_k field elements.
    total_k_values: Vec<usize>,

    /// The committed matrix has 2^log_rows rows and 2^(total_k-log_rows)
    /// entries per row.
    log_rows_values: Vec<usize>,

    /// QA inverse rate c; code rate is 1/c.
    inverse_rate: usize,

    /// Compute the number of Merkle queries from the QA distance bound.
    auto_queries: bool,

    /// Manual Merkle query count, used when auto_queries is false.
    queries: usize,

    field_bits: usize,
    security_bits: usize,
    distance_failure_bits: usize,

    /// Number of measured protocol executions per parameter point.
    samples: usize,

    /// Number of Rayon worker threads.
    threads: usize,

    /// Deterministic benchmark seed.
    seed: u8,
}

impl Default for BenchConfig {
    fn default() -> Self {
        // Deliberately safe defaults. Use command-line ranges for paper-scale
        // experiments; the full two-instance opening uses substantially more
        // memory than the old proximity-only scaffold.
        Self {
            total_k_values: vec![16],
            log_rows_values: vec![4],
            inverse_rate: 2,
            auto_queries: false,
            queries: 8,
            field_bits: 127,
            security_bits: 100,
            distance_failure_bits: 100,
            samples: 1,
            threads: 8,
            seed: 91,
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
    fn new(total_k: usize, log_rows: usize, cfg: &BenchConfig) -> Self {
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

        assert!(queries >= 1, "query count must be positive");

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
    protocol: &'static str,
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
    endpoint_commitments: usize,
    opening_claims: usize,
    unique_opening_points: usize,
    proof_bytes: usize,
    threads: usize,
    setup_time: Duration,
    trim_time: Duration,
    eval_prepare_time: Duration,
    commit_avg: Duration,
    prove_avg: Duration,
    verify_avg: Duration,
}

fn main() {
    let cfg = parse_args();

    ensure_output_dir();
    let path = output_path(&cfg);
    write_header_if_new(&path);

    println!("Quasar two-layer-GKR benchmark config: {cfg:?}");
    println!("writing CSV to {path}");

    let pool = ThreadPoolBuilder::new()
        .num_threads(cfg.threads)
        .build()
        .expect("failed to build Rayon thread pool");

    pool.install(|| {
        println!("rayon current_num_threads = {}", current_num_threads());

        for &total_k in &cfg.total_k_values {
            for &log_rows in &cfg.log_rows_values {
                if total_k < log_rows {
                    println!(
                        "skip total_k={total_k}, log_rows={log_rows}: total_k < log_rows"
                    );
                    continue;
                }

                let result = bench_one(total_k, log_rows, &cfg);

                println!(
                    "total_k={}, row_k={}, rows=2^{}={}, c={}, delta={:.8}, queries={}, endpoints={}, claims={}, unique_points={}, proof={} bytes, setup={} ms, trim={} ms, eval_prepare={} ms, commit={} ms, prove={} ms, verify={} ms",
                    result.total_k,
                    result.row_k,
                    result.log_rows,
                    result.num_rows,
                    result.inverse_rate,
                    result.delta,
                    result.queries,
                    result.endpoint_commitments,
                    result.opening_claims,
                    result.unique_opening_points,
                    result.proof_bytes,
                    millis(result.setup_time),
                    millis(result.trim_time),
                    millis(result.eval_prepare_time),
                    millis(result.commit_avg),
                    millis(result.prove_avg),
                    millis(result.verify_avg),
                );

                append_result(&path, &result);
            }
        }
    });
}

// -----------------------------------------------------------------------------
// Argument parsing
// -----------------------------------------------------------------------------

fn parse_exp_list(value: &str) -> Vec<usize> {
    let mut values = Vec::new();

    for raw_part in value.split(',') {
        let part = raw_part.trim();
        if part.is_empty() {
            continue;
        }

        if let Some((start, end)) = part.split_once("..=") {
            let start = start.parse::<usize>().expect("invalid range start");
            let end = end.parse::<usize>().expect("invalid range end");
            assert!(start <= end, "range start must be <= end");
            values.extend(start..=end);
        } else if let Some((start, end)) = part.split_once("..") {
            let start = start.parse::<usize>().expect("invalid range start");
            let end = end.parse::<usize>().expect("invalid range end");
            assert!(start <= end, "range start must be <= end");
            // For benchmark convenience, a..b is treated as inclusive.
            values.extend(start..=end);
        } else if let Some((start, end)) = part.split_once('-') {
            let start = start.parse::<usize>().expect("invalid range start");
            let end = end.parse::<usize>().expect("invalid range end");
            assert!(start <= end, "range start must be <= end");
            values.extend(start..=end);
        } else {
            values.push(part.parse::<usize>().expect("invalid exponent"));
        }
    }

    assert!(!values.is_empty(), "empty exponent list");
    values.sort_unstable();
    values.dedup();
    values
}

fn require_value<'a>(argv: &'a [String], i: &mut usize, flag: &str) -> &'a str {
    *i += 1;
    argv.get(*i)
        .unwrap_or_else(|| panic!("missing value after {flag}"))
        .as_str()
}

fn parse_args() -> BenchConfig {
    let mut cfg = BenchConfig::default();
    let argv = args().collect::<Vec<_>>();
    let mut i = 1usize;

    while i < argv.len() {
        match argv[i].as_str() {
            // Cargo may forward this libtest-style flag even when this bench
            // target uses `harness = false`. It is not a Quasar option.
            "--bench" => {
                // Be defensive in case a runner forwards `--bench <name>`.
                // In ordinary `cargo bench --bench quasar_bench -- ...`
                // invocation, the next argument is normally another flag.
                if i + 1 < argv.len() && !argv[i + 1].starts_with('-') {
                    i += 1;
                }
            }
            "--total-k" | "--k" => {
                let flag = argv[i].clone();
                cfg.total_k_values = parse_exp_list(require_value(&argv, &mut i, &flag));
            }
            "--log-rows" | "--row-logs" => {
                let flag = argv[i].clone();
                cfg.log_rows_values = parse_exp_list(require_value(&argv, &mut i, &flag));
            }
            "--inverse-rate" => {
                cfg.inverse_rate = require_value(&argv, &mut i, "--inverse-rate")
                    .parse()
                    .expect("invalid --inverse-rate");
            }
            "--auto-queries" => cfg.auto_queries = true,
            "--queries" => {
                cfg.queries = require_value(&argv, &mut i, "--queries")
                    .parse()
                    .expect("invalid --queries");
                cfg.auto_queries = false;
            }
            "--field-bits" => {
                cfg.field_bits = require_value(&argv, &mut i, "--field-bits")
                    .parse()
                    .expect("invalid --field-bits");
            }
            "--security" => {
                cfg.security_bits = require_value(&argv, &mut i, "--security")
                    .parse()
                    .expect("invalid --security");
            }
            "--distance-failure" => {
                cfg.distance_failure_bits = require_value(&argv, &mut i, "--distance-failure")
                    .parse()
                    .expect("invalid --distance-failure");
            }
            "--samples" => {
                cfg.samples = require_value(&argv, &mut i, "--samples")
                    .parse()
                    .expect("invalid --samples");
            }
            "--threads" => {
                cfg.threads = require_value(&argv, &mut i, "--threads")
                    .parse()
                    .expect("invalid --threads");
            }
            "--seed" => {
                cfg.seed = require_value(&argv, &mut i, "--seed")
                    .parse()
                    .expect("invalid --seed");
            }
            "--smoke" => {
                cfg.total_k_values = vec![12];
                cfg.log_rows_values = vec![2];
                cfg.inverse_rate = 2;
                cfg.auto_queries = false;
                cfg.queries = 4;
                cfg.samples = 1;
            }
            "--help" | "-h" => print_help_and_exit(),
            other => panic!("unknown argument: {other}"),
        }
        i += 1;
    }

    assert!(cfg.samples >= 1, "samples must be positive");
    assert!(cfg.threads >= 1, "threads must be positive");
    assert!(cfg.inverse_rate >= 2, "inverse_rate must be at least 2");
    assert!(
        cfg.inverse_rate.is_power_of_two(),
        "inverse_rate must be a power of two"
    );
    assert!(cfg.field_bits >= 32, "field_bits is unexpectedly small");
    assert!(cfg.security_bits >= 1, "security must be positive");
    assert!(
        cfg.distance_failure_bits >= 1,
        "distance-failure must be positive"
    );
    for &log_rows in &cfg.log_rows_values {
        assert!(log_rows >= 1, "log_rows must be positive");
    }

    cfg
}

fn print_help_and_exit() -> ! {
    eprintln!(
        "Quasar full PCS benchmark with endpoint-only two-layer GKR\n\n\
         Smoke test:\n\
           cargo bench -p benchmark --bench quasar_bench -- --smoke --threads 8\n\n\
         Rate-1/2 experiment:\n\
           cargo bench -p benchmark --bench quasar_bench -- \\\n             --total-k 20..=30 --log-rows 6 --inverse-rate 2 \\\n             --auto-queries --security 100 --distance-failure 100 \\\n             --samples 5 --threads 32\n\n\
         Rate-1/4 experiment:\n\
           cargo bench -p benchmark --bench quasar_bench -- \\\n             --total-k 20..=30 --log-rows 6 --inverse-rate 4 \\\n             --auto-queries --security 100 --distance-failure 100 \\\n             --samples 5 --threads 32\n\n\
         Options:\n\
           --total-k <k|a..b|a..=b|a-b|list>\n\
           --log-rows <r|a..b|a..=b|a-b|list>\n\
           --inverse-rate <c>\n\
           --auto-queries\n\
           --queries <q>\n\
           --field-bits <bits>\n\
           --security <bits>\n\
           --distance-failure <bits>\n\
           --samples <n>\n\
           --threads <n>\n\
           --seed <0..255>\n\
           --smoke\n"
    );
    std::process::exit(0);
}

// -----------------------------------------------------------------------------
// QA-distance parameter selection
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

    let log_term1_a = ((c * (c - 1)) as f64 / 2.0).log2() + log_n - 2.0 * log_p;
    let threshold1 = (((c - 1) as f64) / ((c as f64) * delta)).ceil();
    let log_term1_b =
        -log_p * threshold1 * (c as f64) * eps - denom_log - log_p_minus_one;
    let log_bound1 = log2_add(log_term1_a, log_term1_b);

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
// Benchmark helpers
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
        .collect()
}

fn random_point(len: usize, rng: &mut ChaCha8Rng) -> Vec<BenchField> {
    (0..len)
        .map(|_| BenchField::random(&mut *rng))
        .collect()
}

fn fold_rows_for_evaluation(
    rows: &[Vec<BenchField>],
    row_weights: &[BenchField],
) -> Vec<BenchField> {
    assert!(!rows.is_empty(), "cannot fold an empty matrix");
    assert_eq!(rows.len(), row_weights.len(), "row-weight count mismatch");

    let row_len = rows[0].len();
    for row in rows {
        assert_eq!(row.len(), row_len, "matrix rows must have equal length");
    }

    (0..row_len)
        .into_par_iter()
        .map(|column| {
            let mut acc = BenchField::ZERO;
            for (weight, row) in row_weights.iter().zip(rows.iter()) {
                acc += *weight * row[column];
            }
            acc
        })
        .collect()
}

fn avg_after_warmup(times: &[Duration]) -> Duration {
    assert!(!times.is_empty());
    let start = if times.len() >= 5 { 2 } else { 0 };
    let sum = times[start..]
        .iter()
        .copied()
        .fold(Duration::ZERO, |acc, item| acc + item);
    sum / ((times.len() - start) as u32)
}

fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn bench_one(total_k: usize, log_rows: usize, cfg: &BenchConfig) -> BenchResult {
    let choice = SecurityChoice::new(total_k, log_rows, cfg);
    let row_size = choice.row_size();

    println!(
        "parameters: total=2^{}, row=2^{}, rows=2^{}={}, c={}, field_bits={}, security={}, distance_failure={}, delta={:.8}, queries={}, threads={}",
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

    let mut seed = [cfg.seed; 32];
    seed[0] ^= total_k as u8;
    seed[1] ^= log_rows as u8;
    seed[2] ^= choice.inverse_rate as u8;
    let mut rng = ChaCha8Rng::from_seed(seed);

    let setup_start = Instant::now();
    let param = setup::<BenchField, Blake2s>(
        row_size,
        1,
        &mut rng,
        Some(choice.num_rows),
        Some(choice.inverse_rate),
        Some(choice.queries),
    );
    let setup_time = setup_start.elapsed();

    // trim performs preprocessing/indexing for the public E_i commitments.
    // It is reported separately and excluded from online prove time.
    let trim_start = Instant::now();
    let (pp, vp) = trim::<BenchField, Blake2s>(&param, row_size, 1);
    let trim_time = trim_start.elapsed();
    drop(param);

    let matrix = make_random_matrix(choice.row_k, choice.num_rows, &mut rng);

    // Prepare one valid multilinear evaluation claim. This work belongs to the
    // benchmark driver, not to the Quasar prover, and is therefore timed
    // separately.
    let eval_prepare_start = Instant::now();
    let full_point = random_point(choice.total_k, &mut rng);
    let (z_left, z_right) = qabase_split_evaluation_point::<BenchField>(
        &full_point,
        choice.num_rows,
        choice.row_k,
    );
    let row_weights =
        qabase_row_weights_from_z_left::<BenchField>(choice.num_rows, &z_left);
    let eval_msg = fold_rows_for_evaluation(&matrix, &row_weights);
    let claimed_value = eval_mle_from_evals::<BenchField>(&eval_msg, &z_right);
    let eval_prepare_time = eval_prepare_start.elapsed();

    let mut commit_times = Vec::with_capacity(cfg.samples);
    let mut prove_times = Vec::with_capacity(cfg.samples);
    let mut verify_times = Vec::with_capacity(cfg.samples);
    let mut last_proof_bytes = 0usize;
    let mut endpoint_commitments = 0usize;
    let mut opening_claims = 0usize;
    let mut unique_opening_points = 0usize;

    // ---------------------------------------------------------------------
    // Phase A: commitment-only measurements.
    //
    // No full proving or verification is executed between commit samples.
    // This prevents the much larger two-instance opening from polluting the
    // allocator/cache state used by the commitment benchmark.
    // ---------------------------------------------------------------------
    for sample_idx in 0..cfg.samples {
        let mut commit_transcript = TestTranscript::new(());

        let commit_start = Instant::now();
        let comm = commit_and_write::<BenchField, Blake2s>(
            &pp,
            &matrix,
            &mut commit_transcript,
        );
        let elapsed = commit_start.elapsed();

        // Ensure the compiler cannot discard the constructed commitment.
        black_box(&comm);
        black_box(&commit_transcript);
        commit_times.push(elapsed);

        println!(
            "commit-only sample {sample_idx}: total_k={}, row_k={}, rows={}, c={}, commit={:.3} ms",
            choice.total_k,
            choice.row_k,
            choice.num_rows,
            choice.inverse_rate,
            millis(elapsed),
        );

        drop(comm);
        drop(commit_transcript);
    }

    // Build one reusable commitment outside all measured prove/verify regions.
    // Every proof transcript below begins by writing exactly this commitment
    // root, which is equivalent to the transcript effect of commit_and_write.
    let mut commitment_build_transcript = TestTranscript::new(());
    let comm: QABaseCommitment<BenchField, Blake2s> =
        commit_and_write::<BenchField, Blake2s>(
            &pp,
            &matrix,
            &mut commitment_build_transcript,
        );
    drop(commitment_build_transcript);

    let committed_root: &Output<Blake2s> = comm.as_ref();

    // ---------------------------------------------------------------------
    // Phase B: full prove/verify measurements using the fixed commitment.
    // ---------------------------------------------------------------------
    for sample_idx in 0..cfg.samples {
        let mut prover_transcript = TestTranscript::new(());
        <TestTranscript as TranscriptWrite<Output<Blake2s>, BenchField>>::write_commitment(
            &mut prover_transcript,
            committed_root,
        )
        .expect("failed to write reusable Quasar commitment root");

        let prove_start = Instant::now();
        let prover_output = prove_qabase_open_full_two_layer_gkr::<BenchField, Blake2s>(
            &pp,
            &matrix,
            &comm,
            z_left.clone(),
            z_right.clone(),
            claimed_value,
            &mut prover_transcript,
        )
        .expect("Quasar two-layer-GKR prover failed");
        prove_times.push(prove_start.elapsed());

        endpoint_commitments = prover_output.endpoint_commitment_count;
        opening_claims = prover_output.opening_claim_count;
        unique_opening_points = prover_output.unique_opening_point_count;
        assert_eq!(
            endpoint_commitments,
            2 * choice.inverse_rate,
            "full opening should commit exactly two instances times c endpoint blocks"
        );
        assert!(prover_output.ok_eval_value);

        let proof = prover_transcript.into_proof();
        last_proof_bytes = proof.len();

        let mut verifier_transcript =
            TestTranscript::from_proof((), proof.as_slice());

        let verify_start = Instant::now();
        let (ok, verifier_output) =
            verify_qabase_open_full_two_layer_gkr::<BenchField, Blake2s>(
                &vp,
                &comm,
                z_left.clone(),
                z_right.clone(),
                claimed_value,
                &mut verifier_transcript,
            )
            .expect("Quasar two-layer-GKR verifier errored");
        verify_times.push(verify_start.elapsed());

        assert!(ok, "Quasar verifier rejected a valid proof");
        assert_eq!(
            prover_output.query_indices,
            verifier_output.query_indices,
            "prover/verifier query positions differ"
        );

        println!(
            "prove/verify sample {sample_idx}: total_k={}, row_k={}, rows={}, c={}, queries={}, endpoints={}, claims={}, unique_points={}, proof={} bytes",
            choice.total_k,
            choice.row_k,
            choice.num_rows,
            choice.inverse_rate,
            choice.queries,
            endpoint_commitments,
            opening_claims,
            unique_opening_points,
            last_proof_bytes,
        );
    }

    BenchResult {
        field: "mersenne127",
        protocol: "quasar_two_layer_gkr_optimized",
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
        endpoint_commitments,
        opening_claims,
        unique_opening_points,
        proof_bytes: last_proof_bytes,
        threads: current_num_threads(),
        setup_time,
        trim_time,
        eval_prepare_time,
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

    let row_tag = cfg
        .log_rows_values
        .iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join("_");

    format!(
        "{OUTPUT_DIR}/quasar_gkr_optimized_mersenne127_rho{}_logrows{}_sec{}_df{}_queries{}_th{}.csv",
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
        let mut file = File::create(path).expect("failed to create output CSV");
        writeln!(
            &mut file,
            "field,protocol,total_k,row_k,log_rows,num_rows,inverse_rate,field_bits,security_bits,distance_failure_bits,delta,queries,endpoint_commitments,opening_claims,unique_opening_points,proof_bytes,threads,setup_ms,trim_ms,eval_prepare_ms,commit_ms,prove_ms,verify_ms"
        )
        .expect("failed to write CSV header");
    }
}

fn append_result(path: &str, result: &BenchResult) {
    let mut file = OpenOptions::new()
        .append(true)
        .open(path)
        .expect("failed to open output CSV");

    writeln!(
        &mut file,
        "{},{},{},{},{},{},{},{},{},{},{:.8},{},{},{},{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
        result.field,
        result.protocol,
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
        result.endpoint_commitments,
        result.opening_claims,
        result.unique_opening_points,
        result.proof_bytes,
        result.threads,
        millis(result.setup_time),
        millis(result.trim_time),
        millis(result.eval_prepare_time),
        millis(result.commit_avg),
        millis(result.prove_avg),
        millis(result.verify_avg),
    )
    .expect("failed to append benchmark result");
}
