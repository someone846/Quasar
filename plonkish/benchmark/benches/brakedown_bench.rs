//! Standalone Brakedown benchmark over the Mersenne127 field (p = 2^127 - 1).
//!
//! Target benchmark:
//!   - field: Mersenne127
//!   - polynomial size: 2^20 through 2^30 by default
//!   - Brakedown inverse rate: 2.0, i.e. code rate 1/2
//!   - Brakedown distance parameter: delta = beta / R = 0.1
//!   - target security: 100-bit
//!   - column openings: ceil(100 / -log2(1 - delta/3)) = 2045
//!   - threads: 32 by default
//!   - samples: 5 by default, average after dropping the first 2 samples when samples >= 5
//!
//! Example:
//!   cargo bench -p benchmark --bench brakedown_bench -- \
//!     --k 20..31 \
//!     --threads 32 \
//!     --samples 5

use plonkish_backend::{
    pcs::{multilinear::MultilinearBrakedown, PolynomialCommitmentScheme},
    poly::{multilinear::MultilinearPolynomial, Polynomial},
    util::{
        code::{BrakedownSpec, LinearCodes},
        hash::{Blake2s, Output},
        new_fields::Mersenne127,
        transcript::{Blake2sTranscript, InMemoryTranscript, TranscriptRead, TranscriptWrite},
    },
};

use rand::rngs::OsRng;
use std::{
    env::args,
    fs::{create_dir_all, OpenOptions},
    io::Write,
    ops::Range,
    path::Path,
    time::{Duration, Instant},
};

const OUTPUT_DIR: &str = "./bench_data/pcs";
const DEFAULT_K_START: usize = 20;
const DEFAULT_K_END_EXCLUSIVE: usize = 31;
const DEFAULT_THREADS: usize = 32;
const DEFAULT_SAMPLES: usize = 5;

const BRAKEDOWN_SECURITY_BITS: usize = 100;
const BRAKEDOWN_ALPHA: f64 = 0.30;
const BRAKEDOWN_BETA: f64 = 0.20;
const BRAKEDOWN_INVERSE_RATE: f64 = 2.0;
const BRAKEDOWN_DELTA: f64 = BRAKEDOWN_BETA / BRAKEDOWN_INVERSE_RATE;
const BRAKEDOWN_CN: usize = 11;
const BRAKEDOWN_DN: usize = 22;
const BRAKEDOWN_COLUMN_OPENINGS: usize = 2045;

#[derive(Debug)]
struct BrakedownRate2Security100;

impl BrakedownSpec for BrakedownRate2Security100 {
    const LAMBDA: f64 = BRAKEDOWN_SECURITY_BITS as f64;
    const ALPHA: f64 = BRAKEDOWN_ALPHA;
    const BETA: f64 = BRAKEDOWN_BETA;
    const R: f64 = BRAKEDOWN_INVERSE_RATE;

    fn c_n(_n: usize) -> usize {
        BRAKEDOWN_CN
    }

    fn d_n(_log2_q: usize, _n: usize) -> usize {
        BRAKEDOWN_DN
    }

    fn num_column_opening() -> usize {
        BRAKEDOWN_COLUMN_OPENINGS
    }
}

type Brakedown127 = MultilinearBrakedown<Mersenne127, Blake2s, BrakedownRate2Security100>;

#[derive(Clone, Debug)]
struct BenchArgs {
    k_range: Range<usize>,
    threads: usize,
    samples: usize,
}

fn main() {
    let bench_args = parse_args();

    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(bench_args.threads)
        .build_global();

    create_output_files();

    println!(
        "Brakedown127 bench: k={:?}, field=Mersenne127, inverse_rate={:.3}, delta={:.3}, security_bits={}, column_openings={}, c_n={}, d_n={}, threads={}, samples={}",
        bench_args.k_range,
        BRAKEDOWN_INVERSE_RATE,
        BRAKEDOWN_DELTA,
        BRAKEDOWN_SECURITY_BITS,
        BRAKEDOWN_COLUMN_OPENINGS,
        BRAKEDOWN_CN,
        BRAKEDOWN_DN,
        rayon::current_num_threads(),
        bench_args.samples,
    );

    for k in bench_args.k_range.clone() {
        bench_brakedown127::<Blake2sTranscript<_>>(k, &bench_args);
    }
}

fn bench_brakedown127<T>(k: usize, bench_args: &BenchArgs)
where
    T: TranscriptRead<Output<Blake2s>, Mersenne127>
        + TranscriptWrite<Output<Blake2s>, Mersenne127>
        + InMemoryTranscript<Param = ()>,
{
    let mut rng = OsRng;
    let poly_size = 1usize << k;

    let param = Brakedown127::setup(poly_size, 1, &mut rng).unwrap();
    let (pp, vp) = Brakedown127::trim(&param, poly_size, 1).unwrap();

    let row_len = pp.brakedown().row_len();
    let row_log = row_len.trailing_zeros() as usize;
    let num_rows = pp.num_rows();
    let log_rows = num_rows.trailing_zeros() as usize;
    let codeword_len = pp.brakedown().codeword_len();
    let proximity_reps = pp.brakedown().num_proximity_testing();
    let column_queries = pp.brakedown().num_column_opening();
    let actual_inverse_rate = codeword_len as f64 / row_len as f64;

    println!(
        "\nRunning Brakedown127: k={}, poly_size=2^{}, row_log={}, row_len={}, log_rows={}, num_rows={}, codeword_len={}, actual_inverse_rate={:.4}, proximity_reps={}, column_queries={}, samples={}, threads={}",
        k,
        k,
        row_log,
        row_len,
        log_rows,
        num_rows,
        codeword_len,
        actual_inverse_rate,
        proximity_reps,
        column_queries,
        bench_args.samples,
        rayon::current_num_threads(),
    );

    assert_eq!(
        proximity_reps, 1,
        "This 100-bit benchmark expects one Brakedown proximity repetition over Mersenne127. If this fails, verify the selected row/codeword length."
    );

    let poly = MultilinearPolynomial::<Mersenne127>::rand(k, OsRng);

    let sample_count = bench_args.samples.max(1);
    let warmup = if sample_count >= 5 { 2 } else { 0 };
    let denom = (sample_count - warmup) as u32;

    let mut commit_times = Vec::with_capacity(sample_count);
    let mut prove_times = Vec::with_capacity(sample_count);
    let mut last_proof = Vec::new();

    for sample_idx in 0..sample_count {
        let mut transcript = T::new(());

        let commit_start = Instant::now();
        let comm = Brakedown127::commit_and_write(&pp, &poly, &mut transcript).unwrap();
        let commit_elapsed = commit_start.elapsed();
        commit_times.push(commit_elapsed);

        let prove_start = Instant::now();
        let point = transcript.squeeze_challenges(k);
        let eval = poly.evaluate(point.as_slice());
        transcript.write_field_element(&eval).unwrap();
        Brakedown127::open(&pp, &poly, &comm, &point, &eval, &mut transcript).unwrap();
        let prove_elapsed = prove_start.elapsed();
        prove_times.push(prove_elapsed);

        let proof = transcript.into_proof();

        println!(
            "  sample {}: commit={} ms, prove={} ms, proof_bytes={}",
            sample_idx,
            commit_elapsed.as_millis(),
            prove_elapsed.as_millis(),
            proof.len(),
        );

        last_proof = proof;
    }

    let commit_avg = commit_times[warmup..].iter().copied().sum::<Duration>() / denom;
    let prove_avg = prove_times[warmup..].iter().copied().sum::<Duration>() / denom;

    let proof_bytes = last_proof.len();
    let proof_kb = proof_bytes as f64 / 1024.0;

    let mut verify_times = Vec::with_capacity(sample_count);
    for _ in 0..sample_count {
        let mut transcript = T::from_proof((), last_proof.as_slice());
        let verify_start = Instant::now();

        let comm = Brakedown127::read_commitment(&vp, &mut transcript).unwrap();
        let point = transcript.squeeze_challenges(k);
        let eval = transcript.read_field_element().unwrap();
        Brakedown127::verify(&vp, &comm, &point, &eval, &mut transcript).unwrap();

        verify_times.push(verify_start.elapsed());
    }

    let verify_avg = verify_times[warmup..].iter().copied().sum::<Duration>() / denom;

    append_line(
        "commit_brakedown127_rate2_100",
        &format!("{}, {}", k, commit_avg.as_millis()),
    );
    append_line(
        "open_brakedown127_rate2_100",
        &format!("{}, {}", k, prove_avg.as_millis()),
    );
    append_line(
        "verify_brakedown127_rate2_100",
        &format!("{}, {}", k, verify_avg.as_millis()),
    );
    append_line(
        "size_brakedown127_rate2_100",
        &format!("{}, {}", k, proof_bytes),
    );

    append_line(
        "summary_brakedown127_rate2_100.csv",
        &format!(
            "Mersenne127,{},{},{},{},{},{},{:.6},{},{:.6},{},{},{},{},{},{},{},{},{},{},{:.2}",
            k,
            row_log,
            log_rows,
            num_rows,
            row_len,
            codeword_len,
            actual_inverse_rate,
            BRAKEDOWN_SECURITY_BITS,
            BRAKEDOWN_DELTA,
            BRAKEDOWN_CN,
            BRAKEDOWN_DN,
            proximity_reps,
            column_queries,
            rayon::current_num_threads(),
            sample_count,
            commit_avg.as_millis(),
            prove_avg.as_millis(),
            verify_avg.as_millis(),
            proof_bytes,
            proof_kb,
        ),
    );

    println!(
        "Brakedown127 result: k={}, row_log={}, log_rows={}, actual_inverse_rate={:.4}, proximity_reps={}, column_queries={}, commit_ms={}, prove_ms={}, verify_ms={}, proof_bytes={}, proof_kb={:.2}",
        k,
        row_log,
        log_rows,
        actual_inverse_rate,
        proximity_reps,
        column_queries,
        commit_avg.as_millis(),
        prove_avg.as_millis(),
        verify_avg.as_millis(),
        proof_bytes,
        proof_kb,
    );
}

fn parse_args() -> BenchArgs {
    let mut k_range = DEFAULT_K_START..DEFAULT_K_END_EXCLUSIVE;
    let mut threads = DEFAULT_THREADS;
    let mut samples = DEFAULT_SAMPLES;

    let argv = args().collect::<Vec<_>>();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--k" => {
                let value = argv.get(i + 1).expect("--k requires a value");
                k_range = parse_range(value);
                i += 2;
            }
            "--threads" => {
                let value = argv.get(i + 1).expect("--threads requires a value");
                threads = value.parse().expect("--threads must be a usize");
                i += 2;
            }
            "--samples" => {
                let value = argv.get(i + 1).expect("--samples requires a value");
                samples = value.parse().expect("--samples must be a usize");
                i += 2;
            }
            "--bench" => {
                i += 1;
            }
            _ => {
                i += 1;
            }
        }
    }

    BenchArgs {
        k_range,
        threads,
        samples,
    }
}

fn parse_range(value: &str) -> Range<usize> {
    if let Some((start, end)) = value.split_once("..") {
        start.parse().expect("k range start must be usize")
            ..end.parse().expect("k range end must be usize")
    } else {
        let k = value.parse().expect("k must be usize");
        k..(k + 1)
    }
}

fn create_output_files() {
    if !Path::new(OUTPUT_DIR).exists() {
        create_dir_all(OUTPUT_DIR).unwrap();
    }

    touch_output_file("commit_brakedown127_rate2_100");
    touch_output_file("open_brakedown127_rate2_100");
    touch_output_file("verify_brakedown127_rate2_100");
    touch_output_file("size_brakedown127_rate2_100");

    let summary_path = format!("{}/summary_brakedown127_rate2_100.csv", OUTPUT_DIR);
    let should_write_header = !Path::new(&summary_path).exists()
        || std::fs::metadata(&summary_path)
            .map(|metadata| metadata.len() == 0)
            .unwrap_or(true);

    let mut summary = OpenOptions::new()
        .create(true)
        .append(true)
        .open(summary_path)
        .unwrap();

    if should_write_header {
        writeln!(
            summary,
            "field,k,row_log,log_rows,num_rows,row_len,codeword_len,actual_inverse_rate,security_bits,delta,c_n,d_n,proximity_reps,column_queries,threads,samples,commit_ms,prove_ms,verify_ms,proof_bytes,proof_kb"
        )
        .unwrap();
    }
}

fn touch_output_file(name: &str) {
    OpenOptions::new()
        .create(true)
        .append(true)
        .open(format!("{}/{}", OUTPUT_DIR, name))
        .unwrap();
}

fn append_line(name: &str, line: &str) {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(format!("{}/{}", OUTPUT_DIR, name))
        .unwrap();
    writeln!(file, "{}", line).unwrap();
}
