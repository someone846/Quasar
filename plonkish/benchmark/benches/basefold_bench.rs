//! Standalone BaseFold benchmark over the Mersenne127 field (p = 2^127 - 1).
//!
//! Default benchmark target:
//!   - field: Mersenne127
//!   - k: 20..31, i.e. polynomial sizes 2^20 through 2^30
//!   - threads: 32
//!   - samples: 5, average after dropping the first 2 samples when samples >= 5
//!   - BaseFold log_rate: 1, i.e. inverse rate 2 / code rate 1/2
//!   - verifier queries: 241, the 100-bit setting for rate 1/2 under the standard unique-decoding estimate
//!   - RS basecode enabled with basecode_rounds = 2
//!
//! Example:
//!   cargo bench -p benchmark --bench basefold_bench -- \
//!     --k 20..31 \
//!     --threads 32 \
//!     --samples 5

use plonkish_backend::{
    pcs::{
        multilinear::{Basefold, BasefoldExtParams},
        PolynomialCommitmentScheme,
    },
    poly::{multilinear::MultilinearPolynomial, Polynomial},
    util::{
        arithmetic::PrimeField,
        hash::Blake2s,
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
const DEFAULT_K_END_EXCLUSIVE: usize = 31; // 20..31 means 2^20 through 2^30.
const DEFAULT_THREADS: usize = 32;
const DEFAULT_SAMPLES: usize = 5;

const BASEFOLD_LOG_RATE: usize = 1; // inverse rate = 2^1 = 2, code rate = 1/2.
const BASEFOLD_REPS: usize = 241; // 100-bit query soundness for rate 1/2.
const BASEFOLD_BASECODE_ROUNDS: usize = 2;
const BASEFOLD_RS_BASECODE: bool = true;
const BASEFOLD_CODE_TYPE: &str = "random";

#[derive(Debug)]
struct Basefold127Params;

impl BasefoldExtParams for Basefold127Params {
    fn get_reps() -> usize {
        BASEFOLD_REPS
    }

    fn get_rate() -> usize {
        BASEFOLD_LOG_RATE
    }

    fn get_basecode_rounds() -> usize {
        BASEFOLD_BASECODE_ROUNDS
    }

    fn get_rs_basecode() -> bool {
        BASEFOLD_RS_BASECODE
    }

    fn get_code_type() -> String {
        BASEFOLD_CODE_TYPE.to_string()
    }
}

type Basefold127 = Basefold<Mersenne127, Blake2s, Basefold127Params>;

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
        "BaseFold127 bench: k={:?}, field=Mersenne127, log_rate={}, inverse_rate={}, queries={}, basecode_rounds={}, rs_basecode={}, code_type={}, threads={}, samples={}",
        bench_args.k_range,
        BASEFOLD_LOG_RATE,
        1usize << BASEFOLD_LOG_RATE,
        BASEFOLD_REPS,
        BASEFOLD_BASECODE_ROUNDS,
        BASEFOLD_RS_BASECODE,
        BASEFOLD_CODE_TYPE,
        rayon::current_num_threads(),
        bench_args.samples,
    );

    for k in bench_args.k_range.clone() {
        bench_basefold127::<Basefold127, Blake2sTranscript<_>>(k, &bench_args);
    }
}

fn bench_basefold127<Pcs, T>(k: usize, bench_args: &BenchArgs)
where
    Pcs: PolynomialCommitmentScheme<Mersenne127, Polynomial = MultilinearPolynomial<Mersenne127>>,
    T: TranscriptRead<Pcs::CommitmentChunk, Mersenne127>
        + TranscriptWrite<Pcs::CommitmentChunk, Mersenne127>
        + InMemoryTranscript<Param = ()>,
{
    let mut rng = OsRng;
    let poly_size = 1usize << k;

    println!(
        "\nRunning BaseFold127: k={}, poly_size=2^{}, codeword_size=2^{}, samples={}, threads={}",
        k,
        k,
        k + BASEFOLD_LOG_RATE,
        bench_args.samples,
        rayon::current_num_threads(),
    );

    let param = Pcs::setup(poly_size, 1, &mut rng).unwrap();
    let (pp, vp) = Pcs::trim(&param, poly_size, 1).unwrap();
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
        let comm = Pcs::commit_and_write(&pp, &poly, &mut transcript).unwrap();
        let commit_elapsed = commit_start.elapsed();
        commit_times.push(commit_elapsed);

        let prove_start = Instant::now();
        let point = transcript.squeeze_challenges(k);
        let eval = poly.evaluate(point.as_slice());
        transcript.write_field_element(&eval).unwrap();
        Pcs::open(&pp, &poly, &comm, &point, &eval, &mut transcript).unwrap();
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

        let comm = Pcs::read_commitment(&vp, &mut transcript).unwrap();
        let point = transcript.squeeze_challenges(k);
        let eval = transcript.read_field_element().unwrap();
        Pcs::verify(&vp, &comm, &point, &eval, &mut transcript).unwrap();

        verify_times.push(verify_start.elapsed());
    }

    let verify_avg = verify_times[warmup..].iter().copied().sum::<Duration>() / denom;

    append_line("commit_basefold127", &format!("{}, {}", k, commit_avg.as_millis()));
    append_line("open_basefold127", &format!("{}, {}", k, prove_avg.as_millis()));
    append_line("verify_basefold127", &format!("{}, {}", k, verify_avg.as_millis()));
    append_line("size_basefold127", &format!("{}, {}", k, proof_bytes));

    append_line(
        "summary_basefold127.csv",
        &format!(
            "Mersenne127,{},{},{},{},{},{},{},{},{},{},{},{},{},{:.2}",
            k,
            BASEFOLD_LOG_RATE,
            1usize << BASEFOLD_LOG_RATE,
            BASEFOLD_REPS,
            BASEFOLD_BASECODE_ROUNDS,
            BASEFOLD_RS_BASECODE,
            BASEFOLD_CODE_TYPE,
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
        "BaseFold127 result: k={}, commit_ms={}, prove_ms={}, verify_ms={}, proof_bytes={}, proof_kb={:.2}",
        k,
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
            // Cargo/criterion may pass these through depending on bench harness settings.
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

    // Do not truncate previous benchmark results.
    // These calls only ensure the files exist. New measurements are appended by append_line().
    touch_output_file("commit_basefold127");
    touch_output_file("open_basefold127");
    touch_output_file("verify_basefold127");
    touch_output_file("size_basefold127");

    let summary_path = format!("{}/summary_basefold127.csv", OUTPUT_DIR);
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
            "field,k,log_rate,inverse_rate,queries,basecode_rounds,rs_basecode,code_type,threads,samples,commit_ms,prove_ms,verify_ms,proof_bytes,proof_kb"
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
