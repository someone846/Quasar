//! Standalone BrakingBase benchmark over Mersenne127.
//!
//! This uses the same rate-1/2, 100-bit Brakedown parameters as
//! `brakedown_bench`, and the rate-1/2, 100-bit BaseFold parameters used by
//! Quasar.  Setup includes SPARK preprocessing and the seven static BaseFold
//! commitments; it is reported separately from the online commitment time.
//!
//! Example:
//!   cargo bench -p benchmark --bench brakingbase_bench -- \
//!     --k 20..30 --rows 8 --threads 32 --samples 5

use plonkish_backend::{
    pcs::{
        multilinear::{BasefoldExtParams, MultilinearBrakingBase},
        PolynomialCommitmentScheme,
    },
    poly::multilinear::MultilinearPolynomial,
    util::{
        code::BrakedownSpec,
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

const BRAKEDOWN_ALPHA: f64 = 0.30;
const BRAKEDOWN_BETA: f64 = 0.20;
const BRAKEDOWN_INVERSE_RATE: f64 = 2.0;
const BRAKEDOWN_CN: usize = 11;
const BRAKEDOWN_DN: usize = 22;
const BRAKEDOWN_COLUMN_OPENINGS: usize = 2045;
const BASEFOLD_LOG_RATE: usize = 1;
const BASEFOLD_QUERIES: usize = 241;

#[derive(Debug)]
struct BrakedownRate2Security100;

impl BrakedownSpec for BrakedownRate2Security100 {
    const LAMBDA: f64 = 100.0;
    const ALPHA: f64 = BRAKEDOWN_ALPHA;
    const BETA: f64 = BRAKEDOWN_BETA;
    const R: f64 = BRAKEDOWN_INVERSE_RATE;

    fn c_n(_: usize) -> usize {
        BRAKEDOWN_CN
    }

    fn d_n(_: usize, _: usize) -> usize {
        BRAKEDOWN_DN
    }

    fn num_column_opening() -> usize {
        BRAKEDOWN_COLUMN_OPENINGS
    }
}

#[derive(Debug)]
struct BasefoldRate2Security100;

impl BasefoldExtParams for BasefoldRate2Security100 {
    fn get_reps() -> usize {
        BASEFOLD_QUERIES
    }

    fn get_rate() -> usize {
        BASEFOLD_LOG_RATE
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

type BrakingBase127 = MultilinearBrakingBase<
    Mersenne127,
    Blake2s,
    BrakedownRate2Security100,
    BasefoldRate2Security100,
>;

#[derive(Clone, Debug)]
struct BenchArgs {
    k_range: Range<usize>,
    rows: Option<usize>,
    threads: usize,
    samples: usize,
}

fn main() {
    let bench_args = parse_args();
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(bench_args.threads)
        .build_global();
    create_output_file();

    println!(
        "BrakingBase127: k={:?}, requested_rows={:?}, Brakedown rate=1/2, columns={}, BaseFold rate=1/2, BaseFold queries={}, threads={}, samples={}",
        bench_args.k_range,
        bench_args.rows,
        BRAKEDOWN_COLUMN_OPENINGS,
        BASEFOLD_QUERIES,
        rayon::current_num_threads(),
        bench_args.samples,
    );

    for k in bench_args.k_range.clone() {
        bench_one::<Blake2sTranscript<_>>(k, &bench_args);
    }
}

fn bench_one<T>(k: usize, bench_args: &BenchArgs)
where
    T: TranscriptRead<Output<Blake2s>, Mersenne127>
        + TranscriptWrite<Output<Blake2s>, Mersenne127>
        + InMemoryTranscript<Param = ()>,
{
    let poly_size = 1usize << k;
    let setup_start = Instant::now();
    let params = if let Some(rows) = bench_args.rows {
        BrakingBase127::setup_with_num_rows(poly_size, rows, OsRng).unwrap()
    } else {
        BrakingBase127::setup(poly_size, 1, OsRng).unwrap()
    };
    let (pp, vp) = BrakingBase127::trim(&params, poly_size, 1).unwrap();
    let setup_time = setup_start.elapsed();

    println!(
        "\nk={}, rows={}, row_len={}, codeword_len={}, SPARK ops={}, SPARK aux_len={}, setup={} ms",
        k,
        pp.num_rows(),
        pp.row_len(),
        pp.codeword_len(),
        pp.spark_num_ops(),
        pp.spark_aux_len(),
        setup_time.as_millis(),
    );

    let poly = MultilinearPolynomial::<Mersenne127>::rand(k, OsRng);
    let sample_count = bench_args.samples.max(1);
    let warmup = if sample_count >= 5 { 2 } else { 0 };
    let denom = (sample_count - warmup) as u32;
    let mut commit_times = Vec::with_capacity(sample_count);
    let mut prove_times = Vec::with_capacity(sample_count);
    let mut last_proof = Vec::new();

    for sample in 0..sample_count {
        let mut transcript = T::new(());
        let start = Instant::now();
        let comm = BrakingBase127::commit_and_write(&pp, &poly, &mut transcript).unwrap();
        let commit_time = start.elapsed();

        let point = transcript.squeeze_challenges(k);
        let eval = poly.evaluate(&point);
        transcript.write_field_element(&eval).unwrap();
        let start = Instant::now();
        BrakingBase127::open(&pp, &poly, &comm, &point, &eval, &mut transcript).unwrap();
        let prove_time = start.elapsed();
        let proof = transcript.into_proof();

        println!(
            "  sample {}: commit={} ms, prove={} ms, proof_bytes={}",
            sample,
            commit_time.as_millis(),
            prove_time.as_millis(),
            proof.len(),
        );
        commit_times.push(commit_time);
        prove_times.push(prove_time);
        last_proof = proof;
    }

    let commit_avg = average(&commit_times[warmup..], denom);
    let prove_avg = average(&prove_times[warmup..], denom);
    let mut verify_times = Vec::with_capacity(sample_count);
    for _ in 0..sample_count {
        let mut transcript = T::from_proof((), &last_proof);
        let start = Instant::now();
        let comm = BrakingBase127::read_commitment(&vp, &mut transcript).unwrap();
        let point = transcript.squeeze_challenges(k);
        let eval = transcript.read_field_element().unwrap();
        BrakingBase127::verify(&vp, &comm, &point, &eval, &mut transcript).unwrap();
        verify_times.push(start.elapsed());
    }
    let verify_avg = average(&verify_times[warmup..], denom);

    append_summary(&format!(
        "Mersenne127,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
        k,
        pp.num_rows(),
        pp.row_len(),
        pp.codeword_len(),
        pp.num_queries(),
        pp.spark_num_ops(),
        pp.spark_aux_len(),
        BASEFOLD_QUERIES,
        rayon::current_num_threads(),
        sample_count,
        setup_time.as_millis(),
        commit_avg.as_millis(),
        prove_avg.as_millis(),
        verify_avg.as_millis(),
        last_proof.len(),
    ));
}

fn average(times: &[Duration], denominator: u32) -> Duration {
    times.iter().copied().sum::<Duration>() / denominator
}

fn parse_args() -> BenchArgs {
    let mut parsed = BenchArgs {
        k_range: DEFAULT_K_START..DEFAULT_K_END_EXCLUSIVE,
        rows: None,
        threads: DEFAULT_THREADS,
        samples: DEFAULT_SAMPLES,
    };
    let argv = args().collect::<Vec<_>>();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--k" => {
                parsed.k_range = parse_range(argv.get(i + 1).expect("--k requires a value"));
                i += 2;
            }
            "--threads" => {
                parsed.threads = argv
                    .get(i + 1)
                    .expect("--threads requires a value")
                    .parse()
                    .expect("--threads must be a usize");
                i += 2;
            }
            "--rows" => {
                parsed.rows = Some(
                    argv.get(i + 1)
                        .expect("--rows requires a value")
                        .parse()
                        .expect("--rows must be a usize"),
                );
                i += 2;
            }
            "--samples" => {
                parsed.samples = argv
                    .get(i + 1)
                    .expect("--samples requires a value")
                    .parse()
                    .expect("--samples must be a usize");
                i += 2;
            }
            "--bench" => i += 1,
            _ => i += 1,
        }
    }
    parsed
}

fn parse_range(value: &str) -> Range<usize> {
    if let Some((start, end)) = value.split_once("..") {
        start.parse().expect("k range start must be usize")
            ..end.parse().expect("k range end must be usize")
    } else {
        let k = value.parse().expect("k must be usize");
        k..k + 1
    }
}

fn create_output_file() {
    create_dir_all(OUTPUT_DIR).unwrap();
    let path = format!("{OUTPUT_DIR}/summary_brakingbase127_rate2_100.csv");
    let needs_header = !Path::new(&path).exists()
        || std::fs::metadata(&path)
            .map(|metadata| metadata.len() == 0)
            .unwrap_or(true);
    if needs_header {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .unwrap();
        writeln!(
            file,
            "field,k,num_rows,row_len,codeword_len,column_queries,spark_ops,spark_aux_len,basefold_queries,threads,samples,setup_ms,commit_ms,prove_ms,verify_ms,proof_bytes"
        )
        .unwrap();
    }
}

fn append_summary(line: &str) {
    let path = format!("{OUTPUT_DIR}/summary_brakingbase127_rate2_100.csv");
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .unwrap();
    writeln!(file, "{line}").unwrap();
}
