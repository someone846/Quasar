use plonkish_backend::{
    pcs::multilinear::quasar::{
        commit_and_write, eval_mle_from_evals,
        prove_qabase_open_full_two_layer_gkr, qabase_split_evaluation_point,
        setup, trim, verify_qabase_open_full_two_layer_gkr, QABaseCommitment,
        QABaseProverParams, QABaseVerifierParams, QACodewordColumns,
    },
    util::{
        arithmetic::Field,
        hash::{Blake2s, Output},
        new_fields::Mersenne127,
        transcript::{Blake2sTranscript, InMemoryTranscript},
    },
};
use qa_gpu_overlay::quasar_commit::{CudaQuasarCommitter, QuasarCommitBackend};
use rand_chacha::{rand_core::SeedableRng, ChaCha8Rng};
use rayon::{current_num_threads, ThreadPoolBuilder};
use std::{
    env,
    io::Cursor,
    process::ExitCode,
    str::FromStr,
    time::{Duration, Instant},
};

type TestTranscript = Blake2sTranscript<Cursor<Vec<u8>>>;

#[derive(Clone, Debug)]
struct Config {
    total_k: usize,
    log_rows: usize,
    inverse_rate: usize,
    queries: usize,
    samples: usize,
    threads: usize,
    backend: QuasarCommitBackend,
    gpu_batch_rows: usize,
    check_cpu_root: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            total_k: 20,
            log_rows: 6,
            inverse_rate: 2,
            queries: 8,
            samples: 3,
            threads: 32,
            backend: QuasarCommitBackend::Cpu,
            gpu_batch_rows: 8,
            check_cpu_root: false,
        }
    }
}

fn parse_args() -> Config {
    let mut config = Config::default();
    let args = env::args().collect::<Vec<_>>();
    let mut i = 1usize;
    while i < args.len() {
        let read_value = |i: &mut usize, flag: &str| -> &str {
            *i += 1;
            args.get(*i)
                .map(String::as_str)
                .unwrap_or_else(|| panic!("missing value after {flag}"))
        };
        match args[i].as_str() {
            "--total-k" => {
                config.total_k = read_value(&mut i, "--total-k")
                    .parse()
                    .expect("invalid --total-k")
            }
            "--log-rows" => {
                config.log_rows = read_value(&mut i, "--log-rows")
                    .parse()
                    .expect("invalid --log-rows")
            }
            "--inverse-rate" => {
                config.inverse_rate = read_value(&mut i, "--inverse-rate")
                    .parse()
                    .expect("invalid --inverse-rate")
            }
            "--queries" => {
                config.queries = read_value(&mut i, "--queries")
                    .parse()
                    .expect("invalid --queries")
            }
            "--samples" => {
                config.samples = read_value(&mut i, "--samples")
                    .parse()
                    .expect("invalid --samples")
            }
            "--threads" => {
                config.threads = read_value(&mut i, "--threads")
                    .parse()
                    .expect("invalid --threads")
            }
            "--commit-backend" => {
                config.backend = QuasarCommitBackend::from_str(read_value(
                    &mut i,
                    "--commit-backend",
                ))
                .expect("invalid --commit-backend")
            }
            "--gpu-batch-rows" => {
                config.gpu_batch_rows = read_value(&mut i, "--gpu-batch-rows")
                    .parse()
                    .expect("invalid --gpu-batch-rows")
            }
            "--check-cpu-root" => config.check_cpu_root = true,
            "--help" | "-h" => {
                println!(
                    "qa_commit_cpu_cuda [options]\n\
                     \n\
                     --commit-backend cpu|cuda  commitment encoder (default cpu)\n\
                     --total-k K                total polynomial exponent (default 20)\n\
                     --log-rows R               matrix row exponent (default 6)\n\
                     --inverse-rate C           QA inverse rate (default 2)\n\
                     --queries Q                Merkle queries (default 8)\n\
                     --samples N                repetitions (default 3)\n\
                     --threads N                Rayon threads (default 32)\n\
                     --gpu-batch-rows N         CUDA batch rows (default 8)\n\
                     --check-cpu-root           exact CPU/CUDA root check outside timing"
                );
                std::process::exit(0);
            }
            other => panic!("unknown argument {other}"),
        }
        i += 1;
    }

    assert!(config.total_k >= config.log_rows);
    assert!(config.log_rows > 0);
    assert!(config.inverse_rate >= 2 && config.inverse_rate.is_power_of_two());
    assert!(config.queries > 0);
    assert!(config.samples > 0);
    assert!(config.threads > 0);
    assert!(config.gpu_batch_rows > 0);
    config
}

fn random_matrix(
    rows: usize,
    cols: usize,
    rng: &mut ChaCha8Rng,
) -> Vec<Vec<Mersenne127>> {
    (0..rows)
        .map(|_| {
            (0..cols)
                .map(|_| Mersenne127::random(&mut *rng))
                .collect()
        })
        .collect()
}

fn prove_and_verify<C>(
    pp: &QABaseProverParams<Mersenne127, Blake2s>,
    vp: &QABaseVerifierParams<Mersenne127, Blake2s>,
    word: &[Vec<Mersenne127>],
    commitment: &QABaseCommitment<Mersenne127, Blake2s, C>,
    z_left: &[Mersenne127],
    z_right: &[Mersenne127],
    claimed_value: Mersenne127,
    mut prover_transcript: TestTranscript,
) -> (Duration, Duration, usize)
where
    C: QACodewordColumns<Mersenne127>,
{
    let prove_start = Instant::now();
    prove_qabase_open_full_two_layer_gkr(
        pp,
        word,
        commitment,
        z_left.to_vec(),
        z_right.to_vec(),
        claimed_value,
        &mut prover_transcript,
    )
    .expect("Quasar prover failed");
    let prove_time = prove_start.elapsed();

    let proof = prover_transcript.into_proof();
    let proof_bytes = proof.len();
    let mut verifier_transcript = TestTranscript::from_proof((), proof.as_slice());
    let verify_start = Instant::now();
    let (ok, _) = verify_qabase_open_full_two_layer_gkr(
        vp,
        commitment,
        z_left.to_vec(),
        z_right.to_vec(),
        claimed_value,
        &mut verifier_transcript,
    )
    .expect("Quasar verifier errored");
    let verify_time = verify_start.elapsed();
    assert!(ok, "Quasar verifier rejected");
    (prove_time, verify_time, proof_bytes)
}

fn average(values: &[Duration]) -> Duration {
    values.iter().copied().sum::<Duration>() / values.len() as u32
}

fn run(config: Config) -> Result<(), String> {
    let row_k = config.total_k - config.log_rows;
    let row_len = 1usize << row_k;
    let rows = 1usize << config.log_rows;
    println!(
        "Quasar commitment backend={} total=2^{} rows={} row_len=2^{} rate=1/{} queries={} samples={} rayon_threads={}",
        config.backend,
        config.total_k,
        rows,
        row_k,
        config.inverse_rate,
        config.queries,
        config.samples,
        current_num_threads(),
    );

    let mut rng = ChaCha8Rng::from_seed([91u8; 32]);
    let params = setup::<Mersenne127, Blake2s>(
        row_len,
        1,
        &mut rng,
        Some(rows),
        Some(config.inverse_rate),
        Some(config.queries),
    );
    let (pp, vp) = trim::<Mersenne127, Blake2s>(&params, row_len, 1);
    let word = random_matrix(rows, row_len, &mut rng);
    let full_point = (0..config.total_k)
        .map(|_| Mersenne127::random(&mut rng))
        .collect::<Vec<_>>();
    let (z_left, z_right) =
        qabase_split_evaluation_point(&full_point, rows, row_k);
    let flat_word = word.iter().flatten().copied().collect::<Vec<_>>();
    let claimed_value = eval_mle_from_evals(&flat_word, &full_point);
    drop(flat_word);

    let mut commit_times = Vec::with_capacity(config.samples);
    let mut prove_times = Vec::with_capacity(config.samples);
    let mut verify_times = Vec::with_capacity(config.samples);
    let mut proof_bytes = 0usize;

    match config.backend {
        QuasarCommitBackend::Cpu => {
            for sample in 1..=config.samples {
                let mut transcript = TestTranscript::new(());
                let commit_start = Instant::now();
                let commitment = commit_and_write(&pp, &word, &mut transcript);
                let commit_time = commit_start.elapsed();
                let (prove_time, verify_time, bytes) = prove_and_verify(
                    &pp,
                    &vp,
                    &word,
                    &commitment,
                    &z_left,
                    &z_right,
                    claimed_value,
                    transcript,
                );
                proof_bytes = bytes;
                commit_times.push(commit_time);
                prove_times.push(prove_time);
                verify_times.push(verify_time);
                println!(
                    "sample {sample}: commit={:.3} ms prove={:.3} ms verify={:.3} ms proof={} bytes",
                    commit_time.as_secs_f64() * 1e3,
                    prove_time.as_secs_f64() * 1e3,
                    verify_time.as_secs_f64() * 1e3,
                    bytes,
                );
            }
        }
        QuasarCommitBackend::Cuda => {
            let expected_cpu_root: Option<Output<Blake2s>> = if config.check_cpu_root {
                let mut cpu_transcript = TestTranscript::new(());
                let cpu_commitment = commit_and_write(&pp, &word, &mut cpu_transcript);
                let root: &Output<Blake2s> = cpu_commitment.as_ref();
                Some(root.clone())
            } else {
                None
            };
            let mut cuda =
                CudaQuasarCommitter::new(&pp, &word, config.gpu_batch_rows)?;
            println!("GPU: {}", cuda.device_name()?);
            println!(
                "one-time pinned input registration: {:.3} ms",
                cuda.input_setup_timing().pin_registration.as_secs_f64() * 1e3
            );
            println!(
                "one-time device commitment + pinned digest setup: {:.3} ms",
                cuda.output_setup_timing().total.as_secs_f64() * 1e3
            );
            let warm_up = cuda.warm_up()?;
            println!(
                "one-time CUDA warm-up: {:.3} ms",
                warm_up.total_wall.as_secs_f64() * 1e3
            );

            for sample in 1..=config.samples {
                let mut transcript = TestTranscript::new(());
                let commit_start = Instant::now();
                let (commitment, gpu_timing) =
                    cuda.commit_and_write(&pp, &mut transcript)?;
                let commit_time = commit_start.elapsed();

                if let Some(cpu_root) = &expected_cpu_root {
                    let gpu_root: &Output<Blake2s> = commitment.as_ref();
                    assert_eq!(gpu_root, cpu_root, "CPU/CUDA Quasar roots differ");
                }

                let (prove_time, verify_time, bytes) = prove_and_verify(
                    &pp,
                    &vp,
                    &word,
                    &commitment,
                    &z_left,
                    &z_right,
                    claimed_value,
                    transcript,
                );
                let (query_transfers, query_transfer_time) =
                    commitment.codeword.query_transfer_stats();
                cuda.reclaim(commitment)?;
                proof_bytes = bytes;
                commit_times.push(commit_time);
                prove_times.push(prove_time);
                verify_times.push(verify_time);
                println!(
                    "sample {sample}: commit={:.3} ms [device pipeline={:.3} ms, H2D={:.3} ms, GPU leaf hash={:.3} ms, digest D2H={:.3} ms, host leaf decode={:.3} ms, CPU upper Merkle={:.3} ms] prove={:.3} ms [query D2H={} cols/{:.3} ms] verify={:.3} ms proof={} bytes",
                    commit_time.as_secs_f64() * 1e3,
                    gpu_timing.total_wall.as_secs_f64() * 1e3,
                    gpu_timing.host_to_device_ms,
                    gpu_timing.column_hash_ms,
                    gpu_timing.digest_device_to_host_ms,
                    gpu_timing.host_leaf_decode.as_secs_f64() * 1e3,
                    gpu_timing.cpu_upper_merkle.as_secs_f64() * 1e3,
                    prove_time.as_secs_f64() * 1e3,
                    query_transfers,
                    query_transfer_time.as_secs_f64() * 1e3,
                    verify_time.as_secs_f64() * 1e3,
                    bytes,
                );
            }
        }
    }

    println!(
        "summary backend={}: commit={:.3} ms prove={:.3} ms verify={:.3} ms proof={} bytes",
        config.backend,
        average(&commit_times).as_secs_f64() * 1e3,
        average(&prove_times).as_secs_f64() * 1e3,
        average(&verify_times).as_secs_f64() * 1e3,
        proof_bytes,
    );
    Ok(())
}

fn main() -> ExitCode {
    let config = parse_args();
    let pool = ThreadPoolBuilder::new()
        .num_threads(config.threads)
        .build()
        .expect("failed to build Rayon thread pool");
    match pool.install(|| run(config)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}
