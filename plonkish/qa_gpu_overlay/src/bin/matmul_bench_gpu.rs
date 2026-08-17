//! End-to-end non-ZK verifiable matrix multiplication using Quasar's
//! device-resident hybrid CUDA commitment path.

#[path = "../../../benchmark/benches/matmul_common.rs"]
mod matmul_common;

use matmul_common::{
    application_points, build_evals, parse_exp_list, prepare_application, prove_product_sumcheck,
    seed32, verify_product_sumcheck, BenchField, MatmulShape,
};
use plonkish_backend::{
    pcs::multilinear::quasar::{
        commit_and_write as cpu_commit_and_write, prove_qabase_open_full_two_layer_gkr,
        qabase_distance_lower_bound, qabase_queries_from_distance, qabase_split_evaluation_point,
        setup, trim, verify_qabase_open_full_two_layer_gkr, QABaseCommitment, QABaseProverParams,
        QABaseVerifierParams,
    },
    util::{
        arithmetic::{Field, PrimeField},
        hash::{Blake2s, Output},
        transcript::{Blake2sTranscript, InMemoryTranscript},
    },
};
use qa_gpu_overlay::{
    gpu::{GpuQaDeviceOutput, GpuQaInput},
    quasar_commit::CudaQuasarCommitter,
};
use rand_chacha::{rand_core::SeedableRng, ChaCha8Rng};
use rayon::{current_num_threads, ThreadPoolBuilder};
use std::{
    env,
    fs::{create_dir_all, File, OpenOptions},
    hint::black_box,
    io::{Cursor, Write},
    path::{Path, PathBuf},
    process::ExitCode,
    time::{Duration, Instant},
};

type BenchHash = Blake2s;
type BenchTranscript = Blake2sTranscript<Cursor<Vec<u8>>>;
type DeviceCommitment = QABaseCommitment<BenchField, BenchHash, GpuQaDeviceOutput>;

const HASH_BYTES: usize = 32;

#[derive(Clone, Debug)]
struct Args {
    k_values: Vec<usize>,
    samples: usize,
    threads: usize,
    log_m: usize,
    quasar_log_rows: usize,
    inverse_rate: usize,
    field_bits: usize,
    security_bits: usize,
    distance_failure_bits: usize,
    gpu_batch_rows: usize,
    check_cpu_root: bool,
    seed: u8,
    output: PathBuf,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            k_values: (20..=30).collect(),
            samples: 5,
            threads: 32,
            log_m: 5,
            quasar_log_rows: 6,
            inverse_rate: 2,
            field_bits: 127,
            security_bits: 100,
            distance_failure_bits: 100,
            gpu_batch_rows: 8,
            check_cpu_root: false,
            seed: 113,
            output: PathBuf::from("./bench_data/matmul/matmul_20_30.csv"),
        }
    }
}

impl Args {
    fn parse() -> Self {
        let mut cfg = Self::default();
        let argv = env::args().collect::<Vec<_>>();
        let mut i = 1;
        while i < argv.len() {
            let value = |i: &mut usize, flag: &str| -> &str {
                *i += 1;
                argv.get(*i)
                    .map(String::as_str)
                    .unwrap_or_else(|| panic!("missing value after {flag}"))
            };
            match argv[i].as_str() {
                "--k" | "--total-k" => {
                    cfg.k_values = parse_exp_list(value(&mut i, "--k"));
                }
                "--systems" => {
                    let systems = value(&mut i, "--systems");
                    assert!(
                        systems.split(',').all(|name| matches!(
                            name.trim().to_ascii_lowercase().as_str(),
                            "quasar-gpu" | "quasar_gpu" | "gpu"
                        )),
                        "the CUDA companion only runs quasar-gpu"
                    );
                }
                "--samples" => {
                    cfg.samples = value(&mut i, "--samples")
                        .parse()
                        .expect("invalid --samples")
                }
                "--threads" => {
                    cfg.threads = value(&mut i, "--threads")
                        .parse()
                        .expect("invalid --threads")
                }
                "--log-m" => cfg.log_m = value(&mut i, "--log-m").parse().expect("invalid --log-m"),
                "--quasar-log-rows" | "--log-rows" => {
                    cfg.quasar_log_rows = value(&mut i, "--quasar-log-rows")
                        .parse()
                        .expect("invalid --quasar-log-rows")
                }
                "--inverse-rate" => {
                    cfg.inverse_rate = value(&mut i, "--inverse-rate")
                        .parse()
                        .expect("invalid --inverse-rate")
                }
                "--field-bits" => {
                    cfg.field_bits = value(&mut i, "--field-bits")
                        .parse()
                        .expect("invalid --field-bits")
                }
                "--security" => {
                    cfg.security_bits = value(&mut i, "--security")
                        .parse()
                        .expect("invalid --security")
                }
                "--distance-failure" => {
                    cfg.distance_failure_bits = value(&mut i, "--distance-failure")
                        .parse()
                        .expect("invalid --distance-failure")
                }
                "--gpu-batch-rows" => {
                    cfg.gpu_batch_rows = value(&mut i, "--gpu-batch-rows")
                        .parse()
                        .expect("invalid --gpu-batch-rows")
                }
                "--check-cpu-root" => cfg.check_cpu_root = true,
                "--seed" => cfg.seed = value(&mut i, "--seed").parse().expect("invalid --seed"),
                "--output" => cfg.output = PathBuf::from(value(&mut i, "--output")),
                "--smoke" => {
                    cfg.k_values = vec![12];
                    cfg.samples = 1;
                    cfg.threads = 8;
                    cfg.gpu_batch_rows = 4;
                }
                "--help" | "-h" => print_help_and_exit(),
                other => panic!("unknown argument: {other}"),
            }
            i += 1;
        }
        assert!(!cfg.k_values.is_empty());
        assert!(cfg.samples > 0 && cfg.threads > 0 && cfg.gpu_batch_rows > 0);
        assert!(cfg.log_m > 0 && cfg.quasar_log_rows > 0);
        assert_eq!(cfg.inverse_rate, 2, "comparison is fixed to rate 1/2");
        assert_eq!(cfg.field_bits, 127, "comparison is fixed to Mersenne127");
        cfg
    }
}

struct QuasarKeys {
    pp: QABaseProverParams<BenchField, BenchHash>,
    vp: QABaseVerifierParams<BenchField, BenchHash>,
}

fn make_keys(total_log: usize, args: &Args, domain: usize) -> QuasarKeys {
    assert!(total_log >= args.quasar_log_rows);
    let row_log = total_log - args.quasar_log_rows;
    let row_size = 1usize << row_log;
    let num_rows = 1usize << args.quasar_log_rows;
    let delta = qabase_distance_lower_bound(
        row_log,
        args.inverse_rate,
        args.field_bits,
        args.distance_failure_bits,
    );
    let queries = qabase_queries_from_distance(delta, args.security_bits);
    let mut rng = ChaCha8Rng::from_seed(seed32(args.seed, total_log, domain, 71));
    let params = setup::<BenchField, BenchHash>(
        row_size,
        1,
        &mut rng,
        Some(num_rows),
        Some(args.inverse_rate),
        Some(queries),
    );
    let (pp, vp) = trim::<BenchField, BenchHash>(&params, row_size, 1);
    QuasarKeys { pp, vp }
}

fn open_one(
    keys: &QuasarKeys,
    word: &GpuQaInput,
    comm: &DeviceCommitment,
    point: &[BenchField],
    claimed: BenchField,
    mut transcript: BenchTranscript,
) -> (Vec<u8>, Duration) {
    let (z_left, z_right) =
        qabase_split_evaluation_point(point, keys.pp.num_rows, keys.pp.num_vars);
    let start = Instant::now();
    let output = prove_qabase_open_full_two_layer_gkr(
        &keys.pp,
        word,
        comm,
        z_left,
        z_right,
        claimed,
        &mut transcript,
    )
    .expect("Quasar-GPU opening prover failed");
    assert!(output.ok_eval_value);
    (transcript.into_proof(), start.elapsed())
}

fn verify_one(
    keys: &QuasarKeys,
    comm: &DeviceCommitment,
    point: &[BenchField],
    claimed: BenchField,
    proof: &[u8],
) -> Duration {
    let (z_left, z_right) =
        qabase_split_evaluation_point(point, keys.vp.num_rows, keys.vp.num_vars);
    let mut transcript = BenchTranscript::from_proof((), proof);
    let start = Instant::now();
    let (ok, _) = verify_qabase_open_full_two_layer_gkr(
        &keys.vp,
        comm,
        z_left,
        z_right,
        claimed,
        &mut transcript,
    )
    .expect("Quasar-GPU verifier errored");
    assert!(ok, "Quasar-GPU verifier rejected");
    start.elapsed()
}

#[derive(Debug)]
struct ResultRow {
    shape: MatmulShape,
    samples: usize,
    threads: usize,
    commit_ms: f64,
    app_ms: f64,
    sc_prove_ms: f64,
    open_ms: f64,
    prover_ms: f64,
    sc_verify_ms: f64,
    pcs_verify_ms: f64,
    verifier_ms: f64,
    pcs_bytes: usize,
    sumcheck_bytes: usize,
}

fn bench_shape(shape: MatmulShape, args: &Args) -> Result<ResultRow, String> {
    let keys_a = make_keys(shape.log_a(), args, 1);
    let keys_b = make_keys(shape.log_b(), args, 2);
    let keys_c = make_keys(shape.log_c(), args, 3);
    let evals = build_evals(&shape, seed32(args.seed, shape.k, usize::MAX - 1, 31));

    // Move each dense witness directly into pinned storage. The same table is
    // used by the application, opening prover, and CUDA H2D path.
    let mut cuda_a = CudaQuasarCommitter::new_flat(&keys_a.pp, evals.a, args.gpu_batch_rows)?;
    let mut cuda_b = CudaQuasarCommitter::new_flat(&keys_b.pp, evals.b, args.gpu_batch_rows)?;
    let mut cuda_c = CudaQuasarCommitter::new_flat(&keys_c.pp, evals.c, args.gpu_batch_rows)?;
    eprintln!("GPU: {}", cuda_b.device_name()?);
    cuda_a.warm_up()?;
    cuda_b.warm_up()?;
    cuda_c.warm_up()?;

    let cpu_roots = if args.check_cpu_root {
        Some(
            [(&keys_a, &cuda_a), (&keys_b, &cuda_b), (&keys_c, &cuda_c)]
                .into_iter()
                .map(|(keys, cuda)| {
                    let mut transcript = BenchTranscript::new(());
                    let comm = cpu_commit_and_write(&keys.pp, cuda.word(), &mut transcript);
                    let root: &Output<BenchHash> = comm.as_ref();
                    root.clone()
                })
                .collect::<Vec<_>>(),
        )
    } else {
        None
    };

    let mut commits = Vec::with_capacity(args.samples);
    let mut apps = Vec::with_capacity(args.samples);
    let mut sc_proves = Vec::with_capacity(args.samples);
    let mut opens = Vec::with_capacity(args.samples);
    let mut provers = Vec::with_capacity(args.samples);
    let mut sc_verifies = Vec::with_capacity(args.samples);
    let mut pcs_verifies = Vec::with_capacity(args.samples);
    let mut verifiers = Vec::with_capacity(args.samples);
    let mut pcs_bytes = 0;
    let mut sumcheck_bytes = 0;

    for sample in 0..args.samples {
        let prover_start = Instant::now();
        let commit_start = Instant::now();
        let mut tr_a = BenchTranscript::new(());
        let (comm_a, _) = cuda_a.commit_and_write(&keys_a.pp, &mut tr_a)?;
        let mut tr_b = BenchTranscript::new(());
        let (comm_b, _) = cuda_b.commit_and_write(&keys_b.pp, &mut tr_b)?;
        let mut tr_c = BenchTranscript::new(());
        let (comm_c, _) = cuda_c.commit_and_write(&keys_c.pp, &mut tr_c)?;
        let commit_time = commit_start.elapsed();
        commits.push(commit_time);

        if let Some(roots) = &cpu_roots {
            let gpu_a: &Output<BenchHash> = comm_a.as_ref();
            let gpu_b: &Output<BenchHash> = comm_b.as_ref();
            let gpu_c: &Output<BenchHash> = comm_c.as_ref();
            assert_eq!(gpu_a, &roots[0], "CPU/CUDA A roots differ");
            assert_eq!(gpu_b, &roots[1], "CPU/CUDA B roots differ");
            assert_eq!(gpu_c, &roots[2], "CPU/CUDA C roots differ");
        }

        let app_start = Instant::now();
        let app = prepare_application(
            &shape,
            cuda_a.input_values(),
            cuda_b.input_values(),
            cuda_c.input_values(),
            ChaCha8Rng::from_seed(seed32(args.seed, shape.k, sample, 51)),
        );
        let app_time = app_start.elapsed();
        apps.push(app_time);

        let sc_seed = seed32(args.seed, shape.k, sample, 52);
        let sc_start = Instant::now();
        let sc = prove_product_sumcheck(
            &app.a_y,
            &app.b_y,
            app.c_eval,
            ChaCha8Rng::from_seed(sc_seed),
        );
        let sc_time = sc_start.elapsed();
        sc_proves.push(sc_time);
        let (point_a, point_b, point_c) = application_points(&app, &sc.ry);

        let (proof_a, open_a) =
            open_one(&keys_a, cuda_a.word(), &comm_a, &point_a, sc.a_eval, tr_a);
        let (proof_b, open_b) =
            open_one(&keys_b, cuda_b.word(), &comm_b, &point_b, sc.b_eval, tr_b);
        let (proof_c, open_c) =
            open_one(&keys_c, cuda_c.word(), &comm_c, &point_c, app.c_eval, tr_c);
        let open_time = open_a + open_b + open_c;
        opens.push(open_time);
        let prover_time = prover_start.elapsed();
        provers.push(prover_time);

        let raw_pcs_bytes = proof_a.len() + proof_b.len() + proof_c.len();
        assert!(raw_pcs_bytes >= 3 * HASH_BYTES);
        pcs_bytes = raw_pcs_bytes - 3 * HASH_BYTES;
        sumcheck_bytes = 3 * sc.proof.rounds.len() * field_bytes();

        let verifier_start = Instant::now();
        let sc_verify_start = Instant::now();
        let verifier_ry = verify_product_sumcheck(
            &sc.proof,
            app.c_eval,
            sc.a_eval,
            sc.b_eval,
            ChaCha8Rng::from_seed(sc_seed),
        )
        .expect("sumcheck verifier rejected");
        assert_eq!(verifier_ry, sc.ry);
        let sc_verify_time = sc_verify_start.elapsed();
        sc_verifies.push(sc_verify_time);
        let pcs_verify_time = verify_one(&keys_a, &comm_a, &point_a, sc.a_eval, &proof_a)
            + verify_one(&keys_b, &comm_b, &point_b, sc.b_eval, &proof_b)
            + verify_one(&keys_c, &comm_c, &point_c, app.c_eval, &proof_c);
        pcs_verifies.push(pcs_verify_time);
        let verifier_time = verifier_start.elapsed();
        verifiers.push(verifier_time);

        let query_stats = [
            comm_a.codeword.query_transfer_stats(),
            comm_b.codeword.query_transfer_stats(),
            comm_c.codeword.query_transfer_stats(),
        ];
        let query_columns = query_stats.iter().map(|x| x.0).sum::<u64>();
        let query_time = query_stats.iter().map(|x| x.1).sum::<Duration>();
        black_box((&comm_a, &comm_b, &comm_c));
        cuda_a.reclaim(comm_a)?;
        cuda_b.reclaim(comm_b)?;
        cuda_c.reclaim(comm_c)?;

        eprintln!(
            "[Quasar-GPU] k={} sample={} commit={:.3} app={:.3} sc={:.3} open={:.3} prover={:.3} verify={:.3} query_d2h={}/{:.3}ms proof={} B",
            shape.k, sample, ms(commit_time), ms(app_time), ms(sc_time),
            ms(open_time), ms(prover_time), ms(verifier_time), query_columns,
            ms(query_time), pcs_bytes + sumcheck_bytes + 3 * field_bytes(),
        );
    }

    Ok(ResultRow {
        shape,
        samples: args.samples,
        threads: current_num_threads(),
        commit_ms: ms(avg_after_warmup(&commits)),
        app_ms: ms(avg_after_warmup(&apps)),
        sc_prove_ms: ms(avg_after_warmup(&sc_proves)),
        open_ms: ms(avg_after_warmup(&opens)),
        prover_ms: ms(avg_after_warmup(&provers)),
        sc_verify_ms: ms(avg_after_warmup(&sc_verifies)),
        pcs_verify_ms: ms(avg_after_warmup(&pcs_verifies)),
        verifier_ms: ms(avg_after_warmup(&verifiers)),
        pcs_bytes,
        sumcheck_bytes,
    })
}

fn field_bytes() -> usize {
    BenchField::ZERO.to_repr().as_ref().len()
}

fn avg_after_warmup(times: &[Duration]) -> Duration {
    let start = if times.len() >= 5 { 2 } else { 0 };
    times[start..].iter().copied().sum::<Duration>() / (times.len() - start) as u32
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn write_header_if_new(path: &Path) {
    if !path.exists() {
        let mut file = File::create(path).expect("failed to create CSV");
        writeln!(file, "system,k,m,n,p,log_a,log_b,log_c,samples,threads,commit_ms,app_prepare_ms,sumcheck_prove_ms,pcs_open_ms,prover_total_ms,sumcheck_verify_ms,pcs_verify_ms,verifier_total_ms,pcs_proof_bytes,sumcheck_bytes,eval_claim_bytes,proof_bytes_excl_input_commitments,proof_bytes_incl_input_commitments,proof_kb_excl_input_commitments,proof_kb_incl_input_commitments").unwrap();
    }
}

fn append_result(path: &Path, r: &ResultRow) {
    let eval_bytes = 3 * field_bytes();
    let proof_excl = r.pcs_bytes + r.sumcheck_bytes + eval_bytes;
    let proof_incl = proof_excl + 3 * HASH_BYTES;
    let mut file = OpenOptions::new()
        .append(true)
        .open(path)
        .expect("failed to open CSV");
    writeln!(
        file,
        "Quasar-GPU,{},{},{},{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{},{},{},{},{:.3},{:.3}",
        r.shape.k, r.shape.m, r.shape.n, r.shape.p, r.shape.log_a(),
        r.shape.log_b(), r.shape.log_c(), r.samples, r.threads, r.commit_ms,
        r.app_ms, r.sc_prove_ms, r.open_ms, r.prover_ms, r.sc_verify_ms,
        r.pcs_verify_ms, r.verifier_ms, r.pcs_bytes, r.sumcheck_bytes,
        eval_bytes, proof_excl, proof_incl, proof_excl as f64 / 1024.0,
        proof_incl as f64 / 1024.0,
    ).unwrap();
}

fn run(args: Args) -> Result<(), String> {
    if let Some(parent) = args.output.parent() {
        create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    write_header_if_new(&args.output);
    eprintln!("matrix-multiplication CUDA benchmark config: {args:?}");
    for &k in &args.k_values {
        let result = bench_shape(MatmulShape::from_k(k, args.log_m), &args)?;
        println!(
            "Quasar-GPU k={} [{}x{} * {}x{}] prover={:.3} ms (commit {:.3}, app {:.3}, sc {:.3}, open {:.3}) verifier={:.3} ms",
            result.shape.k, result.shape.m, result.shape.n, result.shape.n,
            result.shape.p, result.prover_ms, result.commit_ms, result.app_ms,
            result.sc_prove_ms, result.open_ms, result.verifier_ms,
        );
        append_result(&args.output, &result);
    }
    Ok(())
}

fn main() -> ExitCode {
    let args = Args::parse();
    let pool = ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build()
        .expect("failed to build Rayon thread pool");
    match pool.install(|| run(args)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

fn print_help_and_exit() -> ! {
    println!(
        "Quasar-GPU verifiable matrix-multiplication benchmark\n\n\
         cargo +nightly run --release --features cuda --manifest-path qa_gpu_overlay/Cargo.toml --bin matmul_bench_gpu -- --k 20..=30 --samples 5 --threads 32 --quasar-log-rows 6 --gpu-batch-rows 8 --output ./bench_data/matmul/matmul_20_30.csv\n\n\
         Add --smoke --check-cpu-root for a small correctness run."
    );
    std::process::exit(0)
}
