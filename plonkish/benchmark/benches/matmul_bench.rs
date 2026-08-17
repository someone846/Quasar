//! End-to-end non-ZK verifiable matrix-multiplication benchmark.
//!
//! Intended path:
//!
//!     benchmark/benches/matmul_bench.rs
//!
//! This benchmark evaluates
//!
//!     C = A * B,
//!
//! through the standard multilinear/sumcheck reduction
//!
//!     C~(r_x, r_z) = sum_y A~(r_x, y) B~(y, r_z),
//!
//! followed by three multilinear PCS openings.  The application layer is the
//! same for every PCS.  Only the commitment/opening backend changes.
//!
//! Sweep convention
//! ----------------
//! `--k 20..=30` means that the *largest committed MLE*, B, contains 2^k field
//! elements.  We fix m = 32 and choose n,p with n*p = 2^k.  For even k we use
//! p=4n; for odd k we use p=2n.  Thus k=26 gives the natural transformer-style
//! shape
//!
//!     A : 32 x 4096
//!     B : 4096 x 16384
//!     C : 32 x 16384.
//!
//! To make the complete 20--30 sweep practical without spending O(mnp) time
//! merely constructing C, the driver generates a dense rank-one valid product:
//!
//!     A = u v^T,  B = s w^T,  C = <v,s> u w^T.
//!
//! This structure is used only to construct a correct witness.  The benchmarked
//! prover does not exploit the rank-one form: it receives materialized dense
//! evaluation tables and performs the same generic partial-MLE reductions,
//! sumcheck, commitments, and openings as for an arbitrary dense matrix.
//!
//! Timing convention
//! -----------------
//! Input generation and PCS setup/trim are excluded.  Prover total is
//!
//!     commitments(A,B,C)
//!       + application partial-MLE reduction
//!       + degree-2 sumcheck
//!       + PCS openings(A,B,C).
//!
//! Verifier total is sumcheck verification plus the three PCS verifications.
//!
//! CPU run example:
//!
//!   cargo bench -p benchmark --bench matmul_bench -- \
//!     --k 20..=30 \
//!     --systems basefold,brakedown,brakingbase,qapcs,quasar \
//!     --samples 5 --threads 32 \
//!     --quasar-log-rows 6 \
//!     --inverse-rate 2 --field-bits 127 \
//!     --security 100 --distance-failure 100 \
//!     --output ./bench_data/matmul/matmul_20_30.csv
//!
//! IMPORTANT ABOUT QUASAR-GPU
//! --------------------------
//! The repository's CUDA path lives in the separate `qa_gpu_overlay` crate and
//! uses the device-resident hybrid commitment path from `qa_commit_cpu_cuda`.
//! Do NOT substitute the encoding-only `GpuQaEncoder` here: doing so would copy
//! the full codeword back to the CPU and would benchmark a different system.
//! The CPU benchmark deliberately rejects `quasar-gpu`. Run the fifth backend
//! through `qa_gpu_overlay`'s `matmul_bench_gpu` binary; both drivers import the
//! same witness/application/sumcheck module from `matmul_common.rs`.

mod matmul_common;

use matmul_common::{
    application_points, build_evals, parse_exp_list, prove_product_sumcheck, seed32,
    verify_product_sumcheck, AppPrepared, BenchField, MatmulShape,
};
use plonkish_backend::{
    pcs::{
        multilinear::{
            qapcs::{MultilinearQAPCS, QAPCSSpecRateHalf100},
            quasar::{
                commit_and_write as quasar_commit_and_write, prove_qabase_open_full_two_layer_gkr,
                qabase_distance_lower_bound, qabase_queries_from_distance,
                qabase_split_evaluation_point, setup as quasar_setup, trim as quasar_trim,
                verify_qabase_open_full_two_layer_gkr, QABaseCommitment, QABaseProverParams,
                QABaseVerifierParams,
            },
            Basefold, BasefoldExtParams, MultilinearBrakedown, MultilinearBrakingBase,
        },
        PolynomialCommitmentScheme,
    },
    poly::{multilinear::MultilinearPolynomial, Polynomial},
    util::{
        arithmetic::{Field, PrimeField},
        code::BrakedownSpec,
        hash::{Blake2s, Output},
        transcript::{
            Blake2sTranscript, FieldTranscript, InMemoryTranscript, Transcript, TranscriptRead,
            TranscriptWrite,
        },
    },
};

use rand_chacha::{rand_core::SeedableRng, ChaCha8Rng};
use rayon::{current_num_threads, prelude::*, ThreadPoolBuilder};
use std::{
    env,
    fs::{create_dir_all, File, OpenOptions},
    hint::black_box,
    io::{Cursor, Write},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

type BenchHash = Blake2s;
type BenchTranscript = Blake2sTranscript<Cursor<Vec<u8>>>;

type BasefoldPcs = Basefold<BenchField, BenchHash, MatmulBasefoldRate2>;
type BrakedownPcs = MultilinearBrakedown<BenchField, BenchHash, BrakedownRate2>;
type BrakingBasePcs =
    MultilinearBrakingBase<BenchField, BenchHash, BrakedownRate2, MatmulBasefoldRate2>;
type QapcsPcs = MultilinearQAPCS<BenchField, BenchHash, QAPCSSpecRateHalf100>;

const DEFAULT_K_START: usize = 20;
const DEFAULT_K_END: usize = 30;
const DEFAULT_SAMPLES: usize = 5;
const DEFAULT_THREADS: usize = 32;
const DEFAULT_LOG_M: usize = 5; // m = 32
const DEFAULT_QUASAR_LOG_ROWS: usize = 6;
const DEFAULT_INVERSE_RATE: usize = 2;
const DEFAULT_FIELD_BITS: usize = 127;
const DEFAULT_SECURITY: usize = 100;
const DEFAULT_DISTANCE_FAILURE: usize = 100;
const DEFAULT_SEED: u8 = 113;
const HASH_BYTES: usize = 32;

// -----------------------------------------------------------------------------
// Concrete PCS parameters matching the paper comparison.
// -----------------------------------------------------------------------------

#[derive(Debug)]
struct MatmulBasefoldRate2;

impl BasefoldExtParams for MatmulBasefoldRate2 {
    fn get_reps() -> usize {
        // Same 100-bit setting used by the current Quasar/BaseFold backend.
        241
    }

    fn get_rate() -> usize {
        // BaseFold names this parameter "rate", but it is log2(inverse rate).
        // 1 therefore means inverse rate 2, i.e. code rate 1/2.
        1
    }

    fn get_basecode_rounds() -> usize {
        0
    }

    fn get_rs_basecode() -> bool {
        false
    }

    fn get_code_type() -> String {
        "random".to_owned()
    }
}

/// Brakedown2 / rate-1/2 parameters used in the current comparison.
#[derive(Debug)]
struct BrakedownRate2;

impl BrakedownSpec for BrakedownRate2 {
    const LAMBDA: f64 = 100.0;
    const ALPHA: f64 = 0.3;
    const BETA: f64 = 0.2;
    const R: f64 = 2.0;

    fn c_n(_n: usize) -> usize {
        11
    }

    fn d_n(_log2_q: usize, _n: usize) -> usize {
        22
    }

    fn num_column_opening() -> usize {
        2045
    }
}

// -----------------------------------------------------------------------------
// CLI and result records.
// -----------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum System {
    Basefold,
    Brakedown,
    BrakingBase,
    Qapcs,
    Quasar,
    QuasarGpu,
}

impl System {
    fn parse(name: &str) -> Self {
        match name.trim().to_ascii_lowercase().as_str() {
            "basefold" => Self::Basefold,
            "brakedown" => Self::Brakedown,
            "brakingbase" | "braking-base" | "braking_base" => Self::BrakingBase,
            "qapcs" => Self::Qapcs,
            "quasar" => Self::Quasar,
            "quasar-gpu" | "quasar_gpu" | "gpu" => Self::QuasarGpu,
            other => panic!("unknown system: {other}"),
        }
    }
}

#[derive(Clone, Debug)]
struct Args {
    k_values: Vec<usize>,
    systems: Vec<System>,
    samples: usize,
    threads: usize,
    log_m: usize,
    quasar_log_rows: usize,
    inverse_rate: usize,
    field_bits: usize,
    security_bits: usize,
    distance_failure_bits: usize,
    seed: u8,
    output: PathBuf,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            k_values: (DEFAULT_K_START..=DEFAULT_K_END).collect(),
            systems: vec![
                System::Basefold,
                System::Brakedown,
                System::BrakingBase,
                System::Qapcs,
                System::Quasar,
            ],
            samples: DEFAULT_SAMPLES,
            threads: DEFAULT_THREADS,
            log_m: DEFAULT_LOG_M,
            quasar_log_rows: DEFAULT_QUASAR_LOG_ROWS,
            inverse_rate: DEFAULT_INVERSE_RATE,
            field_bits: DEFAULT_FIELD_BITS,
            security_bits: DEFAULT_SECURITY,
            distance_failure_bits: DEFAULT_DISTANCE_FAILURE,
            seed: DEFAULT_SEED,
            output: PathBuf::from("./bench_data/matmul/matmul_20_30.csv"),
        }
    }
}

impl Args {
    fn parse() -> Self {
        let mut cfg = Self::default();
        let argv = env::args().collect::<Vec<_>>();
        let mut i = 1usize;

        while i < argv.len() {
            match argv[i].as_str() {
                "--bench" => {
                    if i + 1 < argv.len() && !argv[i + 1].starts_with('-') {
                        i += 1;
                    }
                }
                "--k" | "--total-k" => {
                    i += 1;
                    cfg.k_values = parse_exp_list(argv.get(i).expect("missing --k value"));
                }
                "--systems" => {
                    i += 1;
                    cfg.systems = argv
                        .get(i)
                        .expect("missing --systems value")
                        .split(',')
                        .filter(|x| !x.trim().is_empty())
                        .map(System::parse)
                        .collect();
                }
                "--samples" => {
                    i += 1;
                    cfg.samples = argv[i].parse().expect("invalid --samples");
                }
                "--threads" => {
                    i += 1;
                    cfg.threads = argv[i].parse().expect("invalid --threads");
                }
                "--log-m" => {
                    i += 1;
                    cfg.log_m = argv[i].parse().expect("invalid --log-m");
                }
                "--quasar-log-rows" => {
                    i += 1;
                    cfg.quasar_log_rows = argv[i].parse().expect("invalid --quasar-log-rows");
                }
                "--inverse-rate" => {
                    i += 1;
                    cfg.inverse_rate = argv[i].parse().expect("invalid --inverse-rate");
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
                    cfg.distance_failure_bits =
                        argv[i].parse().expect("invalid --distance-failure");
                }
                "--seed" => {
                    i += 1;
                    cfg.seed = argv[i].parse().expect("invalid --seed");
                }
                "--output" => {
                    i += 1;
                    cfg.output = PathBuf::from(argv.get(i).expect("missing --output value"));
                }
                "--smoke" => {
                    cfg.k_values = vec![12];
                    cfg.samples = 1;
                    cfg.threads = 8;
                }
                "--help" | "-h" => print_help_and_exit(),
                other => panic!("unknown argument: {other}"),
            }
            i += 1;
        }

        assert!(!cfg.k_values.is_empty());
        assert!(!cfg.systems.is_empty());
        assert!(cfg.samples >= 1);
        assert!(cfg.threads >= 1);
        assert!(cfg.log_m >= 1);
        assert!(cfg.quasar_log_rows >= 1);
        assert_eq!(cfg.inverse_rate, 2, "this comparison is fixed to rate 1/2");
        assert_eq!(
            cfg.field_bits, 127,
            "this benchmark is fixed to Mersenne127"
        );

        if cfg.systems.contains(&System::QuasarGpu) {
            panic!(
                "Quasar-GPU must be run from qa_gpu_overlay with the existing device-resident hybrid commitment API. \
                 Do not use the encoding-only GpuQaEncoder as a substitute. \
                 Run qa_gpu_overlay's matmul_bench_gpu binary for this backend."
            );
        }

        cfg
    }
}

#[derive(Clone, Debug)]
struct BenchResult {
    system: &'static str,
    k: usize,
    m: usize,
    n: usize,
    p: usize,
    log_a: usize,
    log_b: usize,
    log_c: usize,
    samples: usize,
    threads: usize,
    commit_ms: f64,
    app_prepare_ms: f64,
    sumcheck_prove_ms: f64,
    pcs_open_ms: f64,
    prover_total_ms: f64,
    sumcheck_verify_ms: f64,
    pcs_verify_ms: f64,
    verifier_total_ms: f64,
    pcs_proof_bytes: usize,
    sumcheck_bytes: usize,
    eval_claim_bytes: usize,
    proof_bytes_excl_input_commitments: usize,
    proof_bytes_incl_input_commitments: usize,
}

#[derive(Debug)]
struct MatmulInstance {
    shape: MatmulShape,
    a: MultilinearPolynomial<BenchField>,
    b: MultilinearPolynomial<BenchField>,
    c: MultilinearPolynomial<BenchField>,
}

fn build_instance(shape: MatmulShape, seed: [u8; 32]) -> MatmulInstance {
    let evals = build_evals(&shape, seed);
    MatmulInstance {
        shape,
        a: MultilinearPolynomial::new(evals.a),
        b: MultilinearPolynomial::new(evals.b),
        c: MultilinearPolynomial::new(evals.c),
    }
}

fn prepare_application(instance: &MatmulInstance, rng: ChaCha8Rng) -> AppPrepared {
    matmul_common::prepare_application(
        &instance.shape,
        instance.a.evals(),
        instance.b.evals(),
        instance.c.evals(),
        rng,
    )
}

// -----------------------------------------------------------------------------
// Standard PCS backends: BaseFold, Brakedown, BrakingBase, QAPCS.
// -----------------------------------------------------------------------------

fn bench_standard_pcs<Pcs>(
    name: &'static str,
    instance: &MatmulInstance,
    args: &Args,
) -> BenchResult
where
    Pcs: PolynomialCommitmentScheme<BenchField, Polynomial = MultilinearPolynomial<BenchField>>,
    BenchTranscript: TranscriptRead<Pcs::CommitmentChunk, BenchField>
        + TranscriptWrite<Pcs::CommitmentChunk, BenchField>,
{
    let a_len = instance.a.evals().len();
    let b_len = instance.b.evals().len();
    let c_len = instance.c.evals().len();

    let mut setup_rng_a = ChaCha8Rng::from_seed(seed32(args.seed, instance.shape.k, 0, 11));
    let mut setup_rng_b = ChaCha8Rng::from_seed(seed32(args.seed, instance.shape.k, 0, 12));
    let mut setup_rng_c = ChaCha8Rng::from_seed(seed32(args.seed, instance.shape.k, 0, 13));

    let param_a = Pcs::setup(a_len, 1, &mut setup_rng_a).expect("PCS setup(A) failed");
    let param_b = Pcs::setup(b_len, 1, &mut setup_rng_b).expect("PCS setup(B) failed");
    let param_c = Pcs::setup(c_len, 1, &mut setup_rng_c).expect("PCS setup(C) failed");

    let (pp_a, vp_a) = Pcs::trim(&param_a, a_len, 1).expect("PCS trim(A) failed");
    let (pp_b, vp_b) = Pcs::trim(&param_b, b_len, 1).expect("PCS trim(B) failed");
    let (pp_c, vp_c) = Pcs::trim(&param_c, c_len, 1).expect("PCS trim(C) failed");
    drop((param_a, param_b, param_c));

    let mut commit_times = Vec::with_capacity(args.samples);
    let mut app_times = Vec::with_capacity(args.samples);
    let mut sumcheck_prove_times = Vec::with_capacity(args.samples);
    let mut open_times = Vec::with_capacity(args.samples);
    let mut sumcheck_verify_times = Vec::with_capacity(args.samples);
    let mut pcs_verify_times = Vec::with_capacity(args.samples);
    let mut prover_total_times = Vec::with_capacity(args.samples);
    let mut verifier_total_times = Vec::with_capacity(args.samples);
    let mut last_pcs_proof_bytes = 0usize;
    let mut last_sumcheck_bytes = 0usize;

    // Bind the opening transcript to its public statement without serializing
    // those public values into the proof.
    let bind_statement = |transcript: &mut BenchTranscript,
                          commitment: &Pcs::Commitment,
                          point: &[BenchField],
                          eval: &BenchField| {
        <BenchTranscript as Transcript<Pcs::CommitmentChunk, BenchField>>::common_commitments(
            transcript,
            commitment.as_ref(),
        )
        .unwrap();
        <BenchTranscript as FieldTranscript<BenchField>>::common_field_elements(transcript, point)
            .unwrap();
        <BenchTranscript as FieldTranscript<BenchField>>::common_field_element(transcript, eval)
            .unwrap();
    };

    for sample in 0..args.samples {
        let prover_total_start = Instant::now();

        let commit_start = Instant::now();
        let comm_a = Pcs::commit(&pp_a, &instance.a).expect("PCS commit(A) failed");
        let comm_b = Pcs::commit(&pp_b, &instance.b).expect("PCS commit(B) failed");
        let comm_c = Pcs::commit(&pp_c, &instance.c).expect("PCS commit(C) failed");
        let commit_elapsed = commit_start.elapsed();
        commit_times.push(commit_elapsed);

        let coin_seed = seed32(args.seed, instance.shape.k, sample, 51);
        let app_start = Instant::now();
        let app = prepare_application(instance, ChaCha8Rng::from_seed(coin_seed));
        let app_elapsed = app_start.elapsed();
        app_times.push(app_elapsed);

        let sc_seed = seed32(args.seed, instance.shape.k, sample, 52);
        let sumcheck_start = Instant::now();
        let sc = prove_product_sumcheck(
            &app.a_y,
            &app.b_y,
            app.c_eval,
            ChaCha8Rng::from_seed(sc_seed),
        );
        let sumcheck_elapsed = sumcheck_start.elapsed();
        sumcheck_prove_times.push(sumcheck_elapsed);

        let (point_a, point_b, point_c) = application_points(&app, &sc.ry);
        assert_eq!(point_a.len(), instance.shape.log_a());
        assert_eq!(point_b.len(), instance.shape.log_b());
        assert_eq!(point_c.len(), instance.shape.log_c());

        let open_start = Instant::now();
        let mut tr_a = BenchTranscript::new(());
        bind_statement(&mut tr_a, &comm_a, &point_a, &sc.a_eval);
        Pcs::open(&pp_a, &instance.a, &comm_a, &point_a, &sc.a_eval, &mut tr_a)
            .expect("PCS open(A) failed");
        let proof_a = tr_a.into_proof();

        let mut tr_b = BenchTranscript::new(());
        bind_statement(&mut tr_b, &comm_b, &point_b, &sc.b_eval);
        Pcs::open(&pp_b, &instance.b, &comm_b, &point_b, &sc.b_eval, &mut tr_b)
            .expect("PCS open(B) failed");
        let proof_b = tr_b.into_proof();

        let mut tr_c = BenchTranscript::new(());
        bind_statement(&mut tr_c, &comm_c, &point_c, &app.c_eval);
        Pcs::open(
            &pp_c,
            &instance.c,
            &comm_c,
            &point_c,
            &app.c_eval,
            &mut tr_c,
        )
        .expect("PCS open(C) failed");
        let proof_c = tr_c.into_proof();
        let open_elapsed = open_start.elapsed();
        open_times.push(open_elapsed);

        let prover_total = prover_total_start.elapsed();
        prover_total_times.push(prover_total);

        last_pcs_proof_bytes = proof_a.len() + proof_b.len() + proof_c.len();
        last_sumcheck_bytes = product_sumcheck_bytes(sc.proof.rounds.len());

        // Verifier: standard sumcheck, then the three succinct MLE openings.
        let verifier_total_start = Instant::now();

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
        let sc_verify_elapsed = sc_verify_start.elapsed();
        sumcheck_verify_times.push(sc_verify_elapsed);

        let pcs_verify_start = Instant::now();
        let mut vr_a = BenchTranscript::from_proof((), proof_a.as_slice());
        bind_statement(&mut vr_a, &comm_a, &point_a, &sc.a_eval);
        Pcs::verify(&vp_a, &comm_a, &point_a, &sc.a_eval, &mut vr_a).expect("PCS verify(A) failed");

        let mut vr_b = BenchTranscript::from_proof((), proof_b.as_slice());
        bind_statement(&mut vr_b, &comm_b, &point_b, &sc.b_eval);
        Pcs::verify(&vp_b, &comm_b, &point_b, &sc.b_eval, &mut vr_b).expect("PCS verify(B) failed");

        let mut vr_c = BenchTranscript::from_proof((), proof_c.as_slice());
        bind_statement(&mut vr_c, &comm_c, &point_c, &app.c_eval);
        Pcs::verify(&vp_c, &comm_c, &point_c, &app.c_eval, &mut vr_c)
            .expect("PCS verify(C) failed");
        let pcs_verify_elapsed = pcs_verify_start.elapsed();
        pcs_verify_times.push(pcs_verify_elapsed);

        let verifier_total = verifier_total_start.elapsed();
        verifier_total_times.push(verifier_total);

        black_box((&comm_a, &comm_b, &comm_c));

        eprintln!(
            "[{name}] k={} sample={} dims={}x{} * {}x{}: commit={:.3} app={:.3} sc={:.3} open={:.3} prover={:.3} verify={:.3} proof={} B",
            instance.shape.k,
            sample,
            instance.shape.m,
            instance.shape.n,
            instance.shape.n,
            instance.shape.p,
            ms(commit_elapsed),
            ms(app_elapsed),
            ms(sumcheck_elapsed),
            ms(open_elapsed),
            ms(prover_total),
            ms(verifier_total),
            last_pcs_proof_bytes + last_sumcheck_bytes + 3 * field_bytes(),
        );
    }

    build_result(
        name,
        instance,
        args,
        &commit_times,
        &app_times,
        &sumcheck_prove_times,
        &open_times,
        &prover_total_times,
        &sumcheck_verify_times,
        &pcs_verify_times,
        &verifier_total_times,
        last_pcs_proof_bytes,
        last_sumcheck_bytes,
    )
}

// -----------------------------------------------------------------------------
// Quasar CPU backend.
// -----------------------------------------------------------------------------

struct QuasarKeys {
    pp: QABaseProverParams<BenchField, BenchHash>,
    vp: QABaseVerifierParams<BenchField, BenchHash>,
}

fn make_quasar_keys(total_log: usize, args: &Args, domain_tag: usize) -> QuasarKeys {
    assert!(total_log >= args.quasar_log_rows);
    let row_log = total_log - args.quasar_log_rows;
    let num_rows = 1usize << args.quasar_log_rows;
    let row_size = 1usize << row_log;

    let delta = qabase_distance_lower_bound(
        row_log,
        args.inverse_rate,
        args.field_bits,
        args.distance_failure_bits,
    );
    let queries = qabase_queries_from_distance(delta, args.security_bits);

    let mut rng = ChaCha8Rng::from_seed(seed32(args.seed, total_log, domain_tag, 71));
    let param = quasar_setup::<BenchField, BenchHash>(
        row_size,
        1,
        &mut rng,
        Some(num_rows),
        Some(args.inverse_rate),
        Some(queries),
    );
    let (pp, vp) = quasar_trim::<BenchField, BenchHash>(&param, row_size, 1);
    drop(param);

    QuasarKeys { pp, vp }
}

fn reshape_for_quasar(
    poly: &MultilinearPolynomial<BenchField>,
    log_rows: usize,
) -> Vec<Vec<BenchField>> {
    let evals = poly.evals();
    let num_rows = 1usize << log_rows;
    assert_eq!(evals.len() % num_rows, 0);
    let row_len = evals.len() / num_rows;
    assert!(row_len.is_power_of_two());
    evals
        .par_chunks(row_len)
        .map(|row| row.to_vec())
        .collect::<Vec<_>>()
}

fn quasar_open_one(
    keys: &QuasarKeys,
    rows: &[Vec<BenchField>],
    comm: &QABaseCommitment<BenchField, BenchHash>,
    full_point: &[BenchField],
    claimed: BenchField,
) -> (Vec<u8>, Duration) {
    let row_log = keys.pp.num_vars;
    let (z_left, z_right) =
        qabase_split_evaluation_point::<BenchField>(full_point, keys.pp.num_rows, row_log);

    let root: &Output<BenchHash> = comm.as_ref();
    let mut transcript = BenchTranscript::new(());
    <BenchTranscript as TranscriptWrite<Output<BenchHash>, BenchField>>::write_commitment(
        &mut transcript,
        root,
    )
    .expect("failed to write Quasar root");

    let start = Instant::now();
    let out = prove_qabase_open_full_two_layer_gkr::<BenchField, BenchHash>(
        &keys.pp,
        rows,
        comm,
        z_left,
        z_right,
        claimed,
        &mut transcript,
    )
    .expect("Quasar opening prover failed");
    assert!(out.ok_eval_value);
    let elapsed = start.elapsed();
    (transcript.into_proof(), elapsed)
}

fn quasar_verify_one(
    keys: &QuasarKeys,
    comm: &QABaseCommitment<BenchField, BenchHash>,
    full_point: &[BenchField],
    claimed: BenchField,
    proof: &[u8],
) -> Duration {
    let row_log = keys.vp.num_vars;
    let (z_left, z_right) =
        qabase_split_evaluation_point::<BenchField>(full_point, keys.vp.num_rows, row_log);

    let mut transcript = BenchTranscript::from_proof((), proof);
    let start = Instant::now();
    let (ok, _) = verify_qabase_open_full_two_layer_gkr::<BenchField, BenchHash>(
        &keys.vp,
        comm,
        z_left,
        z_right,
        claimed,
        &mut transcript,
    )
    .expect("Quasar verifier errored");
    assert!(
        ok,
        "Quasar verifier rejected a valid matrix-multiplication opening"
    );
    start.elapsed()
}

fn bench_quasar(instance: &MatmulInstance, args: &Args) -> BenchResult {
    let keys_a = make_quasar_keys(instance.shape.log_a(), args, 1);
    let keys_b = make_quasar_keys(instance.shape.log_b(), args, 2);
    let keys_c = make_quasar_keys(instance.shape.log_c(), args, 3);

    // Reshaping is an adapter/data-layout cost, not cryptographic proving work.
    // Do it once outside all measured samples.
    let rows_a = reshape_for_quasar(&instance.a, args.quasar_log_rows);
    let rows_b = reshape_for_quasar(&instance.b, args.quasar_log_rows);
    let rows_c = reshape_for_quasar(&instance.c, args.quasar_log_rows);

    let mut commit_times = Vec::with_capacity(args.samples);
    let mut app_times = Vec::with_capacity(args.samples);
    let mut sumcheck_prove_times = Vec::with_capacity(args.samples);
    let mut open_times = Vec::with_capacity(args.samples);
    let mut sumcheck_verify_times = Vec::with_capacity(args.samples);
    let mut pcs_verify_times = Vec::with_capacity(args.samples);
    let mut prover_total_times = Vec::with_capacity(args.samples);
    let mut verifier_total_times = Vec::with_capacity(args.samples);
    let mut last_pcs_proof_bytes = 0usize;
    let mut last_sumcheck_bytes = 0usize;

    for sample in 0..args.samples {
        let prover_total_start = Instant::now();

        let commit_start = Instant::now();
        let mut ct_a = BenchTranscript::new(());
        let comm_a =
            quasar_commit_and_write::<BenchField, BenchHash>(&keys_a.pp, &rows_a, &mut ct_a);
        let mut ct_b = BenchTranscript::new(());
        let comm_b =
            quasar_commit_and_write::<BenchField, BenchHash>(&keys_b.pp, &rows_b, &mut ct_b);
        let mut ct_c = BenchTranscript::new(());
        let comm_c =
            quasar_commit_and_write::<BenchField, BenchHash>(&keys_c.pp, &rows_c, &mut ct_c);
        let commit_elapsed = commit_start.elapsed();
        commit_times.push(commit_elapsed);
        drop((ct_a, ct_b, ct_c));

        let coin_seed = seed32(args.seed, instance.shape.k, sample, 51);
        let app_start = Instant::now();
        let app = prepare_application(instance, ChaCha8Rng::from_seed(coin_seed));
        let app_elapsed = app_start.elapsed();
        app_times.push(app_elapsed);

        let sc_seed = seed32(args.seed, instance.shape.k, sample, 52);
        let sc_start = Instant::now();
        let sc = prove_product_sumcheck(
            &app.a_y,
            &app.b_y,
            app.c_eval,
            ChaCha8Rng::from_seed(sc_seed),
        );
        let sc_elapsed = sc_start.elapsed();
        sumcheck_prove_times.push(sc_elapsed);

        let (point_a, point_b, point_c) = application_points(&app, &sc.ry);

        let (proof_a, open_a) = quasar_open_one(&keys_a, &rows_a, &comm_a, &point_a, sc.a_eval);
        let (proof_b, open_b) = quasar_open_one(&keys_b, &rows_b, &comm_b, &point_b, sc.b_eval);
        let (proof_c, open_c) = quasar_open_one(&keys_c, &rows_c, &comm_c, &point_c, app.c_eval);
        let open_elapsed = open_a + open_b + open_c;
        open_times.push(open_elapsed);

        let prover_total = prover_total_start.elapsed();
        prover_total_times.push(prover_total);

        // Each Quasar opening transcript begins with the corresponding input
        // commitment root because the custom verifier reads it from the
        // transcript.  Strip those three 32-byte roots here so the common
        // accounting below treats input commitments uniformly across all PCSs.
        // `proof_bytes_incl_input_commitments` adds the three roots back once.
        let raw_quasar_opening_bytes = proof_a.len() + proof_b.len() + proof_c.len();
        assert!(raw_quasar_opening_bytes >= 3 * HASH_BYTES);
        last_pcs_proof_bytes = raw_quasar_opening_bytes - 3 * HASH_BYTES;
        last_sumcheck_bytes = product_sumcheck_bytes(sc.proof.rounds.len());

        let verifier_total_start = Instant::now();

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
        let sc_verify_elapsed = sc_verify_start.elapsed();
        sumcheck_verify_times.push(sc_verify_elapsed);

        let v_a = quasar_verify_one(&keys_a, &comm_a, &point_a, sc.a_eval, &proof_a);
        let v_b = quasar_verify_one(&keys_b, &comm_b, &point_b, sc.b_eval, &proof_b);
        let v_c = quasar_verify_one(&keys_c, &comm_c, &point_c, app.c_eval, &proof_c);
        let pcs_verify_elapsed = v_a + v_b + v_c;
        pcs_verify_times.push(pcs_verify_elapsed);

        let verifier_total = verifier_total_start.elapsed();
        verifier_total_times.push(verifier_total);

        black_box((&comm_a, &comm_b, &comm_c));

        eprintln!(
            "[Quasar] k={} sample={} dims={}x{} * {}x{}: commit={:.3} app={:.3} sc={:.3} open={:.3} prover={:.3} verify={:.3} proof={} B",
            instance.shape.k,
            sample,
            instance.shape.m,
            instance.shape.n,
            instance.shape.n,
            instance.shape.p,
            ms(commit_elapsed),
            ms(app_elapsed),
            ms(sc_elapsed),
            ms(open_elapsed),
            ms(prover_total),
            ms(verifier_total),
            last_pcs_proof_bytes + last_sumcheck_bytes + 3 * field_bytes(),
        );
    }

    build_result(
        "Quasar",
        instance,
        args,
        &commit_times,
        &app_times,
        &sumcheck_prove_times,
        &open_times,
        &prover_total_times,
        &sumcheck_verify_times,
        &pcs_verify_times,
        &verifier_total_times,
        last_pcs_proof_bytes,
        last_sumcheck_bytes,
    )
}

// -----------------------------------------------------------------------------
// Main driver.
// -----------------------------------------------------------------------------

fn main() {
    let args = Args::parse();

    if let Some(parent) = args.output.parent() {
        create_dir_all(parent).expect("failed to create output directory");
    }
    write_header_if_new(&args.output);

    env::set_var("RAYON_NUM_THREADS", args.threads.to_string());
    let pool = ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build()
        .expect("failed to construct Rayon thread pool");

    eprintln!("matrix-multiplication benchmark config: {args:?}");
    eprintln!("CSV: {}", args.output.display());

    pool.install(|| {
        eprintln!("rayon current_num_threads = {}", current_num_threads());

        for &k in &args.k_values {
            let shape = MatmulShape::from_k(k, args.log_m);
            eprintln!(
                "\n=== k={k}: A={}x{} (2^{}), B={}x{} (2^{}), C={}x{} (2^{}) ===",
                shape.m,
                shape.n,
                shape.log_a(),
                shape.n,
                shape.p,
                shape.log_b(),
                shape.m,
                shape.p,
                shape.log_c(),
            );

            let instance_seed = seed32(args.seed, k, usize::MAX - 1, 31);
            let instance = build_instance(shape, instance_seed);

            for system in &args.systems {
                let result = match system {
                    System::Basefold => {
                        bench_standard_pcs::<BasefoldPcs>("BaseFold", &instance, &args)
                    }
                    System::Brakedown => {
                        bench_standard_pcs::<BrakedownPcs>("Brakedown", &instance, &args)
                    }
                    System::BrakingBase => {
                        bench_standard_pcs::<BrakingBasePcs>("BrakingBase", &instance, &args)
                    }
                    System::Qapcs => bench_standard_pcs::<QapcsPcs>("QAPCS", &instance, &args),
                    System::Quasar => bench_quasar(&instance, &args),
                    System::QuasarGpu => unreachable!(),
                };

                print_result(&result);
                append_result(&args.output, &result);
            }

            drop(instance);
        }
    });
}

// -----------------------------------------------------------------------------
// Utilities / CSV.
// -----------------------------------------------------------------------------

fn build_result(
    name: &'static str,
    instance: &MatmulInstance,
    args: &Args,
    commit_times: &[Duration],
    app_times: &[Duration],
    sumcheck_prove_times: &[Duration],
    open_times: &[Duration],
    prover_total_times: &[Duration],
    sumcheck_verify_times: &[Duration],
    pcs_verify_times: &[Duration],
    verifier_total_times: &[Duration],
    pcs_proof_bytes: usize,
    sumcheck_bytes: usize,
) -> BenchResult {
    let commit_ms = ms(avg_after_warmup(commit_times));
    let app_prepare_ms = ms(avg_after_warmup(app_times));
    let sumcheck_prove_ms = ms(avg_after_warmup(sumcheck_prove_times));
    let pcs_open_ms = ms(avg_after_warmup(open_times));
    let prover_total_ms = ms(avg_after_warmup(prover_total_times));
    let sumcheck_verify_ms = ms(avg_after_warmup(sumcheck_verify_times));
    let pcs_verify_ms = ms(avg_after_warmup(pcs_verify_times));
    let verifier_total_ms = ms(avg_after_warmup(verifier_total_times));

    // The three final field evaluations A(r_x,r_y), B(r_y,r_z), C(r_x,r_z)
    // are application messages and must be counted once in addition to the PCS
    // transcript bytes.  Input commitments are reported separately as 3x32 B.
    let eval_claim_bytes = 3 * field_bytes();
    let proof_excl = pcs_proof_bytes + sumcheck_bytes + eval_claim_bytes;
    let proof_incl = proof_excl + 3 * HASH_BYTES;

    BenchResult {
        system: name,
        k: instance.shape.k,
        m: instance.shape.m,
        n: instance.shape.n,
        p: instance.shape.p,
        log_a: instance.shape.log_a(),
        log_b: instance.shape.log_b(),
        log_c: instance.shape.log_c(),
        samples: args.samples,
        threads: current_num_threads(),
        commit_ms,
        app_prepare_ms,
        sumcheck_prove_ms,
        pcs_open_ms,
        prover_total_ms,
        sumcheck_verify_ms,
        pcs_verify_ms,
        verifier_total_ms,
        pcs_proof_bytes,
        sumcheck_bytes,
        eval_claim_bytes,
        proof_bytes_excl_input_commitments: proof_excl,
        proof_bytes_incl_input_commitments: proof_incl,
    }
}

fn field_bytes() -> usize {
    BenchField::ZERO.to_repr().as_ref().len()
}

fn product_sumcheck_bytes(rounds: usize) -> usize {
    3 * rounds * field_bytes()
}

fn avg_after_warmup(times: &[Duration]) -> Duration {
    assert!(!times.is_empty());
    let start = if times.len() >= 5 { 2 } else { 0 };
    let sum = times[start..]
        .iter()
        .copied()
        .fold(Duration::ZERO, |acc, t| acc + t);
    sum / ((times.len() - start) as u32)
}

fn ms(t: Duration) -> f64 {
    t.as_secs_f64() * 1_000.0
}

fn write_header_if_new(path: &Path) {
    if !path.exists() {
        let mut file = File::create(path).expect("failed to create CSV");
        writeln!(
            file,
            "system,k,m,n,p,log_a,log_b,log_c,samples,threads,commit_ms,app_prepare_ms,sumcheck_prove_ms,pcs_open_ms,prover_total_ms,sumcheck_verify_ms,pcs_verify_ms,verifier_total_ms,pcs_proof_bytes,sumcheck_bytes,eval_claim_bytes,proof_bytes_excl_input_commitments,proof_bytes_incl_input_commitments,proof_kb_excl_input_commitments,proof_kb_incl_input_commitments"
        )
        .unwrap();
    }
}

fn append_result(path: &Path, r: &BenchResult) {
    let mut file = OpenOptions::new()
        .append(true)
        .open(path)
        .expect("failed to open CSV");
    writeln!(
        file,
        "{},{},{},{},{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{},{},{},{},{:.3},{:.3}",
        r.system,
        r.k,
        r.m,
        r.n,
        r.p,
        r.log_a,
        r.log_b,
        r.log_c,
        r.samples,
        r.threads,
        r.commit_ms,
        r.app_prepare_ms,
        r.sumcheck_prove_ms,
        r.pcs_open_ms,
        r.prover_total_ms,
        r.sumcheck_verify_ms,
        r.pcs_verify_ms,
        r.verifier_total_ms,
        r.pcs_proof_bytes,
        r.sumcheck_bytes,
        r.eval_claim_bytes,
        r.proof_bytes_excl_input_commitments,
        r.proof_bytes_incl_input_commitments,
        r.proof_bytes_excl_input_commitments as f64 / 1024.0,
        r.proof_bytes_incl_input_commitments as f64 / 1024.0,
    )
    .unwrap();
}

fn print_result(r: &BenchResult) {
    println!(
        "{} k={} [{}x{} * {}x{}] prover={:.3} ms (commit {:.3}, app {:.3}, sc {:.3}, open {:.3}) verifier={:.3} ms proof={:.2} KB",
        r.system,
        r.k,
        r.m,
        r.n,
        r.n,
        r.p,
        r.prover_total_ms,
        r.commit_ms,
        r.app_prepare_ms,
        r.sumcheck_prove_ms,
        r.pcs_open_ms,
        r.verifier_total_ms,
        r.proof_bytes_excl_input_commitments as f64 / 1024.0,
    );
}

fn print_help_and_exit() -> ! {
    eprintln!(
        "Verifiable matrix-multiplication benchmark\n\n\
         CPU sweep:\n\
           cargo bench -p benchmark --bench matmul_bench -- \\\n             --k 20..=30 \\\n             --systems basefold,brakedown,qapcs,quasar \\\n             --samples 5 --threads 32 \\\n             --quasar-log-rows 6 \\\n             --inverse-rate 2 --field-bits 127 \\\n             --security 100 --distance-failure 100\n\n\
         Options:\n\
           --k LIST                  exponents, e.g. 20..=30 or 24,26,28\n\
           --systems LIST            basefold,brakedown,brakingbase,qapcs,quasar\n\
           --samples N               measured repetitions (default 5)\n\
           --threads N               Rayon workers (default 32)\n\
           --log-m K                 m=2^K (default 5 => 32)\n\
           --quasar-log-rows K       internal Quasar rows=2^K (default 6)\n\
           --output PATH             CSV output path\n\
           --smoke                   k=12, one sample, 8 threads\n\n\
         Quasar-GPU is intentionally not emulated in this CPU target. Run the\n\
         qa_gpu_overlay matmul_bench_gpu binary for the device-resident path."
    );
    std::process::exit(0)
}
