//! Encoding-only comparison for QA, CUDA QA, RAA, Brakedown, and BaseFold.
//!
//! The non-QA call sites follow hadasz/plonkish_basefold commit
//! c29f08653121637b456e4705d901d078cd6fc668:
//! - RAA: `util::code::encode_bits_ser`, configured at rate 1/2 as in the
//!   QA-code paper's encoding-time experiment.
//! - Brakedown: the paper's Brakedown2 parameters (inverse rate 2,
//!   alpha=0.3, c_n=11, d_n=22) and `LinearCodes::encode`.
//! - BaseFold: the interpolation and foldable-domain evaluation performed by
//!   `Basefold::commit`, stopping before Merkle hashing.
//!
//! Run from the `plonkish/` workspace with:
//!
//! cargo run --release --features cuda \
//!   --manifest-path qa_gpu_overlay/Cargo.toml \
//!   --bin encoding_comparison -- \
//!   --log-min 12 --log-max 20 --repetitions 10 --warmups 5 --threads 32 \
//!   --output qa_gpu_overlay/qa_serial_parallel_gpu_12_20.csv
//!
//! The QA section reports both a serial single-codeword baseline and a
//! parallel single-codeword implementation.  The latter parallelizes each WHT
//! stage and the pointwise products without splitting the message into several
//! independent codewords.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("encoding_comparison requires --features cuda");
    std::process::exit(2);
}

#[cfg(feature = "cuda")]
mod with_cuda {
    use plonkish_backend::{
        pcs::{
            multilinear::{
                evaluate_over_foldable_domain, interpolate_over_boolean_hypercube_with_copy,
                Basefold, BasefoldExtParams, Type2Polynomial,
            },
            PolynomialCommitmentScheme,
        },
        util::{
            arithmetic::{Field, PrimeField},
            avx_int_types::u64::Blazeu64,
            code::{encode_bits_ser, Brakedown, BrakedownSpec, LinearCodes, Permutation},
            hash::Blake2s,
            new_fields::Mersenne127,
        },
    };
    use qa_gpu_overlay::{
        cpu::{qa_encode_cpu_parallel_single_row, qa_encode_cpu_serial_single_row},
        gpu::{GpuQaEncoder, GpuQaTiming},
    };
    use rand_chacha::{
        rand_core::{RngCore, SeedableRng},
        ChaCha8Rng,
    };
    use rayon::ThreadPoolBuilder;
    use std::{
        env,
        fs::{self, File},
        hint::black_box,
        io::{BufWriter, Write},
        path::{Path, PathBuf},
        process::ExitCode,
        time::{Duration, Instant},
    };

    const PUBLIC_REPOSITORY_REVISION: &str = "c29f08653121637b456e4705d901d078cd6fc668";

    #[derive(Clone, Debug)]
    struct Config {
        log_min: usize,
        log_max: usize,
        repetitions: usize,
        warmups: usize,
        threads: usize,
        qa_inverse_rate: usize,
        raa_inverse_rate: usize,
        basefold_inverse_rate: usize,
        qa_rows: usize,
        gpu_batch_rows: usize,
        qa_only: bool,
        seed: u64,
        output: PathBuf,
    }

    impl Default for Config {
        fn default() -> Self {
            Self {
                log_min: 12,
                log_max: 20,
                repetitions: 10,
                warmups: 1,
                threads: 32,
                qa_inverse_rate: 2,
                // The encoding-time comparison in the QA-code paper uses rate 1/2.
                // Its cited distance analysis, in contrast, requires inverse rate >= 4.
                raa_inverse_rate: 2,
                basefold_inverse_rate: 2,
                qa_rows: 1,
                gpu_batch_rows: 8,
                qa_only: false,
                seed: 0x5155_4153_4152,
                output: PathBuf::from("qa_gpu_overlay/encoding_comparison_12_20.csv"),
            }
        }
    }

    fn usage() -> &'static str {
        "encoding_comparison [options]\n\
         \n\
         --log-min K                 first input size is 2^K (default 12)\n\
         --log-max K                 last input size is 2^K (default 20)\n\
         --repetitions N             measured repetitions (default 10)\n\
         --warmups N                 unmeasured repetitions (default 1)\n\
         --threads N                 Rayon workers for single-row parallel QA (default 32)\n\
         --qa-inverse-rate N         QA/QA-GPU inverse rate (default 2)\n\
         --raa-inverse-rate N        RAA inverse rate (default 2)\n\
         --basefold-inverse-rate N   BaseFold inverse rate (currently 2 only)\n\
         --qa-rows N                 must be 1 for the fair CPU comparison (default 1)\n\
         --gpu-batch-rows N          maximum CUDA row batch (default 8)\n\
         --qa-only                   benchmark only QA serial, QA parallel, and QA-GPU\n\
         --seed N                    deterministic seed (default fixed)\n\
         --output PATH               CSV output path\n\
         -h, --help                  show this help"
    }

    fn next_usize(name: &str, args: &mut impl Iterator<Item = String>) -> Result<usize, String> {
        args.next()
            .ok_or_else(|| format!("missing value for {name}"))?
            .parse::<usize>()
            .map_err(|error| format!("invalid value for {name}: {error}"))
    }

    fn parse_config() -> Result<Option<Config>, String> {
        let mut config = Config::default();
        let mut args = env::args().skip(1);
        while let Some(argument) = args.next() {
            match argument.as_str() {
                "-h" | "--help" => return Ok(None),
                "--log-min" => config.log_min = next_usize(&argument, &mut args)?,
                "--log-max" => config.log_max = next_usize(&argument, &mut args)?,
                "--repetitions" => config.repetitions = next_usize(&argument, &mut args)?,
                "--warmups" => config.warmups = next_usize(&argument, &mut args)?,
                "--threads" => config.threads = next_usize(&argument, &mut args)?,
                "--qa-inverse-rate" => config.qa_inverse_rate = next_usize(&argument, &mut args)?,
                "--raa-inverse-rate" => config.raa_inverse_rate = next_usize(&argument, &mut args)?,
                "--basefold-inverse-rate" => {
                    config.basefold_inverse_rate = next_usize(&argument, &mut args)?
                }
                "--qa-rows" => config.qa_rows = next_usize(&argument, &mut args)?,
                "--gpu-batch-rows" => config.gpu_batch_rows = next_usize(&argument, &mut args)?,
                "--qa-only" => config.qa_only = true,
                "--seed" => {
                    config.seed = args
                        .next()
                        .ok_or_else(|| "missing value for --seed".to_owned())?
                        .parse::<u64>()
                        .map_err(|error| format!("invalid value for --seed: {error}"))?;
                }
                "--output" => {
                    config.output = PathBuf::from(
                        args.next()
                            .ok_or_else(|| "missing value for --output".to_owned())?,
                    )
                }
                _ => return Err(format!("unknown argument {argument}\n\n{}", usage())),
            }
        }

        if config.log_min > config.log_max || config.log_max >= usize::BITS as usize {
            return Err("require 0 <= log-min <= log-max < usize::BITS".to_owned());
        }
        if config.repetitions == 0 || config.qa_rows == 0 {
            return Err("repetitions and qa-rows must be non-zero".to_owned());
        }
        if config.threads == 0 {
            return Err("threads must be non-zero".to_owned());
        }
        for (name, rate) in [
            ("QA", config.qa_inverse_rate),
            ("RAA", config.raa_inverse_rate),
            ("BaseFold", config.basefold_inverse_rate),
        ] {
            if rate < 2 || !rate.is_power_of_two() {
                return Err(format!("{name} inverse rate must be a power of two >= 2"));
            }
        }
        if config.basefold_inverse_rate != 2 {
            return Err(
                "this benchmark currently supports BaseFold inverse rate 2 only".to_owned(),
            );
        }
        if !config.qa_rows.is_power_of_two() || config.qa_rows > (1usize << config.log_min) {
            return Err("qa-rows must be a power of two no larger than 2^log-min".to_owned());
        }
        if config.qa_rows != 1 {
            return Err(
                "fair comparison requires --qa-rows 1 so QA, RAA, and BaseFold each encode one message/codeword"
                    .to_owned(),
            );
        }
        if config.gpu_batch_rows == 0 {
            return Err("gpu-batch-rows must be non-zero".to_owned());
        }
        Ok(Some(config))
    }

    #[derive(Clone, Debug)]
    struct Stats {
        mean_ms: f64,
        median_ms: f64,
        min_ms: f64,
        max_ms: f64,
        cv_percent: f64,
    }

    impl Stats {
        fn from_durations(samples: &[Duration]) -> Self {
            let mut milliseconds = samples
                .iter()
                .map(|sample| sample.as_secs_f64() * 1e3)
                .collect::<Vec<_>>();
            milliseconds.sort_by(f64::total_cmp);
            let mean_ms = milliseconds.iter().sum::<f64>() / milliseconds.len() as f64;
            let median_ms = if milliseconds.len() % 2 == 0 {
                let upper = milliseconds.len() / 2;
                (milliseconds[upper - 1] + milliseconds[upper]) / 2.0
            } else {
                milliseconds[milliseconds.len() / 2]
            };
            let variance = milliseconds
                .iter()
                .map(|sample| (sample - mean_ms).powi(2))
                .sum::<f64>()
                / milliseconds.len() as f64;
            Self {
                mean_ms,
                median_ms,
                min_ms: milliseconds[0],
                max_ms: *milliseconds.last().unwrap(),
                cv_percent: if mean_ms == 0.0 {
                    0.0
                } else {
                    100.0 * variance.sqrt() / mean_ms
                },
            }
        }
    }

    #[derive(Clone, Debug, Default)]
    struct GpuBreakdown {
        h2d_mean_ms: f64,
        kernel_mean_ms: f64,
        d2h_mean_ms: f64,
        cuda_total_mean_ms: f64,
    }

    impl GpuBreakdown {
        fn from_samples(samples: &[GpuQaTiming]) -> Self {
            let divisor = samples.len() as f64;
            let sum =
                |select: fn(&GpuQaTiming) -> f64| samples.iter().map(select).sum::<f64>() / divisor;
            Self {
                h2d_mean_ms: sum(|sample| sample.host_to_device_ms),
                kernel_mean_ms: sum(|sample| {
                    sample.device_input_copy_ms
                        + sample.first_wht_ms
                        + sample.scaling_ms
                        + sample.second_wht_ms
                        + sample.assemble_ms
                }),
                d2h_mean_ms: sum(|sample| sample.device_to_host_ms),
                cuda_total_mean_ms: sum(|sample| sample.total_cuda_ms),
            }
        }
    }

    #[derive(Clone, Debug)]
    struct Record<'a> {
        protocol: &'a str,
        field: &'a str,
        log_input: usize,
        message_elements: usize,
        codeword_elements: usize,
        setup: Duration,
        stats: Stats,
        gpu: Option<GpuBreakdown>,
        notes: &'a str,
    }

    fn timed_samples(
        mut operation: impl FnMut(),
        warmups: usize,
        repetitions: usize,
    ) -> Vec<Duration> {
        for _ in 0..warmups {
            operation();
        }
        (0..repetitions)
            .map(|_| {
                let start = Instant::now();
                operation();
                start.elapsed()
            })
            .collect()
    }

    fn csv_escape(value: &str) -> String {
        if value.contains(|character| matches!(character, ',' | '"' | '\n' | '\r')) {
            format!("\"{}\"", value.replace('"', "\"\""))
        } else {
            value.to_owned()
        }
    }

    fn write_record(
        output: &mut BufWriter<File>,
        config: &Config,
        record: &Record<'_>,
    ) -> Result<(), String> {
        let inverse_rate = record.codeword_elements as f64 / record.message_elements as f64;
        let ns_per_input = record.stats.mean_ms * 1e6 / record.message_elements as f64;
        let execution_mode = match record.protocol {
            "QA-CPU-Parallel" => "parallel-single-codeword",
            "QA-GPU" => "gpu",
            "BaseFold" if config.threads > 1 => "parallel",
            _ => "serial",
        };
        let reported_threads = match execution_mode {
            "serial" => 1,
            "gpu" => 0,
            _ => config.threads,
        };
        let (h2d, kernel, d2h, cuda_total) = record.gpu.as_ref().map_or_else(
            || (String::new(), String::new(), String::new(), String::new()),
            |gpu| {
                (
                    format!("{:.6}", gpu.h2d_mean_ms),
                    format!("{:.6}", gpu.kernel_mean_ms),
                    format!("{:.6}", gpu.d2h_mean_ms),
                    format!("{:.6}", gpu.cuda_total_mean_ms),
                )
            },
        );
        writeln!(
            output,
            "{},{},{},{},{},{:.9},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.4},{:.6},{},{},{},{},{},{},{}",
            record.protocol,
            csv_escape(record.field),
            record.log_input,
            record.message_elements,
            record.codeword_elements,
            inverse_rate,
            reported_threads,
            execution_mode,
            config.warmups,
            config.repetitions,
            record.setup.as_secs_f64() * 1e3,
            record.stats.mean_ms,
            record.stats.median_ms,
            record.stats.min_ms,
            record.stats.max_ms,
            record.stats.cv_percent,
            ns_per_input,
            h2d,
            kernel,
            d2h,
            cuda_total,
            PUBLIC_REPOSITORY_REVISION,
            csv_escape(record.notes),
        )
        .map_err(|error| format!("failed to write CSV: {error}"))
    }

    fn print_record(record: &Record<'_>) {
        println!(
            "  {:10} mean={:10.3} ms  median={:10.3} ms  CV={:6.2}%  codeword={} ({:.3}x)",
            record.protocol,
            record.stats.mean_ms,
            record.stats.median_ms,
            record.stats.cv_percent,
            record.codeword_elements,
            record.codeword_elements as f64 / record.message_elements as f64,
        );
    }

    #[derive(Debug)]
    struct BasefoldRate2;

    impl BasefoldExtParams for BasefoldRate2 {
        fn get_reps() -> usize {
            100
        }

        // The public API calls this `rate`, but it is log2(inverse rate).
        // The runtime check below restricts this concrete type to inverse rate 2.
        fn get_rate() -> usize {
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

    type ConcreteBasefold = Basefold<Mersenne127, Blake2s, BasefoldRate2>;

    /// Brakedown2 (B2) from the QA-code paper's encoding-time comparison.
    ///
    /// The pinned plonkish_basefold revision only exports the six original
    /// GLSTW21 parameter sets, whose largest inverse rate is about 1.72.  The
    /// BrakedownSpec trait is public, so we instantiate the improved rate-1/2
    /// parameters used by the paper directly: alpha=0.3, c_n=11, and d_n=22.
    #[derive(Debug)]
    struct BrakedownRate2;

    impl BrakedownSpec for BrakedownRate2 {
        const LAMBDA: f64 = 128.0;
        const ALPHA: f64 = 0.3;
        const BETA: f64 = 0.2;
        const R: f64 = 2.0;

        fn c_n(_n: usize) -> usize {
            11
        }

        fn d_n(_log2_q: usize, _n: usize) -> usize {
            22
        }
    }

    fn benchmark_size(
        config: &Config,
        log_input: usize,
        output: &mut BufWriter<File>,
        rng: &mut ChaCha8Rng,
    ) -> Result<(), String> {
        let input_len = 1usize << log_input;
        let qa_row_len = input_len / config.qa_rows;
        println!(
            "input=2^{log_input} ({input_len} elements), QA rows={}",
            config.qa_rows
        );

        let field_input = (0..input_len)
            .map(|_| <Mersenne127 as Field>::random(&mut *rng))
            .collect::<Vec<_>>();
        let raa_input = (0..input_len)
            .map(|_| Blazeu64 {
                value: rng.next_u64(),
            })
            .collect::<Vec<_>>();

        // QA setup is public randomness generation and is intentionally outside encoding time.
        let qa_setup_start = Instant::now();
        let qa_params =
            plonkish_backend::pcs::multilinear::quasar::QAParams::<Mersenne127>::new_random(
                qa_row_len,
                config.qa_inverse_rate,
                &mut *rng,
            );
        let qa_setup = qa_setup_start.elapsed();

        let qa_serial_samples = timed_samples(
            || {
                let codeword = qa_encode_cpu_serial_single_row(&field_input, &qa_params);
                black_box(codeword.as_slice());
            },
            config.warmups,
            config.repetitions,
        );
        let qa_serial_reference = qa_encode_cpu_serial_single_row(&field_input, &qa_params);
        let qa_serial_record = Record {
            protocol: "QA-CPU-Serial",
            field: "Mersenne127",
            log_input,
            message_elements: input_len,
            codeword_elements: input_len * config.qa_inverse_rate,
            setup: qa_setup,
            stats: Stats::from_durations(&qa_serial_samples),
            gpu: None,
            notes:
                "single QA codeword; existing serial per-row WHT path; output allocation included",
        };
        write_record(output, config, &qa_serial_record)?;
        print_record(&qa_serial_record);

        let qa_parallel_samples = timed_samples(
            || {
                let codeword = qa_encode_cpu_parallel_single_row(&field_input, &qa_params);
                black_box(codeword.as_slice());
            },
            config.warmups,
            config.repetitions,
        );
        let qa_parallel_reference = qa_encode_cpu_parallel_single_row(&field_input, &qa_params);
        if qa_parallel_reference != qa_serial_reference {
            return Err(format!(
                "QA serial/parallel output mismatch at input 2^{log_input}"
            ));
        }
        let qa_parallel_record = Record {
            protocol: "QA-CPU-Parallel",
            field: "Mersenne127",
            log_input,
            message_elements: input_len,
            codeword_elements: input_len * config.qa_inverse_rate,
            setup: Duration::ZERO,
            stats: Stats::from_durations(&qa_parallel_samples),
            gpu: None,
            notes: "single QA codeword; WHT butterfly blocks and pointwise products parallelized within the row; output allocation included",
        };
        write_record(output, config, &qa_parallel_record)?;
        print_record(&qa_parallel_record);

        // CUDA context creation, coefficient upload, host registration, output allocation,
        // and one or more warm-ups are all excluded from measured encoding time.
        let gpu_setup_start = Instant::now();
        let mut gpu = GpuQaEncoder::new(&qa_params, config.gpu_batch_rows)
            .map_err(|error| format!("CUDA setup failed for 2^{log_input}: {error}"))?;
        let gpu_input = gpu
            .register_input(field_input.clone())
            .map_err(|error| format!("CUDA input registration failed: {error}"))?;
        let mut gpu_output = gpu
            .allocate_output(config.qa_rows)
            .map_err(|error| format!("CUDA output allocation failed: {error}"))?;
        let gpu_setup = gpu_setup_start.elapsed();
        for _ in 0..config.warmups {
            gpu.encode_rows_into(&gpu_input, &mut gpu_output)
                .map_err(|error| format!("CUDA warm-up failed: {error}"))?;
        }
        let mut gpu_timings = Vec::with_capacity(config.repetitions);
        let mut gpu_durations = Vec::with_capacity(config.repetitions);
        for _ in 0..config.repetitions {
            let timing = gpu
                .encode_rows_into(&gpu_input, &mut gpu_output)
                .map_err(|error| format!("CUDA encoding failed: {error}"))?;
            gpu_durations.push(timing.total_wall);
            gpu_timings.push(timing);
        }
        if gpu_output.as_slice() != qa_serial_reference.as_slice() {
            return Err(format!("QA CPU/GPU output mismatch at input 2^{log_input}"));
        }
        let gpu_record = Record {
            protocol: "QA-GPU",
            field: "Mersenne127",
            log_input,
            message_elements: input_len,
            codeword_elements: input_len * config.qa_inverse_rate,
            setup: gpu_setup,
            stats: Stats::from_durations(&gpu_durations),
            gpu: Some(GpuBreakdown::from_samples(&gpu_timings)),
            notes: "pinned H2D + CUDA QA encoding + pinned D2H; CPU/GPU equality checked",
        };
        write_record(output, config, &gpu_record)?;
        print_record(&gpu_record);
        drop(qa_serial_reference);
        drop(qa_parallel_reference);
        if config.qa_only {
            output
                .flush()
                .map_err(|error| format!("failed to flush CSV: {error}"))?;
            return Ok(());
        }

        // RAA uses the public repository's encode_bits_ser implementation,
        // but selects inverse rate 2 to match the paper's speed experiment.
        // Note that the pinned public revision's repeat_interleave constructs
        // new_input but returns repetition; the CSV records that fact rather
        // than silently benchmarking a locally corrected implementation.
        let raa_setup_start = Instant::now();
        let permutation = Permutation::create(&mut *rng, input_len * config.raa_inverse_rate);
        let raa_setup = raa_setup_start.elapsed();
        let raa_validation =
            encode_bits_ser(raa_input.clone(), &permutation, config.raa_inverse_rate);
        if raa_validation.len() != input_len * config.raa_inverse_rate {
            return Err(format!(
                "RAA returned {} elements, expected {}",
                raa_validation.len(),
                input_len * config.raa_inverse_rate
            ));
        }
        drop(raa_validation);
        // encode_bits_ser consumes its input, so clone it strictly outside the clock.
        let mut raa_precise_samples = Vec::with_capacity(config.repetitions);
        for _ in 0..config.warmups {
            let message = raa_input.clone();
            black_box(encode_bits_ser(
                message,
                &permutation,
                config.raa_inverse_rate,
            ));
        }
        for _ in 0..config.repetitions {
            let message = raa_input.clone();
            let start = Instant::now();
            let codeword = encode_bits_ser(message, &permutation, config.raa_inverse_rate);
            let elapsed = start.elapsed();
            black_box(codeword.as_slice());
            raa_precise_samples.push(elapsed);
        }
        let raa_record = Record {
            protocol: "RAA",
            field: "Blazeu64 binary words",
            log_input,
            message_elements: input_len,
            codeword_elements: input_len * config.raa_inverse_rate,
            setup: raa_setup,
            stats: Stats::from_durations(&raa_precise_samples),
            gpu: None,
            notes: "serial rate-1/2 speed experiment using public Blaze encode_bits_ser as-is; input clone/setup excluded; inverse-rate-2 RAA does not inherit the cited inverse-rate>=4 distance theorem; pinned revision repeat_interleave returns repetition instead of constructed new_input",
        };
        write_record(output, config, &raa_record)?;
        print_record(&raa_record);

        // Brakedown2 uses the paper's rate-1/2 parameters. Encode rows
        // sequentially, matching QA, RAA, and BaseFold under the one-worker
        // global Rayon pool.
        let brakedown_setup_start = Instant::now();
        let brakedown = Brakedown::<Mersenne127>::new_multilinear::<BrakedownRate2>(
            log_input,
            20.min(input_len - 1),
            &mut *rng,
        );
        let brakedown_row_len = brakedown.row_len();
        let brakedown_codeword_len = brakedown.codeword_len();
        let brakedown_rows = input_len / brakedown_row_len;
        let brakedown_setup = brakedown_setup_start.elapsed();
        let mut brakedown_output =
            vec![<Mersenne127 as Field>::ZERO; brakedown_rows * brakedown_codeword_len];
        let brakedown_samples = timed_samples(
            || {
                brakedown_output
                    .chunks_mut(brakedown_codeword_len)
                    .zip(field_input.chunks(brakedown_row_len))
                    .for_each(|(row, message)| {
                        row[..brakedown_row_len].copy_from_slice(message);
                        brakedown.encode(row);
                    });
                black_box(brakedown_output.as_slice());
            },
            config.warmups,
            config.repetitions,
        );
        let brakedown_record = Record {
            protocol: "Brakedown",
            field: "Mersenne127",
            log_input,
            message_elements: input_len,
            codeword_elements: brakedown_output.len(),
            setup: brakedown_setup,
            stats: Stats::from_durations(&brakedown_samples),
            gpu: None,
            notes: "Brakedown2 rate-1/2 parameters (alpha=0.3 c_n=11 d_n=22); serial row encoding; output allocation and sparse-matrix setup excluded; row copy included",
        };
        write_record(output, config, &brakedown_record)?;
        print_record(&brakedown_record);

        let basefold_setup_start = Instant::now();
        let basefold_params = ConcreteBasefold::setup(input_len, 1, &mut *rng)
            .map_err(|error| format!("BaseFold setup failed: {:?}", error))?;
        let (basefold_pp, _) = ConcreteBasefold::trim(&basefold_params, input_len, 1)
            .map_err(|error| format!("BaseFold trim failed: {:?}", error))?;
        let basefold_setup = basefold_setup_start.elapsed();
        let basefold_input = Type2Polynomial {
            poly: field_input.clone(),
        };
        let basefold_log_rate = config.basefold_inverse_rate.ilog2() as usize;
        let basefold_samples = timed_samples(
            || {
                let (coefficients, boolean_evaluations) =
                    interpolate_over_boolean_hypercube_with_copy(&basefold_input);
                let codeword = evaluate_over_foldable_domain(
                    basefold_log_rate,
                    coefficients,
                    &basefold_pp.table,
                    "random".to_owned(),
                );
                black_box(boolean_evaluations.poly.as_slice());
                black_box(codeword.poly.as_slice());
            },
            config.warmups,
            config.repetitions,
        );
        let basefold_record = Record {
            protocol: "BaseFold",
            field: "Mersenne127",
            log_input,
            message_elements: input_len,
            codeword_elements: input_len * config.basefold_inverse_rate,
            setup: basefold_setup,
            stats: Stats::from_durations(&basefold_samples),
            gpu: None,
            notes: "commit-path interpolation + foldable-domain evaluation; Merkle hashing excluded; internal Rayon uses the configured global worker count; not part of the serial-vs-parallel QA comparison",
        };
        write_record(output, config, &basefold_record)?;
        print_record(&basefold_record);
        output
            .flush()
            .map_err(|error| format!("failed to flush CSV: {error}"))?;
        Ok(())
    }

    fn create_output(path: &Path) -> Result<BufWriter<File>, String> {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)
                .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
        }
        let file = File::create(path)
            .map_err(|error| format!("failed to create {}: {error}", path.display()))?;
        let mut output = BufWriter::new(file);
        writeln!(
            output,
            "protocol,field,log_input,message_elements,codeword_elements,actual_inverse_rate,threads,execution_mode,warmups,repetitions,setup_ms,mean_ms,median_ms,min_ms,max_ms,cv_percent,mean_ns_per_input_element,gpu_h2d_mean_ms,gpu_kernel_mean_ms,gpu_d2h_mean_ms,gpu_cuda_total_mean_ms,public_repo_revision,notes"
        )
        .map_err(|error| format!("failed to write CSV header: {error}"))?;
        Ok(output)
    }

    pub fn run() -> ExitCode {
        let config = match parse_config() {
            Ok(Some(config)) => config,
            Ok(None) => {
                println!("{}", usage());
                return ExitCode::SUCCESS;
            }
            Err(error) => {
                eprintln!("{error}");
                return ExitCode::from(2);
            }
        };
        if let Err(error) = ThreadPoolBuilder::new()
            .num_threads(config.threads)
            .build_global()
        {
            eprintln!("failed to configure Rayon: {error}");
            return ExitCode::FAILURE;
        }
        let mut output = match create_output(&config.output) {
            Ok(output) => output,
            Err(error) => {
                eprintln!("{error}");
                return ExitCode::FAILURE;
            }
        };
        let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
        println!(
            "encoding comparison: 2^{}..2^{}; repetitions={}; warmups={}; QA serial + single-codeword parallel; Rayon threads={}",
            config.log_min,
            config.log_max,
            config.repetitions,
            config.warmups,
            config.threads,
        );
        println!("GPU: detected on first QA-GPU setup");
        for log_input in config.log_min..=config.log_max {
            if let Err(error) = benchmark_size(&config, log_input, &mut output, &mut rng) {
                eprintln!("benchmark failed: {error}");
                return ExitCode::FAILURE;
            }
        }
        println!("CSV written to {}", config.output.display());
        ExitCode::SUCCESS
    }
}

#[cfg(feature = "cuda")]
fn main() -> std::process::ExitCode {
    with_cuda::run()
}
