use plonkish_backend::{
    pcs::multilinear::quasar::QAParams,
    util::new_fields::Mersenne127,
};
use qa_gpu_overlay::{
    cpu::{
        benchmark_field_operations, qa_encode_cpu_baseline_rows,
        qa_encode_cpu_profiled_rows,
    },
    gpu::GpuQaEncoder,
};
use rand_chacha::{rand_core::SeedableRng, ChaCha8Rng};
use rayon::ThreadPoolBuilder;
use std::{
    env,
    process::ExitCode,
    time::{Duration, Instant},
};

#[derive(Clone, Debug)]
struct Config {
    log_row_len: usize,
    rows: usize,
    inverse_rate: usize,
    repetitions: usize,
    gpu_batch_rows: usize,
    field_op_elements: usize,
    field_op_repetitions: usize,
    threads: usize,
    seed: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            log_row_len: 16,
            rows: 8,
            inverse_rate: 2,
            repetitions: 3,
            gpu_batch_rows: 8,
            field_op_elements: 1 << 20,
            field_op_repetitions: 16,
            threads: 32,
            seed: 91,
        }
    }
}

fn usage() -> &'static str {
    "qa_encode_cpu_gpu [options]\n\
     \n\
     --log-row-len K          QA message row length is 2^K (default 16)\n\
     --rows N                 number of rows (default 8)\n\
     --inverse-rate C         QA inverse rate, e.g. 2 or 4 (default 2)\n\
     --repetitions N          full encoder repetitions (default 3)\n\
     --gpu-batch-rows N       maximum resident GPU row batch (default 8)\n\
     --field-op-elements N    elements in the CPU operation test (default 2^20)\n\
     --field-op-repetitions N repetitions of the operation test (default 16)\n\
     --threads N              Rayon threads (default 32)\n\
     --seed N                 deterministic ChaCha8 seed (default 91)\n\
     -h, --help               show this help"
}

fn parse_usize(name: &str, value: Option<String>) -> Result<usize, String> {
    value
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
            "--log-row-len" => config.log_row_len = parse_usize(&argument, args.next())?,
            "--rows" => config.rows = parse_usize(&argument, args.next())?,
            "--inverse-rate" => config.inverse_rate = parse_usize(&argument, args.next())?,
            "--repetitions" => config.repetitions = parse_usize(&argument, args.next())?,
            "--gpu-batch-rows" => config.gpu_batch_rows = parse_usize(&argument, args.next())?,
            "--field-op-elements" => {
                config.field_op_elements = parse_usize(&argument, args.next())?
            }
            "--field-op-repetitions" => {
                config.field_op_repetitions = parse_usize(&argument, args.next())?
            }
            "--threads" => config.threads = parse_usize(&argument, args.next())?,
            "--seed" => {
                config.seed = args
                    .next()
                    .ok_or_else(|| "missing value for --seed".to_owned())?
                    .parse::<u64>()
                    .map_err(|error| format!("invalid value for --seed: {error}"))?;
            }
            _ => return Err(format!("unknown argument {argument}\n\n{}", usage())),
        }
    }
    if config.log_row_len >= usize::BITS as usize {
        return Err("--log-row-len is too large for this platform".to_owned());
    }
    if config.rows == 0 || config.repetitions == 0 || config.threads == 0 {
        return Err("rows, repetitions, and threads must be non-zero".to_owned());
    }
    if config.inverse_rate < 2 || !config.inverse_rate.is_power_of_two() {
        return Err("inverse rate must be a power of two and at least two".to_owned());
    }
    Ok(Some(config))
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e3
}

fn percent(part: Duration, total: Duration) -> f64 {
    if total.is_zero() {
        0.0
    } else {
        100.0 * part.as_secs_f64() / total.as_secs_f64()
    }
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn coefficient_of_variation(values: &[f64]) -> f64 {
    let average = mean(values);
    if average == 0.0 {
        return 0.0;
    }
    let variance = values
        .iter()
        .map(|value| (value - average) * (value - average))
        .sum::<f64>()
        / values.len() as f64;
    100.0 * variance.sqrt() / average
}

fn main() -> ExitCode {
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

    let row_len = 1usize << config.log_row_len;
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    let params =
        QAParams::<Mersenne127>::new_random(row_len, config.inverse_rate, &mut rng);
    let input_initialize_start = Instant::now();
    let messages = (0..config.rows * row_len)
        .map(|_| <Mersenne127 as plonkish_backend::util::arithmetic::Field>::random(&mut rng))
        .collect::<Vec<_>>();
    let input_initialize = input_initialize_start.elapsed();

    let field_ops = benchmark_field_operations::<Mersenne127>(
        &mut rng,
        config.field_op_elements,
        config.field_op_repetitions,
    );
    println!(
        "field=2^127-1 rows={} row_len=2^{} inverse_rate={} rayon_threads={}",
        config.rows, config.log_row_len, config.inverse_rate, config.threads
    );
    println!("field operations (vector-shaped CPU benchmark):");
    println!("  add        {:8.3} ns/op", field_ops.add_ns_per_op());
    println!("  sub        {:8.3} ns/op", field_ops.sub_ns_per_op());
    println!("  mul        {:8.3} ns/op", field_ops.mul_ns_per_op());
    println!(
        "  butterfly  {:8.3} ns/pair (one add + one sub)",
        field_ops.butterfly_ns()
    );

    let mut gpu = match GpuQaEncoder::new(&params, config.gpu_batch_rows) {
        Ok(gpu) => gpu,
        Err(error) => {
            eprintln!("failed to initialize CUDA encoder: {error}");
            return ExitCode::FAILURE;
        }
    };
    match gpu.device_name() {
        Ok(name) => println!("GPU: {name}"),
        Err(error) => eprintln!("warning: failed to read GPU name: {error}"),
    }
    let messages = match gpu.register_input(messages) {
        Ok(messages) => messages,
        Err(error) => {
            eprintln!("failed to register reusable pinned GPU input: {error}");
            return ExitCode::FAILURE;
        }
    };
    let input_gib = messages.len() as f64
        * std::mem::size_of::<Mersenne127>() as f64
        / (1u64 << 30) as f64;
    println!("reusable pinned host input: {input_gib:.3} GiB");
    println!(
        "  allocation + random initialize {:10.3} ms",
        ms(input_initialize)
    );
    println!(
        "  cudaHostRegister               {:10.3} ms",
        ms(messages.setup_timing().pin_registration)
    );

    let mut gpu_output = match gpu.allocate_output(config.rows) {
        Ok(output) => output,
        Err(error) => {
            eprintln!("failed to allocate reusable pinned GPU output: {error}");
            return ExitCode::FAILURE;
        }
    };
    let output_gib = gpu_output.len() as f64
        * std::mem::size_of::<Mersenne127>() as f64
        / (1u64 << 30) as f64;
    let setup = gpu_output.setup_timing();
    println!("reusable pinned host output: {output_gib:.3} GiB");
    println!("  virtual allocation             {:10.3} ms", ms(setup.allocation));
    println!(
        "  parallel pre-fault + initialize{:10.3} ms",
        ms(setup.prefault_and_initialize)
    );
    println!("  cudaHostRegister               {:10.3} ms", ms(setup.pin_registration));
    println!("  one-time output setup total    {:10.3} ms", ms(setup.total));

    let warm_up = match gpu.encode_rows_into(&messages, &mut gpu_output) {
        Ok(timing) => timing,
        Err(error) => {
            eprintln!("CUDA warm-up failed: {error}");
            return ExitCode::FAILURE;
        }
    };
    println!(
        "one-time GPU warm-up              {:10.3} ms",
        ms(warm_up.total_wall)
    );
    println!(
        "  H2D                             {:10.3} ms",
        warm_up.host_to_device_ms
    );
    println!(
        "  kernels + device copy           {:10.3} ms",
        warm_up.device_input_copy_ms
            + warm_up.first_wht_ms
            + warm_up.scaling_ms
            + warm_up.second_wht_ms
            + warm_up.assemble_ms
    );
    println!(
        "  D2H                             {:10.3} ms",
        warm_up.device_to_host_ms
    );

    let mut baseline_samples = Vec::with_capacity(config.repetitions);
    let mut h2d_samples = Vec::with_capacity(config.repetitions);
    let mut gpu_wall_samples = Vec::with_capacity(config.repetitions);

    for repetition in 0..config.repetitions {
        let start = Instant::now();
        let baseline = qa_encode_cpu_baseline_rows(messages.as_slice(), row_len, &params);
        let baseline_time = start.elapsed();

        let (profiled, cpu) = qa_encode_cpu_profiled_rows(messages.as_slice(), row_len, &params);
        if baseline != profiled {
            eprintln!("CPU profiled encoder disagrees with the current baseline encoder");
            return ExitCode::FAILURE;
        }
        // The baseline is sufficient for the later GPU equality check. Drop
        // this second full 2 GiB codeword before timing H2D to reduce host
        // memory pressure and memory-controller interference.
        drop(profiled);

        let gpu_timing = match gpu.encode_rows_into(&messages, &mut gpu_output) {
            Ok(result) => result,
            Err(error) => {
                eprintln!("CUDA encoding failed: {error}");
                return ExitCode::FAILURE;
            }
        };
        if baseline.as_slice() != gpu_output.as_slice() {
            let mismatch = baseline
                .iter()
                .zip(gpu_output.as_slice().iter())
                .position(|(cpu, gpu)| cpu != gpu)
                .unwrap_or(baseline.len().min(gpu_output.len()));
            eprintln!("CPU/GPU mismatch at flattened codeword index {mismatch}");
            return ExitCode::FAILURE;
        }

        println!("\nrepetition {} (CPU/GPU outputs match):", repetition + 1);
        println!(
            "  current CPU baseline total     {:10.3} ms",
            ms(baseline_time)
        );
        println!("  profiled CPU total             {:10.3} ms", ms(cpu.total));
        println!("    first WHT                    {:10.3} ms  {:6.2}%", ms(cpu.first_wht), percent(cpu.first_wht, cpu.measured_compute()));
        println!("    scaling multiplications      {:10.3} ms  {:6.2}%", ms(cpu.scaling_multiplications), percent(cpu.scaling_multiplications, cpu.measured_compute()));
        println!("    second WHT                   {:10.3} ms  {:6.2}%", ms(cpu.second_wht), percent(cpu.second_wht, cpu.measured_compute()));
        println!("    allocation + systematic copy {:10.3} ms", ms(cpu.allocation + cpu.systematic_copy));
        println!("  GPU CUDA timeline              {:10.3} ms", gpu_timing.total_cuda_ms);
        println!("    H2D                          {:10.3} ms", gpu_timing.host_to_device_ms);
        println!("    device input copy            {:10.3} ms", gpu_timing.device_input_copy_ms);
        println!("    first WHT                    {:10.3} ms", gpu_timing.first_wht_ms);
        println!("    scaling multiplications      {:10.3} ms", gpu_timing.scaling_ms);
        println!("    second WHT                   {:10.3} ms", gpu_timing.second_wht_ms);
        println!("    assembly                     {:10.3} ms", gpu_timing.assemble_ms);
        println!("    D2H                          {:10.3} ms", gpu_timing.device_to_host_ms);
        println!(
            "  GPU steady-state wall          {:10.3} ms  (reused pinned input/output)",
            ms(gpu_timing.total_wall)
        );
        println!("    representation conversion         0.000 ms  (direct Montgomery limbs)");
        println!(
            "  speedup (baseline / GPU wall)  {:10.3} x",
            baseline_time.as_secs_f64() / gpu_timing.total_wall.as_secs_f64()
        );

        baseline_samples.push(ms(baseline_time));
        h2d_samples.push(gpu_timing.host_to_device_ms);
        gpu_wall_samples.push(ms(gpu_timing.total_wall));
    }

    let h2d_min = h2d_samples.iter().copied().fold(f64::INFINITY, f64::min);
    let h2d_max = h2d_samples.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let wall_min = gpu_wall_samples.iter().copied().fold(f64::INFINITY, f64::min);
    let wall_max = gpu_wall_samples.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mean_baseline = mean(&baseline_samples);
    let mean_wall = mean(&gpu_wall_samples);
    println!("\nsteady-state summary (warm-up excluded):");
    println!(
        "  H2D mean / min / max           {:10.3} / {:.3} / {:.3} ms  (CV {:.2}%)",
        mean(&h2d_samples),
        h2d_min,
        h2d_max,
        coefficient_of_variation(&h2d_samples)
    );
    println!(
        "  GPU wall mean / min / max      {:10.3} / {:.3} / {:.3} ms  (CV {:.2}%)",
        mean_wall,
        wall_min,
        wall_max,
        coefficient_of_variation(&gpu_wall_samples)
    );
    println!(
        "  mean CPU / GPU speedup         {:10.3} x",
        mean_baseline / mean_wall
    );

    ExitCode::SUCCESS
}
