#![cfg(feature = "cuda")]

use std::{fs, process::Command};

/// Exercises the complete GPU matrix-multiplication driver, including exact
/// CPU/CUDA roots, application sumcheck, three MLE openings, and verification.
#[test]
fn matmul_cuda_end_to_end_smoke() {
    let output_path = std::env::temp_dir().join(format!(
        "quasar_matmul_gpu_smoke_{}.csv",
        std::process::id()
    ));
    let output = Command::new(env!("CARGO_BIN_EXE_matmul_bench_gpu"))
        .args(["--smoke", "--check-cpu-root", "--output"])
        .arg(&output_path)
        .output()
        .expect("failed to start matmul_bench_gpu");
    assert!(
        output.status.success(),
        "GPU matmul smoke failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    let csv = fs::read_to_string(&output_path).expect("missing GPU matmul CSV");
    assert!(csv.lines().any(|line| line.starts_with("Quasar-GPU,")));
    let _ = fs::remove_file(output_path);
}
