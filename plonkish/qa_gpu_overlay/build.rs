use std::{env, path::PathBuf, process::Command};

fn run(command: &mut Command, description: &str) {
    let status = command
        .status()
        .unwrap_or_else(|error| panic!("failed to run {description}: {error}"));
    assert!(status.success(), "{description} failed with status {status}");
}

fn main() {
    println!("cargo:rerun-if-changed=cuda/qa_mersenne127.cu");

    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    let cuda_root = env::var_os("CUDA_HOME")
        .or_else(|| env::var_os("CUDA_PATH"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/usr/local/cuda"));
    let nvcc = cuda_root.join("bin/nvcc");
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set by Cargo"));
    let object = out_dir.join("qa_mersenne127.o");
    let library = out_dir.join("libqa_mersenne127_cuda.a");

    run(
        Command::new(&nvcc)
            .arg("-O3")
            .arg("--std=c++17")
            .arg("-lineinfo")
            .arg("-Xcompiler=-fPIC")
            .arg("-c")
            .arg("cuda/qa_mersenne127.cu")
            .arg("-o")
            .arg(&object),
        "nvcc",
    );

    run(
        Command::new("ar")
            .arg("crus")
            .arg(&library)
            .arg(&object),
        "ar",
    );

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-search=native={}", cuda_root.join("lib64").display());
    println!("cargo:rustc-link-lib=static=qa_mersenne127_cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}

