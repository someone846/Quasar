# Quasar: Concept Implementation

This repository contains a concept implementation of **Quasar**, a QA-code-based polynomial commitment prototype built on top of the `plonkish` codebase.

Quasar explores the use of **quasi-abelian (QA) codes** and **code-switching arguments** for efficient polynomial commitment schemes. The implementation is intended for research and benchmarking purposes. It is not a production-ready cryptographic library.

## Overview

Quasar commits to a multilinear polynomial by reshaping its evaluation vector into a matrix and encoding each row with a QA code. The resulting QA codeword is Merkle-committed column-wise.

During opening, the prover establishes consistency between the committed codeword and the folded QA witnesses using:

- row-wise QA encoding;
- WHT-based sumcheck relations;
- public QA coefficient vectors;
- selector sumcheck for sampled-column consistency; and
- a final batched BaseFold opening for all multilinear evaluation claims.

The current implementation uses an optimized two-layer sumcheck argument for QA encoding. It eliminates online commitments to intermediate encoding vectors: only the input and output blocks are committed online, while the QA coefficient vectors are public and can be committed during preprocessing.

The main `quasar_bench` target runs the complete endpoint-only two-layer-GKR protocol, including both the proximity and evaluation branches. An optional CUDA implementation accelerates QA encoding and provides a hybrid GPU--CPU commitment path.

## Optimized QA Code Switching

For a message vector `v`, the QA encoder has three stages:

```text
v'      = WHT(v)
u_i'    = a_i' .* v'
u_i     = WHT(u_i')
```

Here, `.*` denotes coordinate-wise multiplication and each public vector `a_i'` determines one output block.

A direct verification of these stages would separately certify the two WHT relations and the pointwise-multiplication relation. It would also require the prover to commit online to intermediate vectors such as `v'` and `u_i'`. Quasar removes these commitments through two optimizations.

### Optimization 1: Merge the Second and Third Stages

For each output block, Quasar substitutes the pointwise-multiplication relation directly into the final WHT relation. The intermediate vector `u_i'` therefore disappears from the verification equation and no longer needs an online commitment.

All output blocks share the same first-stage vector `v'` and differ only in their public coefficient vectors `a_i'`. Their relations can consequently be batched before applying sumcheck. This optimization:

- eliminates all online commitments to `u_i'`;
- merges the sumchecks for pointwise multiplication and the final WHT into one sumcheck;
- verifies all output blocks with a single `log N`-round sumcheck; and
- leaves only one terminal evaluation claim on the shared vector `v'`.

### Optimization 2: Reduce the Remaining Claim to the Input

The remaining evaluation of `v'` is reduced through the first WHT relation

```text
v' = WHT(v).
```

A second sumcheck reduces this claim to an evaluation of the committed input vector `v`, together with a local evaluation of the WHT matrix polynomial. Since the WHT matrix is an exact Kronecker power of a fixed `2 x 2` matrix, its multilinear extension has a product-form representation and can be evaluated by the verifier in `O(log N)` field operations.

Consequently, the complete QA encoding relation is certified by two explicit sumchecks without a SPARK-style general-purpose circuit argument and without online commitments to any intermediate encoding witnesses.

## GPU Acceleration

QA encoding does not require a reduction to generic matrix multiplication. It consists almost entirely of regular data-parallel operations---WHT butterflies and coordinate-wise multiplications---that map naturally to GPU kernels.

The CUDA encoder computes exactly the same codeword as the CPU implementation:

```text
message_wht = WHT(message)
parity_i    = WHT(a_i' .* message_wht)
codeword    = message || parity_0 || ... || parity_(c-2)
```

The implementation includes the following optimizations:

- direct use of the backend's two-limb Montgomery representation for `Mersenne127`, avoiding field-representation conversion;
- fusion of the first WHT layers in shared memory;
- parallel processing across butterflies, rows, and parity blocks;
- one-time upload and reuse of the public QA coefficient vectors;
- batched row streaming when the complete input does not fit comfortably in device memory; and
- reuse of device allocations and pinned host buffers across repetitions.

### Hybrid GPU--CPU Commitment

The integrated commitment path keeps the complete QA codeword on the GPU. The GPU hashes each encoded column using backend-compatible BLAKE2b-256 and transfers only the 32-byte leaf digests to the CPU. The CPU then constructs the upper Merkle-tree layers. After Fiat--Shamir determines the queried indices, only the selected columns are copied back to the host for opening.

Thus, the hybrid path avoids transferring the full encoded codeword from GPU to CPU while retaining the existing CPU-side transcript and proof implementation.

In our benchmark environment:

- over message lengths from `2^12` to `2^25`, the end-to-end GPU QA encoder is `32.9x`--`148.3x` faster than the 32-thread CPU encoder; and
- over polynomial sizes from `2^20` to `2^29`, the hybrid GPU--CPU commitment is `3.4x`--`16.7x` faster than the 32-thread CPU commitment.

These results are hardware- and parameter-dependent. They are included to characterize the current prototype rather than to claim portable performance across all GPUs.

## Repository Structure

The main CPU implementation is located in the multilinear PCS components of the backend, with benchmark code under the benchmark crate. The optional CUDA integration is provided by `qa_gpu_overlay`.

Typical relevant paths include:

```text
plonkish_backend/src/pcs/multilinear/quasar.rs
benchmark/benches/quasar_bench.rs
benchmark/bench_data/
qa_gpu_overlay/
```

The backend module must also be exported from `plonkish_backend/src/pcs/multilinear.rs`:

```rust
pub mod quasar;
```

The benchmark target must be registered with `harness = false` in the benchmark crate's `Cargo.toml`:

```toml
[[bench]]
name = "quasar_bench"
harness = false
```

## CPU Build

Install Rust and Cargo, then build the project:

```bash
cargo build --release
```

For a quick compilation check:

```bash
cargo check -p plonkish_backend --lib
```

To check the benchmark crate:

```bash
cargo check -p benchmark --bench quasar_bench
```

## Running CPU Benchmarks

The `quasar_bench` target executes the complete Quasar protocol using the optimized endpoint-only two-layer-GKR opening. It does not use an `--open-mode` option.

### Smoke Test

```bash
cargo bench -p benchmark --bench quasar_bench -- \
  --smoke \
  --threads 8
```

### Rate-1/2 Experiment

```bash
cargo bench -p benchmark --bench quasar_bench -- \
  --total-k 20..=30 \
  --log-rows 6 \
  --inverse-rate 2 \
  --field-bits 127 \
  --security 100 \
  --distance-failure 100 \
  --auto-queries \
  --samples 5 \
  --threads 32
```

### Rate-1/4 Experiment

```bash
cargo bench -p benchmark --bench quasar_bench -- \
  --total-k 20..=30 \
  --log-rows 6 \
  --inverse-rate 4 \
  --field-bits 127 \
  --security 100 \
  --distance-failure 100 \
  --auto-queries \
  --samples 5 \
  --threads 32
```

The `--log-rows` parameter controls the matrix shape. For example, `--log-rows 6` uses 64 rows, while `--log-rows 8` uses 256 rows. The benchmark accepts one value, an inclusive range such as `6..=8`, or a comma-separated list.

## Building and Running the CUDA Path

The CUDA path requires:

- a recent Rust toolchain;
- the NVIDIA CUDA toolkit (`nvcc`; CUDA 12.x is recommended); and
- an NVIDIA GPU with at least 16 KiB of shared memory per block.

Place `qa_gpu_overlay` next to `plonkish_backend`:

```text
repository/
  plonkish_backend/
  qa_gpu_overlay/
```

If CUDA is not installed at `/usr/local/cuda`, set `CUDA_HOME` to the CUDA installation directory.

The backend definition of `Mersenne127` must use the ABI-stable representation included in `qa_gpu_overlay/mersenne127_repr.patch`. Apply it from the repository root if the change is not already present:

```bash
git apply qa_gpu_overlay/mersenne127_repr.patch
```

### Standalone CPU/GPU Encoder Benchmark

For a small smoke test:

```bash
cargo +nightly run --release --features cuda \
  --manifest-path qa_gpu_overlay/Cargo.toml \
  --bin qa_encode_cpu_gpu -- \
  --log-row-len 16 \
  --rows 8 \
  --inverse-rate 2 \
  --gpu-batch-rows 8 \
  --threads 32 \
  --repetitions 3
```

For the `2^26`-element, 64-row configuration:

```bash
cargo +nightly run --release --features cuda \
  --manifest-path qa_gpu_overlay/Cargo.toml \
  --bin qa_encode_cpu_gpu -- \
  --log-row-len 20 \
  --rows 64 \
  --inverse-rate 2 \
  --gpu-batch-rows 8 \
  --threads 32 \
  --repetitions 5
```

Reduce `--gpu-batch-rows` if GPU memory is insufficient. The coefficient vectors remain resident on the GPU while message rows are streamed in batches.

### Integrated Hybrid Commitment Benchmark

CPU commitment:

```bash
cargo +nightly run --release --features cuda \
  --manifest-path qa_gpu_overlay/Cargo.toml \
  --bin qa_commit_cpu_cuda -- \
  --commit-backend cpu \
  --total-k 20 \
  --log-rows 6 \
  --inverse-rate 2 \
  --queries 8 \
  --samples 3 \
  --threads 32
```

Hybrid CUDA commitment:

```bash
cargo +nightly run --release --features cuda \
  --manifest-path qa_gpu_overlay/Cargo.toml \
  --bin qa_commit_cpu_cuda -- \
  --commit-backend cuda \
  --total-k 20 \
  --log-rows 6 \
  --inverse-rate 2 \
  --queries 8 \
  --samples 3 \
  --threads 32 \
  --gpu-batch-rows 8 \
  --check-cpu-root
```

The `--check-cpu-root` option constructs an additional CPU commitment outside the measured samples and checks exact equality of the CPU and GPU Merkle roots. It is recommended for smoke tests and should normally be omitted from large timed runs.

Run the CUDA integration test with:

```bash
cargo +nightly test --release --features cuda \
  --manifest-path qa_gpu_overlay/Cargo.toml \
  --test quasar_commit_cuda -- --nocapture
```

The integration test compares CPU and GPU leaf digests, the Merkle root, and a complete Quasar proof.

## Important Parameters

### `--total-k`

Specifies the total input-size exponent. For example,

```text
--total-k 27
```

means that the committed polynomial has `2^27` evaluations. A range can also be used:

```text
--total-k 20..30
```

### `--inverse-rate`

Specifies the QA inverse rate `c`. Common values are:

```text
--inverse-rate 2
--inverse-rate 4
```

For QA encoding, each row message of length `N` is encoded into a codeword of length `cN`.

### `--field-bits`

Specifies the field size in bits for parameter selection. The Mersenne-127 benchmark setting uses:

```text
--field-bits 127
```

### `--security`

Specifies the target query soundness in bits:

```text
--security 100
```

### `--distance-failure`

Specifies, in bits, the target failure probability for the random QA-code distance bound:

```text
--distance-failure 100
```

### `--auto-queries`

Automatically computes the number of sampled Merkle columns from the estimated QA-code distance and the target security level.

### `--samples`

Specifies the number of benchmark repetitions:

```text
--samples 5
```

Depending on the benchmark executable, the reported result may average the measured samples after warm-up.

### `--threads`

Specifies the number of Rayon worker threads used by the CPU implementation:

```text
--threads 32
```

### `--gpu-batch-rows`

Specifies how many message rows are processed in each CUDA batch. Smaller values reduce peak device-memory usage at the cost of additional launches and transfers.

## Benchmark Output

The CPU benchmark reports:

- commitment time;
- proving/opening time;
- verification time;
- proof size;
- QA-code distance estimate;
- number of Merkle queries;
- matrix shape;
- inverse rate;
- numbers of endpoint commitments, opening claims, and unique opening points; and
- setup, trim, and evaluation-witness preparation times.

Example output line:

```text
mersenne127,quasar_two_layer_gkr_optimized,26,18,8,256,2,127,100,100,0.42140078,458,4,10,6,5568944,32,85.708,53.173,45.719,622.451,505.156,37.896
```

The CSV header is:

```text
field,protocol,total_k,row_k,log_rows,num_rows,inverse_rate,
field_bits,security_bits,distance_failure_bits,delta,queries,
endpoint_commitments,opening_claims,unique_opening_points,
proof_bytes,threads,setup_ms,trim_ms,eval_prepare_ms,
commit_ms,prove_ms,verify_ms
```

The CUDA benchmarks additionally report stage-separated CPU and GPU timings, transfer costs, GPU encoding and leaf-hashing times, CPU upper-tree time, queried-column transfer time, and CPU/GPU correctness checks.

## Benchmarked Protocol Path

`quasar_bench` runs the complete conceptual protocol:

1. Commit to the row-wise QA codeword.
2. Build the random folded witness `u = QAEnc(r^T M)`.
3. Build the evaluation witness `q = QAEnc(eq(z_L, .)^T M)`.
4. Prove QA encoding correctness for both branches using the optimized two-layer sumcheck argument.
5. Reuse authenticated sampled columns for the evaluation check.
6. Add the final evaluation claim `q^(0)(z_R) = y`.
7. Run one global batched BaseFold opening.

The benchmark reports the complete commitment, proving, and verification costs for this path. Setup, trim, and evaluation-witness preparation are reported separately.

## Status

This repository is a research prototype intended to validate the Quasar design and evaluate the concrete costs of QA-code-based code switching.

The implementation prioritizes clarity and benchmarkability over production hardening. In particular:

- some verifier-side checks may still use assertions in prototype code paths;
- parameter selection is benchmark-oriented;
- the implementation has not undergone a production security audit;
- the CUDA path currently targets the Mersenne-127 field representation and NVIDIA GPUs; and
- generated benchmark data should be treated separately from source code.

## License and Attribution

This repository is based on the original `plonkish` codebase, with additional Quasar-related implementation, benchmarks, and CUDA acceleration code. See the repository's license files and upstream notices for the applicable terms and attribution requirements.