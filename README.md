# Quasar: Concept Implementation

This repository contains a concept implementation of **Quasar**, a QA-code-based polynomial commitment prototype built on top of the `plonkish` codebase.

Quasar explores the use of **quasi-abelian (QA) codes** and **code-switching arguments** for efficient polynomial commitment schemes. The implementation is intended for research and benchmarking purposes. It is not a production-ready cryptographic library.

## Overview

Quasar commits to a multilinear polynomial by reshaping its evaluations into a matrix and encoding each row with a QA code. The committed QA codeword is Merkle-committed column-wise. During opening, the prover checks consistency between the committed codeword and folded QA witnesses using:

* row-wise QA encoding,
* Walsh-Hadamard transform (WHT) relations,
* scaling relations for QA coefficient vectors,
* selector-sumcheck for sampled-column consistency,
* a final batched BaseFold opening for all multilinear evaluation claims.

The implementation currently supports two benchmark modes:

* `merge`: benchmark mode using the current merged/proximity-style opening path. This is the default mode.
* `full`: complete protocol mode that additionally runs the evaluation branch.


## Repository Structure

The implementation is mainly located in the multilinear PCS components of the backend, with benchmark code under the benchmark crate.

Typical relevant files include:

```text
plonkish_backend/src/pcs/multilinear/
benchmark/benches/
benchmark/bench_data/
```

## Build

Install Rust and Cargo, then build the project:

```bash
cargo build --release
```

For a quick compilation check:

```bash
cargo check
```

To check the benchmark crate:

```bash
cargo check -p benchmark --bench qabase_bench
```

## Running Benchmarks

The main benchmark supports both `merge` and `full` opening modes.

### Default: Merge Mode

If `--open-mode` is not specified, the benchmark uses `merge` mode by default.

```bash
cargo bench -p benchmark --bench qabase_bench -- \
  --total-k 20..30 \
  --inverse-rate 2 \
  --field-bits 127 \
  --security 100 \
  --distance-failure 100 \
  --auto-queries \
  --samples 5 \
  --threads 32
```

This runs the current merged/proximity-style opening benchmark.

### Full Mode

To run the complete protocol including the evaluation branch:

```bash
cargo bench -p benchmark --bench qabase_bench -- \
  --total-k 20..30 \
  --open-mode full \
  --inverse-rate 2 \
  --field-bits 127 \
  --security 100 \
  --distance-failure 100 \
  --auto-queries \
  --samples 5 \
  --threads 32
```

In `full` mode, the benchmark additionally constructs an evaluation witness and proves the final evaluation claim.

## Important Parameters

### `--total-k`

Specifies the total input size exponent.

For example:

```text
--total-k 27
```

means the committed polynomial has size (2^{27}).

A range can also be used:

```text
--total-k 20..30
```

### `--inverse-rate`

Specifies the QA inverse rate (c). Common values are:

```text
--inverse-rate 2
--inverse-rate 4
```

For QA encoding, each row message of length (N) is encoded into a codeword of length (cN).

### `--field-bits`

Specifies the field size in bits for parameter selection.

For example:

```text
--field-bits 127
```

is used for the Mersenne-127 benchmark setting.

### `--security`

Target query soundness in bits.

Example:

```text
--security 100
```

### `--distance-failure`

Target failure probability, in bits, for the random QA-code distance bound.

Example:

```text
--distance-failure 100
```

### `--auto-queries`

Automatically computes the number of sampled Merkle columns from the estimated QA-code distance and the target security level.

### `--samples`

Number of benchmark repetitions.

Example:

```text
--samples 5
```

Depending on the benchmark script, the final output may average over the last several samples.

### `--threads`

Number of Rayon worker threads.

Example:

```text
--threads 32
```

## Output

The benchmark reports:

* commitment time,
* proving/opening time,
* verification time,
* proof size,
* QA code distance estimate,
* number of Merkle queries,
* matrix shape,
* inverse rate,
* opening mode.

Example output line:

```text
mersenne127,full,27,21,6,64,2,127,100,100,0.40963537,473,5648000,32,1693,6719,32
```

A typical interpretation is:

```text
field,total_k,row_k,log_rows,num_rows,inverse_rate,field_bits,
security_bits,distance_failure_bits,delta,queries,proof_bytes,
threads,commit_ms,prove_ms,verify_ms
```

If `open_mode` is enabled in the CSV output, the second column records either:

```text
merge
full
```

## Opening Modes

### Merge Mode

`merge` mode is the default benchmark mode. It runs the current merged/proximity-style path and is intended to estimate the optimized cost when the proximity test and evaluation check are merged.

This mode does not separately run the full evaluation branch.

### Full Mode

`full` mode runs the complete protocol path:

1. Commit to the row-wise QA codeword.
2. Build the random folded witness (u = QAEnc(r^T M)).
3. Build the evaluation witness (q = QAEnc(eq(z_L, .)^T M)).
4. Prove QA encoding correctness for both branches.
5. Reuse authenticated sampled columns for the evaluation check.
6. Add the final evaluation claim (q^{(0)}(z_R)=y).
7. Run one global batched BaseFold opening.

This mode is useful for measuring the cost of the full conceptual protocol.


## Status

This is a research prototype. The implementation is intended to validate the Quasar design and evaluate concrete costs of QA-code-based code-switching.

The implementation prioritizes clarity and benchmarkability over production hardening. In particular:

* some verifier-side checks may still use assertions in prototype code paths,
* parameter selection is benchmark-oriented,
* the implementation has not undergone a production security audit,
* generated benchmark data should be treated separately from source code.

## License and Attribution

This repository is based on the original `plonkish` codebase with additional Quasar-related implementation and benchmarks.

