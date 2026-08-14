use crate::gpu::{
    GpuQaDeviceOutput, GpuQaDeviceOutputSetupTiming, GpuQaEncoder, GpuQaInput,
    GpuQaInputSetupTiming, GpuQaTiming,
};
use plonkish_backend::{
    pcs::multilinear::quasar::{
        commit_tree_and_write, merkelize_from_leaves, CommitmentChunk,
        QABaseCommitment, QABaseProverParams,
    },
    util::{
        hash::{Blake2s, Output},
        new_fields::Mersenne127,
        transcript::TranscriptWrite,
    },
};
use rayon::prelude::*;
use std::{fmt, str::FromStr, time::Instant};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QuasarCommitBackend {
    Cpu,
    Cuda,
}

impl fmt::Display for QuasarCommitBackend {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => formatter.write_str("cpu"),
            Self::Cuda => formatter.write_str("cuda"),
        }
    }
}

impl FromStr for QuasarCommitBackend {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "cpu" => Ok(Self::Cpu),
            "cuda" | "gpu" => Ok(Self::Cuda),
            _ => Err(format!(
                "unknown commitment backend {value:?}; expected cpu or cuda"
            )),
        }
    }
}

/// Reusable CUDA state for Quasar commitments of one fixed matrix shape.
///
/// The input and output allocations are registered once.  A CUDA commitment
/// temporarily moves the output buffer into `QABaseCommitment`; call
/// `reclaim` after proving to return it for the next sample.
pub struct CudaQuasarCommitter {
    encoder: GpuQaEncoder,
    input: GpuQaInput,
    output: Option<GpuQaDeviceOutput>,
}

impl CudaQuasarCommitter {
    pub fn new(
        pp: &QABaseProverParams<Mersenne127, Blake2s>,
        word: &[Vec<Mersenne127>],
        gpu_batch_rows: usize,
    ) -> Result<Self, String> {
        if word.len() != pp.num_rows {
            return Err(format!(
                "Quasar matrix has {} rows, expected {}",
                word.len(),
                pp.num_rows
            ));
        }
        let row_len = 1usize << pp.num_vars;
        if word.iter().any(|row| row.len() != row_len) {
            return Err(format!(
                "every Quasar message row must contain {row_len} field elements"
            ));
        }

        let encoder = GpuQaEncoder::new(&pp.qa_params, gpu_batch_rows)?;
        let messages = word.iter().flatten().copied().collect::<Vec<_>>();
        let input = encoder.register_input(messages)?;
        let output = encoder.allocate_device_output(pp.num_rows)?;
        Ok(Self {
            encoder,
            input,
            output: Some(output),
        })
    }

    pub fn device_name(&self) -> Result<String, String> {
        self.encoder.device_name()
    }

    pub fn input_setup_timing(&self) -> &GpuQaInputSetupTiming {
        self.input.setup_timing()
    }

    pub fn output_setup_timing(&self) -> &GpuQaDeviceOutputSetupTiming {
        self.output
            .as_ref()
            .expect("CUDA Quasar output is currently owned by a commitment")
            .setup_timing()
    }

    pub fn warm_up(&mut self) -> Result<GpuQaTiming, String> {
        let mut output = self
            .output
            .take()
            .ok_or_else(|| "CUDA Quasar output is already in use".to_owned())?;
        let result = self.encoder.encode_rows_to_device(&self.input, &mut output);
        self.output = Some(output);
        result
    }

    pub fn commit_and_write(
        &mut self,
        pp: &QABaseProverParams<Mersenne127, Blake2s>,
        transcript: &mut impl TranscriptWrite<CommitmentChunk<Blake2s>, Mersenne127>,
    ) -> Result<
        (
            QABaseCommitment<Mersenne127, Blake2s, GpuQaDeviceOutput>,
            GpuQaTiming,
        ),
        String,
    > {
        let mut output = self
            .output
            .take()
            .ok_or_else(|| "CUDA Quasar output is already in use".to_owned())?;
        let mut timing = match self.encoder.encode_rows_to_device(&self.input, &mut output) {
            Ok(timing) => timing,
            Err(error) => {
                self.output = Some(output);
                return Err(error);
            }
        };
        let decode_start = Instant::now();
        let leaves = output
            .leaf_digest_bytes()
            .par_chunks_exact(32)
            .map(|bytes| {
                let mut digest = Output::<Blake2s>::default();
                digest[..].copy_from_slice(bytes);
                digest
            })
            .collect::<Vec<_>>();
        timing.host_leaf_decode = decode_start.elapsed();
        let upper_start = Instant::now();
        let tree = merkelize_from_leaves::<Blake2s>(leaves);
        timing.cpu_upper_merkle = upper_start.elapsed();
        println!(
            "degree {}, GPU leaves + CPU upper Merkle {:?}",
            pp.num_vars,
            timing.cpu_upper_merkle,
        );
        let commitment = commit_tree_and_write(pp, output, tree, transcript);
        Ok((commitment, timing))
    }

    pub fn reclaim(
        &mut self,
        commitment: QABaseCommitment<Mersenne127, Blake2s, GpuQaDeviceOutput>,
    ) -> Result<(), String> {
        if self.output.is_some() {
            return Err("CUDA Quasar output was reclaimed twice".to_owned());
        }
        self.output = Some(commitment.into_codeword());
        Ok(())
    }
}
