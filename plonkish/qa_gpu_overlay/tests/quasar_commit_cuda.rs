#![cfg(feature = "cuda")]

use plonkish_backend::{
    pcs::multilinear::quasar::{
        commit_and_write, eval_mle_from_evals, prove_qabase_open_full_two_layer_gkr,
        qabase_split_evaluation_point, setup, trim, verify_qabase_open_full_two_layer_gkr,
    },
    util::{
        arithmetic::Field,
        hash::{Blake2s, Output},
        new_fields::Mersenne127,
        transcript::{Blake2sTranscript, InMemoryTranscript},
    },
};
use qa_gpu_overlay::quasar_commit::CudaQuasarCommitter;
use rand_chacha::{rand_core::SeedableRng, ChaCha8Rng};
use std::io::Cursor;

type TestTranscript = Blake2sTranscript<Cursor<Vec<u8>>>;

#[test]
fn cuda_commitment_matches_cpu_and_verifies() {
    let row_len = 1usize << 6;
    let num_rows = 4usize;
    let inverse_rate = 2usize;
    let mut rng = ChaCha8Rng::from_seed([109u8; 32]);
    let params = setup::<Mersenne127, Blake2s>(
        row_len,
        1,
        &mut rng,
        Some(num_rows),
        Some(inverse_rate),
        Some(8),
    );
    let (pp, vp) = trim::<Mersenne127, Blake2s>(&params, row_len, 1);
    let word = (0..num_rows)
        .map(|_| {
            (0..row_len)
                .map(|_| Mersenne127::random(&mut rng))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let mut cpu_transcript = TestTranscript::new(());
    let cpu_commitment = commit_and_write(&pp, &word, &mut cpu_transcript);
    let cpu_root_ref: &Output<Blake2s> = cpu_commitment.as_ref();
    let cpu_root = cpu_root_ref.clone();
    let cpu_leaves = cpu_commitment.codeword_tree[0].clone();
    drop(cpu_commitment);

    let mut cuda = CudaQuasarCommitter::new(&pp, &word, 4).unwrap();
    cuda.warm_up().unwrap();
    let mut prover_transcript = TestTranscript::new(());
    let (cuda_commitment, _) = cuda.commit_and_write(&pp, &mut prover_transcript).unwrap();
    let cuda_root: &Output<Blake2s> = cuda_commitment.as_ref();
    assert_eq!(cuda_commitment.codeword_tree[0], cpu_leaves);
    assert_eq!(cuda_root, &cpu_root);

    let total_vars = row_len.trailing_zeros() as usize + num_rows.trailing_zeros() as usize;
    let point = (0..total_vars)
        .map(|_| Mersenne127::random(&mut rng))
        .collect::<Vec<_>>();
    let (z_left, z_right) =
        qabase_split_evaluation_point(&point, num_rows, row_len.trailing_zeros() as usize);
    let flat = word.iter().flatten().copied().collect::<Vec<_>>();
    let value = eval_mle_from_evals(&flat, &point);

    prove_qabase_open_full_two_layer_gkr(
        &pp,
        &word,
        &cuda_commitment,
        z_left.clone(),
        z_right.clone(),
        value,
        &mut prover_transcript,
    )
    .unwrap();
    let proof = prover_transcript.into_proof();
    let mut verifier_transcript = TestTranscript::from_proof((), proof.as_slice());
    let (ok, _) = verify_qabase_open_full_two_layer_gkr(
        &vp,
        &cuda_commitment,
        z_left,
        z_right,
        value,
        &mut verifier_transcript,
    )
    .unwrap();
    assert!(ok);
    cuda.reclaim(cuda_commitment).unwrap();
}
