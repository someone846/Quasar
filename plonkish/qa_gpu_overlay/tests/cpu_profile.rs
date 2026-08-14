use plonkish_backend::{
    pcs::multilinear::quasar::QAParams,
    util::{arithmetic::Field, new_fields::Mersenne127},
};
use qa_gpu_overlay::cpu::{qa_encode_cpu_baseline_rows, qa_encode_cpu_profiled_rows};
use rand_chacha::{rand_core::SeedableRng, ChaCha8Rng};

#[test]
fn profiled_cpu_encoder_matches_current_encoder_rate_2_and_4() {
    for inverse_rate in [2, 4] {
        let mut rng = ChaCha8Rng::seed_from_u64(91 + inverse_rate as u64);
        let row_len = 1 << 10;
        let rows = 3;
        let params = QAParams::<Mersenne127>::new_random(row_len, inverse_rate, &mut rng);
        let messages = (0..rows * row_len)
            .map(|_| Mersenne127::random(&mut rng))
            .collect::<Vec<_>>();
        let baseline = qa_encode_cpu_baseline_rows(&messages, row_len, &params);
        let (profiled, _) = qa_encode_cpu_profiled_rows(&messages, row_len, &params);
        assert_eq!(baseline, profiled);
    }
}
