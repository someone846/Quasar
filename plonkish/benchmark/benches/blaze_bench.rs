use halo2_proofs::arithmetic::Field;
use itertools::chain;
use plonkish_backend::pcs::multilinear::blaze::log2_strict;
use plonkish_backend::pcs::Evaluation;
use plonkish_backend::util::binary_extension_fields::B128;
use plonkish_backend::util::blaze_transcript::{BlazeBlake2sTranscript, BlazeFiatShamirTranscript};
use plonkish_backend::util::transcript::FieldTranscriptRead;
use plonkish_backend::util::transcript::FieldTranscriptWrite;
use plonkish_backend::util::transcript::{
    Blake2s256Transcript, Blake2sTranscript, Keccak256Transcript,
};
use plonkish_backend::util::transcript::{
    FiatShamirTranscript, FieldTranscript, InMemoryTranscript, TranscriptRead, TranscriptWrite,
};
use rand::Rng;

use num_traits::Zero;

use benchmark::{
    espresso,
    halo2::{AggregationCircuit, Sha256Circuit},
    BasefoldParams::*,
};
use halo2_proofs::{
    plonk::{create_proof, keygen_pk, keygen_vk, verify_proof},
    poly::kzg::{
        commitment::ParamsKZG,
        multiopen::{ProverGWC, VerifierGWC},
        strategy::SingleStrategy,
    },
};
use itertools::Itertools;
use plonkish_backend::{
    backend::{self, PlonkishBackend, PlonkishCircuit},
    frontend::halo2::{circuit::VanillaPlonk, CircuitExt, Halo2Circuit},
    halo2_curves::{
        bn256::{Bn256, Fr},
        secp256k1::Fp,
    },
    pcs::{
        multilinear::{
            blaze::{
                commit as blaze_commit, commit_and_write as blaze_commit_and_write,
                faster_open as blaze_faster_open, faster_verify as blaze_faster_verify,
                open as blaze_open, setup as blaze_setup, trim as blaze_trim,
                verify as blaze_verify, BlazeCommitment, CommitmentChunk as BlazeCommitmentChunk,
            },
            Basefold, BasefoldExtParams, MultilinearBrakedown, MultilinearKzg, ZeromorphFri,
        },
        univariate::Fri,
        PolynomialCommitmentScheme,
    },
    poly::multilinear::MultilinearPolynomial,
    util::{
        arithmetic::{sum, PrimeField},
        avx_int_types::{u256::Blazeu256, u64::Blazeu64, BlazeField},
        code::{BrakedownSpec1, BrakedownSpec3, BrakedownSpec6},
        end_timer,
        ff_255::{ff255::Ft255, ft127::Ft127, ft63::Ft63},
        goldilocksMont::GoldilocksMont,
        hash::{Blake2s, Blake2s256, Hash, Keccak256},
        mersenne_61_mont::Mersenne61Mont,
        new_fields::{Mersenne127, Mersenne61},
        start_timer,
        test::std_rng,
    },
};
use rand::{rngs::OsRng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use std::{
    env::args,
    fmt::Display,
    fs::{create_dir, File, OpenOptions},
    io::Write,
    iter,
    ops::Range,
    path::Path,
    time::{Duration, Instant},
};

const BATCH_SIZE: usize = 64;
const OUTPUT_DIR: &str = "./bench_data/pcs";
const CSV_FILE: &str = "results.csv";
use std::env;

#[derive(Debug)]
struct P {}

impl BasefoldExtParams for P {
    fn get_rate() -> usize {
        return 2;
    }

    fn get_basecode_rounds() -> usize {
        return 2;
    }

    fn get_reps() -> usize {
        return 1000;
    }

    fn get_rs_basecode() -> bool {
        true
    }
    fn get_code_type() -> String {
        "random".to_string()
    }
}
fn main() {
    let (systems, k_range) = parse_args();
    create_output(&systems);
    k_range.for_each(|k| systems.iter().for_each(|system| system.bench(k)));
}

fn csv_output_path() -> String {
    format!("{OUTPUT_DIR}/{CSV_FILE}")
}

fn csv_output() -> File {
    OpenOptions::new()
        .append(true)
        .open(csv_output_path())
        .unwrap()
}

fn write_csv_header() {
    let mut file = File::create(csv_output_path()).unwrap();
    writeln!(
        &mut file,
        "system,variant,row_k,total_k,log_rows,num_rows,queries,samples,commit_ms,open_ms,verify_ms,proof_bytes,proof_kb"
    )
    .unwrap();
}

fn write_blaze_csv_row(
    system: System,
    variant: &str,
    k: usize,
    log_rows: usize,
    num_rows: usize,
    queries: usize,
    samples: usize,
    commit_ms: u128,
    open_ms: u128,
    verify_ms: u128,
    proof_bytes: usize,
) {
    writeln!(
        &mut csv_output(),
        "{},{},{},{},{},{},{},{},{},{},{},{},{:.2}",
        system,
        variant,
        k,
        k + log_rows,
        log_rows,
        num_rows,
        queries,
        samples,
        commit_ms,
        open_ms,
        verify_ms,
        proof_bytes,
        proof_bytes as f64 / 1024.0,
    )
    .unwrap();
}

fn bench_pcs<F, Pcs, T>(k: usize, pcs: System)
where
    F: PrimeField,
    Pcs: PolynomialCommitmentScheme<F, Polynomial = MultilinearPolynomial<F>>,
    T: TranscriptRead<Pcs::CommitmentChunk, F>
        + TranscriptWrite<Pcs::CommitmentChunk, F>
        + InMemoryTranscript<Param = ()>,
{
    let mut rng = OsRng;
    let poly_size = 1 << k;
    let param = Pcs::setup(poly_size, 1, &mut rng).unwrap();

    let (pp, vp) = Pcs::trim(&param, poly_size, 1).unwrap();

    let mut transcript = T::new(());

    let poly = MultilinearPolynomial::rand(k, OsRng);

    let sample_size = sample_size(k);

    let mut commit_times = Vec::new();
    let mut times = Vec::new();
    for i in 0..sample_size {
        println!("commit");
        let cstart = Instant::now();
        let comm = Pcs::commit_and_write(&pp, &poly, &mut transcript).unwrap();

        commit_times.push(cstart.elapsed());

        let start = Instant::now();
        let point = transcript.squeeze_challenges(k);
        let eval = poly.evaluate(point.as_slice());
        transcript.write_field_element(&eval).unwrap();
        Pcs::open(&pp, &poly, &comm, &point, &eval, &mut transcript).unwrap();
        times.push(start.elapsed());
    }
    let sum = times[2..5].iter().sum::<Duration>();
    let csum = commit_times[2..5].iter().sum::<Duration>();

    let avg = sum / 3 as u32;
    let cavg = csum / 3 as u32;

    writeln!(&mut pcs.commit_output(), "{k}, {}", cavg.as_millis()).unwrap();
    writeln!(&mut pcs.output(), "{k}, {}", avg.as_millis()).unwrap();

    let mut end_size = 0;
    while transcript.read_commitment().is_ok() {
        end_size = end_size + 1;
    }
    writeln!(
        &mut pcs.size_output(),
        "{:?} {:?} : {:?}",
        pcs,
        k,
        (end_size) * 256
    );

    let proof = transcript.into_proof();
    //let timer = start_timer(|| format!("verify-{k}"));

    let mut verifier_times = Vec::new();
    for i in 0..sample_size {
        let proof = proof.clone();
        let mut transcript = T::from_proof((), proof.as_slice());
        let now = Instant::now();
        let b = Pcs::verify(
            &vp,
            &Pcs::read_commitment(&vp, &mut transcript).unwrap(),
            &transcript.squeeze_challenges(k),
            &transcript.read_field_element().unwrap(),
            &mut transcript,
        );
        verifier_times.push(now.elapsed());
    }
    let vsum = verifier_times[2..5].iter().sum::<Duration>();
    let vavg = vsum / 3 as u32;
    writeln!(&mut pcs.verify_output(), "{:?}: {:?}", k, vavg.as_millis()).unwrap();
    //let mut end_size = 0;
    //while transcript.read_commitment().is_ok() {
    //    end_size = end_size + 1;
    //}
    //writeln!(&mut pcs.size_output(), "{:?} {:?} : {:?}", pcs, k, (start_size - end_size)*256);
    //b

    //  end_timer(timer);
    //  assert_eq!(result,Ok(()));
}

fn batch_bench_pcs<F, Pcs, T>(k: usize, batch_size: usize, pcs: System)
where
    F: PrimeField,
    Pcs: PolynomialCommitmentScheme<F, Polynomial = MultilinearPolynomial<F>>,
    T: TranscriptRead<Pcs::CommitmentChunk, F>
        + TranscriptWrite<Pcs::CommitmentChunk, F>
        + InMemoryTranscript<Param = ()>,
{
    let mut rng = OsRng;
    let poly_size = 1 << k;
    let num_points = batch_size >> 1;

    let evals = chain![
        (0..num_points).map(|point| (0, point)),
        (0..batch_size).map(|poly| (poly, 0)),
        iter::repeat_with(|| (rng.gen_range(0..batch_size), rng.gen_range(0..num_points)))
            .take(batch_size)
    ]
    .unique()
    .collect_vec();

    let param = Pcs::setup(poly_size, 1, &mut rng).unwrap();

    let (pp, vp) = Pcs::trim(&param, poly_size, 1).unwrap();

    let mut transcript = T::new(());

    let sample_size = sample_size(k);

    let mut commit_times = Vec::new();
    let mut times = Vec::new();
    let polys = iter::repeat_with(|| MultilinearPolynomial::rand(k, OsRng))
        .take(batch_size)
        .collect_vec();
    for i in 0..sample_size {
        let cstart = Instant::now();
        let comms = Pcs::batch_commit_and_write(&pp, &polys, &mut transcript).unwrap();

        commit_times.push(cstart.elapsed());

        let points = iter::repeat_with(|| transcript.squeeze_challenges(k))
            .take(num_points)
            .collect_vec();

        let evals = evals
            .iter()
            .copied()
            .map(|(poly, point)| Evaluation {
                poly,
                point,
                value: polys[poly].evaluate(&points[point]),
            })
            .collect_vec();

        transcript
            .write_field_elements(evals.iter().map(Evaluation::value))
            .unwrap();
        let start = Instant::now();
        Pcs::batch_open(&pp, &polys, &comms, &points, &evals, &mut transcript).unwrap();

        times.push(start.elapsed());
    }
    let sum = times.iter().sum::<Duration>();
    let csum = commit_times.iter().sum::<Duration>();

    let avg = sum / sample_size as u32;
    let cavg = csum / sample_size as u32;

    writeln!(&mut pcs.batch_commit_output(), "{k}, {}", cavg.as_millis()).unwrap();
    writeln!(&mut pcs.batch_open_output(), "{k}, {}", avg.as_millis()).unwrap();

    let mut end_size = 0;
    while transcript.read_commitment().is_ok() {
        end_size = end_size + 1;
    }
    writeln!(
        &mut pcs.size_output(),
        "{:?} {:?} : {:?}",
        pcs,
        k,
        (end_size) * 256
    );

    //let timer = start_timer(|| format!("verify-{k}"));
    //let result = {
    //let mut transcript = T::from_proof((),proof.as_slice());
    //let mut start_size = 0;
    //while transcript.read_commitment().is_ok() {
    //   start_size = start_size + 1;T::from_proof((),proof.as_slice());
    //}

    //let mut transcript = T::from_proof((),proof.as_slice());
    //let now = Instant::now();
    //let b = Pcs::verify(
    //      &vp,
    //    &Pcs::read_commitment(&vp, &mut transcript).unwrap(),
    //  &transcript.squeeze_challenges(k),
    //&transcript.read_field_element().unwrap(),
    //&mut transcript
    //);
    //writeln!(&mut pcs.verify_output(), "{:?}: {:?}", k, now.elapsed().as_millis()).unwrap();
    //let mut end_size = 0;
    //while transcript.read_commitment().is_ok() {
    //    end_size = end_size + 1;
    //}
    //writeln!(&mut pcs.size_output(), "{:?} {:?} : {:?}", pcs, k, (start_size - end_size)*256);
    //b

    //  end_timer(timer);
    //  assert_eq!(result,Ok(()));
}

fn bench_blaze<F, H, T1, T2>(k: usize, pcs: System, log_rows: usize, queries: usize)
where
    F: BlazeField,
    H: Hash,
    T1: TranscriptRead<BlazeCommitmentChunk<H>, B128>
        + TranscriptWrite<BlazeCommitmentChunk<H>, B128>
        + InMemoryTranscript<Param = ()>,
    T2: TranscriptRead<BlazeCommitmentChunk<H>, F>
        + TranscriptWrite<BlazeCommitmentChunk<H>, F>
        + InMemoryTranscript<Param = ()>,
{
    let mut rng = OsRng;
    let num_rows = 1 << log_rows;
    let poly_size = 1 << k;
    let param = blaze_setup::<H>(poly_size, 2, &mut rng, Some(num_rows), Some(queries));
    let (pp, vp) = blaze_trim::<H>(&param, poly_size, 1);

    // Create one fixed random matrix and benchmark independent proofs for it.
    // Important: each sample gets fresh transcripts.  Otherwise proof_size is
    // the concatenation of all samples, while verifier time measures only the
    // first proof from the concatenated transcript.
    let mut matrix = Vec::new();
    for _ in 0..num_rows {
        matrix.push(F::rand_vec(poly_size as usize));
    }

    let sample_size = sample_size(k);
    let mut commit_times = Vec::new();
    let mut open_times = Vec::new();
    let mut verifier_times = Vec::new();
    let mut proof_sizes = Vec::new();

    for _ in 0..sample_size {
        let mut b128_transcript = T1::new(());
        let mut blaze_transcript = T2::new(());

        let cstart = Instant::now();
        let comm = blaze_commit_and_write::<F, H>(&pp, &matrix, &mut blaze_transcript);
        commit_times.push(cstart.elapsed());

        let point = b128_transcript.squeeze_challenges(k);

        let open_start = Instant::now();
        blaze_open(
            &pp,
            &matrix,
            &comm,
            &point,
            &B128::ZERO,
            &mut blaze_transcript,
            &mut b128_transcript,
        )
        .unwrap();
        open_times.push(open_start.elapsed());

        let blaze_proof = blaze_transcript.into_proof();
        let b128_proof = b128_transcript.into_proof();
        proof_sizes.push(blaze_proof.len() + b128_proof.len());

        let mut verify_blaze_transcript = T2::from_proof((), blaze_proof.as_slice());
        let mut verify_b128_transcript = T1::from_proof((), b128_proof.as_slice());
        let verify_start = Instant::now();
        blaze_verify(
            &vp,
            &comm,
            &point,
            &F::zero(),
            &mut verify_b128_transcript,
            &mut verify_blaze_transcript,
        )
        .unwrap();
        verifier_times.push(verify_start.elapsed());
    }

    let commit_sum = commit_times[2..5].iter().sum::<Duration>();
    let open_sum = open_times[2..5].iter().sum::<Duration>();
    let verify_sum = verifier_times[2..5].iter().sum::<Duration>();

    let cavg = commit_sum / 3 as u32;
    let oavg = open_sum / 3 as u32;
    let vavg = verify_sum / 3 as u32;
    let proof_bytes = proof_sizes[2..5].iter().sum::<usize>() / 3;

    writeln!(&mut pcs.commit_output(), "{k}, {}", cavg.as_millis()).unwrap();
    writeln!(&mut pcs.output(), "{k}, {}", oavg.as_millis()).unwrap();
    writeln!(&mut pcs.verify_output(), "{:?}: {:?}", k, vavg.as_millis()).unwrap();

    write_blaze_csv_row(
        pcs,
        "blaze_open",
        k,
        log_rows,
        num_rows,
        queries,
        sample_size,
        cavg.as_millis(),
        oavg.as_millis(),
        vavg.as_millis(),
        proof_bytes,
    );
}

fn bench_faster_blaze<F, H, T1, T2>(k: usize, pcs: System, log_rows: usize, queries: usize)
where
    F: BlazeField,
    H: Hash,
    T1: TranscriptRead<BlazeCommitmentChunk<H>, B128>
        + TranscriptWrite<BlazeCommitmentChunk<H>, B128>
        + InMemoryTranscript<Param = ()>,
    T2: TranscriptRead<BlazeCommitmentChunk<H>, F>
        + TranscriptWrite<BlazeCommitmentChunk<H>, F>
        + InMemoryTranscript<Param = ()>,
{
    let mut rng = OsRng;
    let num_rows = 1 << log_rows;
    let poly_size = 1 << k;
    let param = blaze_setup::<H>(poly_size, 1, &mut rng, Some(num_rows), Some(queries));
    let (pp, vp) = blaze_trim::<H>(&param, poly_size, 1);

    let mut matrix = Vec::new();
    for _ in 0..num_rows {
        matrix.push(F::rand_vec(poly_size as usize));
    }

    let sample_size = sample_size(k);
    let mut commit_times = Vec::new();
    let mut open_times = Vec::new();
    let mut verifier_times = Vec::new();
    let mut proof_sizes = Vec::new();

    for _ in 0..sample_size {
        let mut b128_transcript = T1::new(());
        let mut blaze_transcript = T2::new(());

        let cstart = Instant::now();
        let comm = blaze_commit_and_write::<F, H>(&pp, &matrix, &mut blaze_transcript);
        commit_times.push(cstart.elapsed());

        let point = b128_transcript.squeeze_challenges(k);

        let open_start = Instant::now();
        blaze_faster_open(
            &pp,
            &matrix,
            &comm,
            &point,
            &B128::ZERO,
            &mut blaze_transcript,
            &mut b128_transcript,
        )
        .unwrap();
        open_times.push(open_start.elapsed());

        let blaze_proof = blaze_transcript.into_proof();
        let b128_proof = b128_transcript.into_proof();
        proof_sizes.push(blaze_proof.len() + b128_proof.len());

        let mut verify_blaze_transcript = T2::from_proof((), blaze_proof.as_slice());
        let mut verify_b128_transcript = T1::from_proof((), b128_proof.as_slice());
        let verify_start = Instant::now();
        blaze_faster_verify(
            &vp,
            &comm,
            &point,
            &F::zero(),
            &mut verify_b128_transcript,
            &mut verify_blaze_transcript,
        )
        .unwrap();
        verifier_times.push(verify_start.elapsed());
    }

    let commit_sum = commit_times[2..5].iter().sum::<Duration>();
    let open_sum = open_times[2..5].iter().sum::<Duration>();
    let verify_sum = verifier_times[2..5].iter().sum::<Duration>();

    let cavg = commit_sum / 3 as u32;
    let oavg = open_sum / 3 as u32;
    let vavg = verify_sum / 3 as u32;
    let proof_bytes = proof_sizes[2..5].iter().sum::<usize>() / 3;

    writeln!(&mut pcs.commit_output(), "{k}, {}", cavg.as_millis()).unwrap();
    writeln!(&mut pcs.output(), "{k}, {}", oavg.as_millis()).unwrap();
    writeln!(&mut pcs.verify_output(), "{:?}: {:?}", k, vavg.as_millis()).unwrap();

    write_blaze_csv_row(
        pcs,
        "faster_open",
        k,
        log_rows,
        num_rows,
        queries,
        sample_size,
        cavg.as_millis(),
        oavg.as_millis(),
        vavg.as_millis(),
        proof_bytes,
    );
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum System {
    MultilinearKzg,
    Basefold256,
    Basefold61Mersenne,
    BasefoldBlake2s,
    Brakedown,
    BrakedownBlake2s,
    ZeromorphFri,
    Blaze64,
    BasefoldFri2,
    BasefoldFri4,
    BasefoldFri8,
    Brakedown1,
    Brakedown3,
    Brakedown6,
}

impl System {
    fn all() -> Vec<System> {
        vec![
            System::MultilinearKzg,
            System::Basefold61Mersenne,
            System::Basefold256,
            System::Brakedown,
            System::BasefoldBlake2s,
            System::BrakedownBlake2s,
            System::ZeromorphFri,
            System::Blaze64,
            System::BasefoldFri2,
            System::BasefoldFri4,
            System::BasefoldFri8,
            System::Brakedown1,
            System::Brakedown3,
            System::Brakedown6,
        ]
    }

    fn output_path(&self) -> String {
        format!("{OUTPUT_DIR}/open_{self}")
    }

    fn output(&self) -> File {
        OpenOptions::new()
            .append(true)
            .open(self.output_path())
            .unwrap()
    }
    fn batch_open_output_path(&self) -> String {
        format!("{OUTPUT_DIR}/batch_open_{self}")
    }

    fn commit_output_path(&self) -> String {
        format!("{OUTPUT_DIR}/commit_{self}")
    }
    fn batch_commit_output_path(&self) -> String {
        format!("{OUTPUT_DIR}/batch_commit_{self}")
    }

    fn size_output_path(&self) -> String {
        format!("{OUTPUT_DIR}/size_{self}")
    }

    fn verify_output_path(&self) -> String {
        format!("{OUTPUT_DIR}/verify_{self}")
    }

    fn commit_output(&self) -> File {
        OpenOptions::new()
            .append(true)
            .open(self.commit_output_path())
            .unwrap()
    }
    fn batch_commit_output(&self) -> File {
        OpenOptions::new()
            .append(true)
            .open(self.batch_commit_output_path())
            .unwrap()
    }

    fn batch_open_output(&self) -> File {
        OpenOptions::new()
            .append(true)
            .open(self.batch_open_output_path())
            .unwrap()
    }
    fn size_output(&self) -> File {
        OpenOptions::new()
            .append(true)
            .open(self.size_output_path())
            .unwrap()
    }

    fn verify_output(&self) -> File {
        OpenOptions::new()
            .append(true)
            .open(self.verify_output_path())
            .unwrap()
    }

    fn bench(&self, k: usize) {
        type Kzg = MultilinearKzg<Bn256>;
        type Brakedown = MultilinearBrakedown<Fp, Keccak256, BrakedownSpec6>;
        type Brakedown127 = MultilinearBrakedown<Fp, Blake2s, BrakedownSpec6>;
        type BrakedownBlake2s = MultilinearBrakedown<GoldilocksMont, Blake2s, BrakedownSpec1>;
        type Brakedown1 = MultilinearBrakedown<Mersenne127, Blake2s, BrakedownSpec1>;
        type Brakedown3 = MultilinearBrakedown<Mersenne127, Blake2s, BrakedownSpec3>;
        type Brakedown6 = MultilinearBrakedown<GoldilocksMont, Blake2s, BrakedownSpec6>;

        match self {
            System::MultilinearKzg => {
                bench_pcs::<_, Kzg, Blake2sTranscript<_>>(k, System::MultilinearKzg)
            }
            System::Basefold61Mersenne => bench_pcs::<
                Mersenne127,
                Basefold<Mersenne127, Blake2s, P>,
                Blake2sTranscript<_>,
            >(k, System::Basefold61Mersenne),
            System::Basefold256 => bench_pcs::<
                Fr,
                Basefold<Fr, Blake2s256, BasefoldFri>,
                Blake2s256Transcript<_>,
            >(k, System::Basefold256),
            System::Brakedown => {
                bench_pcs::<Fp, Brakedown, Keccak256Transcript<_>>(k, System::Brakedown)
            }
            System::BasefoldBlake2s => match k {
                10 => bench_pcs::<
                    GoldilocksMont,
                    Basefold<GoldilocksMont, Blake2s, Ten>,
                    Blake2sTranscript<_>,
                >(k, System::BasefoldBlake2s),
                11 => bench_pcs::<
                    GoldilocksMont,
                    Basefold<GoldilocksMont, Blake2s, Eleven>,
                    Blake2sTranscript<_>,
                >(k, System::BasefoldBlake2s),
                12 => bench_pcs::<
                    GoldilocksMont,
                    Basefold<GoldilocksMont, Blake2s, Twelve>,
                    Blake2sTranscript<_>,
                >(k, System::BasefoldBlake2s),
                13 => bench_pcs::<
                    GoldilocksMont,
                    Basefold<GoldilocksMont, Blake2s, Thirteen>,
                    Blake2sTranscript<_>,
                >(k, System::BasefoldBlake2s),
                14 => bench_pcs::<
                    GoldilocksMont,
                    Basefold<GoldilocksMont, Blake2s, Fourteen>,
                    Blake2sTranscript<_>,
                >(k, System::BasefoldBlake2s),
                15 => bench_pcs::<
                    GoldilocksMont,
                    Basefold<GoldilocksMont, Blake2s, Fifteen>,
                    Blake2sTranscript<_>,
                >(k, System::BasefoldBlake2s),
                16 => bench_pcs::<
                    GoldilocksMont,
                    Basefold<GoldilocksMont, Blake2s, Sixteen>,
                    Blake2sTranscript<_>,
                >(k, System::BasefoldBlake2s),
                17 => bench_pcs::<
                    GoldilocksMont,
                    Basefold<GoldilocksMont, Blake2s, Seventeen>,
                    Blake2sTranscript<_>,
                >(k, System::BasefoldBlake2s),
                18 => bench_pcs::<
                    GoldilocksMont,
                    Basefold<GoldilocksMont, Blake2s, Eighteen>,
                    Blake2sTranscript<_>,
                >(k, System::BasefoldBlake2s),
                19 => bench_pcs::<
                    GoldilocksMont,
                    Basefold<GoldilocksMont, Blake2s, Nineteen>,
                    Blake2sTranscript<_>,
                >(k, System::BasefoldBlake2s),
                20 => bench_pcs::<
                    GoldilocksMont,
                    Basefold<GoldilocksMont, Blake2s, Twenty>,
                    Blake2sTranscript<_>,
                >(k, System::BasefoldBlake2s),
                21 => bench_pcs::<
                    GoldilocksMont,
                    Basefold<GoldilocksMont, Blake2s, TwentyOne>,
                    Blake2sTranscript<_>,
                >(k, System::BasefoldBlake2s),
                22 => bench_pcs::<
                    GoldilocksMont,
                    Basefold<GoldilocksMont, Blake2s, TwentyTwo>,
                    Blake2sTranscript<_>,
                >(k, System::BasefoldBlake2s),
                23 => bench_pcs::<
                    GoldilocksMont,
                    Basefold<GoldilocksMont, Blake2s, TwentyThree>,
                    Blake2sTranscript<_>,
                >(k, System::BasefoldBlake2s),
                24 => bench_pcs::<
                    GoldilocksMont,
                    Basefold<GoldilocksMont, Blake2s, TwentyFour>,
                    Blake2sTranscript<_>,
                >(k, System::BasefoldBlake2s),
                25 => bench_pcs::<
                    GoldilocksMont,
                    Basefold<GoldilocksMont, Blake2s, TwentyFive>,
                    Blake2sTranscript<_>,
                >(k, System::BasefoldBlake2s),
                26 => bench_pcs::<
                    GoldilocksMont,
                    Basefold<GoldilocksMont, Blake2s, TwentySix>,
                    Blake2sTranscript<_>,
                >(k, System::BasefoldBlake2s),
                _ => {}
            },

            System::BrakedownBlake2s => bench_pcs::<
                GoldilocksMont,
                BrakedownBlake2s,
                Blake2sTranscript<_>,
            >(k, System::BrakedownBlake2s),
            System::ZeromorphFri => {
                bench_pcs::<
                    GoldilocksMont,
                    ZeromorphFri<Fri<GoldilocksMont, Blake2s>>,
                    Blake2sTranscript<_>,
                >(22, System::ZeromorphFri);
                bench_pcs::<
                    GoldilocksMont,
                    ZeromorphFri<Fri<GoldilocksMont, Blake2s>>,
                    Blake2sTranscript<_>,
                >(23, System::ZeromorphFri);
                bench_pcs::<
                    GoldilocksMont,
                    ZeromorphFri<Fri<GoldilocksMont, Blake2s>>,
                    Blake2sTranscript<_>,
                >(24, System::ZeromorphFri);
                bench_pcs::<
                    GoldilocksMont,
                    ZeromorphFri<Fri<GoldilocksMont, Blake2s>>,
                    Blake2sTranscript<_>,
                >(25, System::ZeromorphFri);
                bench_pcs::<
                    GoldilocksMont,
                    ZeromorphFri<Fri<GoldilocksMont, Blake2s>>,
                    Blake2sTranscript<_>,
                >(26, System::ZeromorphFri);
                bench_pcs::<
                    GoldilocksMont,
                    ZeromorphFri<Fri<GoldilocksMont, Blake2s>>,
                    Blake2sTranscript<_>,
                >(27, System::ZeromorphFri);
                bench_pcs::<
                    GoldilocksMont,
                    ZeromorphFri<Fri<GoldilocksMont, Blake2s>>,
                    Blake2sTranscript<_>,
                >(28, System::ZeromorphFri);
            }
            System::Blaze64 => {
                // Article-baseline Blaze opening benchmark for total sizes 2^25 through 2^30.
                // We keep row_k fixed at 21, so total_k = row_k + log_rows.
                // Therefore log_rows = 4..=9 corresponds to total_k = 25..=30.
                for log_rows in 4..=9 {
                    bench_blaze::<Blazeu64, Blake2s, Blake2sTranscript<_>, BlazeBlake2sTranscript<_>>(
                        21,
                        System::Blaze64,
                        log_rows,
                        2004,
                    );
                }
            }
            System::BasefoldFri2 => {
                bench_pcs::<
                    GoldilocksMont,
                    Basefold<GoldilocksMont, Blake2s, BasefoldFri>,
                    Blake2sTranscript<_>,
                >(22, System::BasefoldFri2);
                bench_pcs::<
                    GoldilocksMont,
                    Basefold<GoldilocksMont, Blake2s, BasefoldFri>,
                    Blake2sTranscript<_>,
                >(23, System::BasefoldFri2);
                bench_pcs::<
                    GoldilocksMont,
                    Basefold<GoldilocksMont, Blake2s, BasefoldFri>,
                    Blake2sTranscript<_>,
                >(24, System::BasefoldFri2);
                bench_pcs::<
                    GoldilocksMont,
                    Basefold<GoldilocksMont, Blake2s, BasefoldFri>,
                    Blake2sTranscript<_>,
                >(25, System::BasefoldFri2);
                bench_pcs::<
                    GoldilocksMont,
                    Basefold<GoldilocksMont, Blake2s, BasefoldFri>,
                    Blake2sTranscript<_>,
                >(26, System::BasefoldFri2);
                bench_pcs::<
                    GoldilocksMont,
                    Basefold<GoldilocksMont, Blake2s, BasefoldFri>,
                    Blake2sTranscript<_>,
                >(27, System::BasefoldFri2);
                bench_pcs::<
                    GoldilocksMont,
                    Basefold<GoldilocksMont, Blake2s, BasefoldFri>,
                    Blake2sTranscript<_>,
                >(28, System::BasefoldFri2);
                bench_pcs::<
                    GoldilocksMont,
                    Basefold<GoldilocksMont, Blake2s, BasefoldFri>,
                    Blake2sTranscript<_>,
                >(29, System::BasefoldFri2);
            }
            System::BasefoldFri4 => bench_pcs::<
                Mersenne127,
                Basefold<Mersenne127, Blake2s, BasefoldFriR4>,
                Blake2sTranscript<_>,
            >(k, System::BasefoldFri4),
            System::BasefoldFri8 => bench_pcs::<
                Mersenne127,
                Basefold<Mersenne127, Blake2s, BasefoldFriR8>,
                Blake2sTranscript<_>,
            >(k, System::BasefoldFri8),
            System::Brakedown1 => {
                bench_pcs::<Mersenne127, Brakedown1, Blake2sTranscript<_>>(k, System::Brakedown1)
            }
            System::Brakedown3 => {
                bench_pcs::<Mersenne127, Brakedown3, Blake2sTranscript<_>>(k, System::Brakedown3)
            }
            System::Brakedown6 => {
                bench_pcs::<GoldilocksMont, Brakedown6, Blake2sTranscript<_>>(
                    30,
                    System::Brakedown6,
                );
                bench_pcs::<GoldilocksMont, Brakedown6, Blake2sTranscript<_>>(
                    31,
                    System::Brakedown6,
                );
                bench_pcs::<GoldilocksMont, Brakedown6, Blake2sTranscript<_>>(
                    32,
                    System::Brakedown6,
                );
            } //        System::BrakedownBlake2s => batch_bench_pcs::<GoldilocksMont, BrakedownBlake2s, Blake2sTranscript<_>>(k,BATCH_SIZE,System::BrakedownBlake2s),
              //       System::ZeromorphFri => batch_bench_pcs::<Fr, ZeromorphFri<Fri<Fr,Blake2s>>, Blake2sTranscript<_>>(k, BATCH_SIZE,System::ZeromorphFri),
              //        System::BasefoldFri2 => batch_bench_pcs::<GoldilocksMont, Basefold<GoldilocksMont, Blake2s, BasefoldFriR2>,Blake2sTranscript<_>>(k, BATCH_SIZE, System::BasefoldFri2),
              //        System::BasefoldFri4 => batch_bench_pcs::<GoldilocksMont, Basefold<GoldilocksMont, Blake2s, BasefoldFriR4>,Blake2sTranscript<_>>(k, BATCH_SIZE,System::BasefoldFri4),
              //        System::BasefoldFri8 => batch_bench_pcs::<GoldilocksMont, Basefold<GoldilocksMont, Blake2s, BasefoldFriR8>,Blake2sTranscript<_>>(k,BATCH_SIZE, System::BasefoldFri8),
              //        System::Brakedown1 =>  batch_bench_pcs::<Mersenne127, Brakedown1, Blake2sTranscript<_>>(k,BATCH_SIZE,System::Brakedown1),
              //        System::Brakedown3 =>  batch_bench_pcs::<Mersenne127, Brakedown3, Blake2sTranscript<_>>(k,BATCH_SIZE,System::Brakedown3),
              //        System::Brakedown6 =>  batch_bench_pcs::<Mersenne127, Brakedown6, Blake2sTranscript<_>>(k,BATCH_SIZE,System::Brakedown6)
        }
    }
}

impl Display for System {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            System::Basefold256 => write!(f, "basefold256"),
            System::Basefold61Mersenne => write!(f, "basefold61Mersenne"),
            System::MultilinearKzg => write!(f, "kzg"),
            System::Brakedown => write!(f, "brakedown"),
            System::BrakedownBlake2s => write!(f, "brakedown_blake"),
            System::BasefoldBlake2s => write!(f, "basefold_blake"),
            System::ZeromorphFri => write!(f, "zeromorph_fri"),
            System::Blaze64 => write!(f, "blaze64"),
            System::BasefoldFri2 => write!(f, "basefoldfri2"),
            System::BasefoldFri4 => write!(f, "basefoldfri4"),
            System::BasefoldFri8 => write!(f, "basefoldfri8"),
            System::Brakedown1 => write!(f, "brakedown1"),
            System::Brakedown3 => write!(f, "brakedown3"),
            System::Brakedown6 => write!(f, "brakedown6"),
        }
    }
}

fn parse_args() -> (Vec<System>, Range<usize>) {
    let (systems, k_range) = args().chain(Some("".to_string())).tuple_windows().fold(
        (Vec::new(), 15..16),
        |(mut systems, mut k_range), (key, value)| {
            match key.as_str() {
                "--system" => match value.as_str() {
                    "all" => systems = System::all(),
                    "blaze64" | "blaze" => systems.push(System::Blaze64),
                    "basefold256" => systems.push(System::Basefold256),
                    "basefold61Mersenne" | "basefold61mersenne" => {
                        systems.push(System::Basefold61Mersenne)
                    }
                    "multilinearkzg" | "kzg" => systems.push(System::MultilinearKzg),
                    "brakedown" => systems.push(System::Brakedown),
                    "brakedown_blake" => systems.push(System::BrakedownBlake2s),
                    "basefold_blake" => systems.push(System::BasefoldBlake2s),
                    "zeromorph_fri" => systems.push(System::ZeromorphFri),
                    "basefoldfri2" => systems.push(System::BasefoldFri2),
                    "basefoldfri4" => systems.push(System::BasefoldFri4),
                    "basefoldfri8" => systems.push(System::BasefoldFri8),
                    "brakedown1" => systems.push(System::Brakedown1),
                    "brakedown3" => systems.push(System::Brakedown3),
                    "brakedown6" => systems.push(System::Brakedown6),
                    _ => panic!(
                        "system should be one of {{all,blaze64,blaze,basefold256,basefold61Mersenne,multilinearkzg,kzg,brakedown,brakedown_blake,basefold_blake,zeromorph_fri,basefoldfri2,basefoldfri4,basefoldfri8,brakedown1,brakedown3,brakedown6}}"
                    ),
                },
                "--k" => {
                    if let Some((start, end)) = value.split_once("..") {
                        k_range = start.parse().expect("k range start to be usize")
                            ..end.parse().expect("k range end to be usize");
                    } else {
                        k_range.start = value.parse().expect("k to be usize");
                        k_range.end = k_range.start + 1;
                    }
                }
                _ => {}
            }
            (systems, k_range)
        },
    );

    let mut systems = systems.into_iter().sorted().dedup().collect_vec();
    if systems.is_empty() {
        systems = vec![System::Blaze64];
    }
    (systems, k_range)
}

fn create_output(systems: &[System]) {
    if !Path::new(OUTPUT_DIR).exists() {
        create_dir(OUTPUT_DIR).unwrap();
    }
    write_csv_header();
    for system in systems {
        File::create(system.output_path()).unwrap();
        File::create(system.batch_open_output_path()).unwrap();
        File::create(system.commit_output_path()).unwrap();
        File::create(system.batch_commit_output_path()).unwrap();
        File::create(system.size_output_path()).unwrap();
        File::create(system.verify_output_path()).unwrap();
    }
}

fn sample<T>(system: System, k: usize, prove: impl Fn() -> T) -> T {
    let mut proof = None;
    let sample_size = sample_size(k);
    let sum = iter::repeat_with(|| {
        let start = Instant::now();
        proof = Some(prove());
        start.elapsed()
    })
    .take(sample_size)
    .sum::<Duration>();
    let avg = sum / sample_size as u32;
    writeln!(&mut system.output(), "{k}, {}", avg.as_millis()).unwrap();
    proof.unwrap()
}

fn sample_size(k: usize) -> usize {
    5
}
