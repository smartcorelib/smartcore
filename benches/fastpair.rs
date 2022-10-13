// use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

// // to run this bench you have to change the declaraion in mod.rs ---> pub mod fastpair;
// use smartcore::algorithm::neighbour::fastpair::FastPair;
// use smartcore::linalg::basic::arrays::matrix::densematrix;
// use std::time::Duration;

// fn closest_pair_bench(n: usize, m: usize) -> () {
//     let x = DenseMatrix::<f64>::rand(n, m);
//     let fastpair = FastPair::new(&x);
//     let result = fastpair.unwrap();

//     result.closest_pair();
// }

// fn closest_pair_brute_bench(n: usize, m: usize) -> () {
//     let x = DenseMatrix::<f64>::rand(n, m);
//     let fastpair = FastPair::new(&x);
//     let result = fastpair.unwrap();

//     result.closest_pair_brute();
// }

// fn bench_fastpair(c: &mut Criterion) {
//     let mut group = c.benchmark_group("FastPair");

//     // with full samples size (100) the test will take too long
//     group.significance_level(0.1).sample_size(30);
//     // increase from default 5.0 secs
//     group.measurement_time(Duration::from_secs(60));

//     for n_samples in [100_usize, 1000_usize].iter() {
//         for n_features in [10_usize, 100_usize, 1000_usize].iter() {
//             group.bench_with_input(
//                 BenchmarkId::from_parameter(format!(
//                     "fastpair --- n_samples: {}, n_features: {}",
//                     n_samples, n_features
//                 )),
//                 n_samples,
//                 |b, _| b.iter(|| closest_pair_bench(*n_samples, *n_features)),
//             );
//             group.bench_with_input(
//                 BenchmarkId::from_parameter(format!(
//                     "brute --- n_samples: {}, n_features: {}",
//                     n_samples, n_features
//                 )),
//                 n_samples,
//                 |b, _| b.iter(|| closest_pair_brute_bench(*n_samples, *n_features)),
//             );
//         }
//     }
//     group.finish();
// }

// criterion_group!(benches, bench_fastpair);
// criterion_main!(benches);
