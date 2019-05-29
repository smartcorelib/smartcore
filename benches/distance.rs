#[macro_use]
extern crate criterion;
extern crate smartcore;
extern crate ndarray;
use ndarray::Array;
use smartcore::math::distance::euclidian::EuclidianDistance;
use smartcore::math::distance::Distance;

use criterion::Criterion;
use criterion::black_box;

fn criterion_benchmark(c: &mut Criterion) {
    let a = Array::from_vec(vec![1., 2., 3.]);

    c.bench_function("Euclidean Distance", move |b| b.iter(|| EuclidianDistance::distance(black_box(&a), black_box(&a))));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);