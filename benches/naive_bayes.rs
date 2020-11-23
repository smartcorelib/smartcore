use criterion::BenchmarkId;
use criterion::{criterion_group, criterion_main, Criterion};

use nalgebra::DMatrix;
use ndarray::Array2;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::linalg::BaseMatrix;
use smartcore::linalg::BaseVector;
use smartcore::naive_bayes::GaussianNB;

pub fn gaussian_naive_bayes_fit_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("GaussianNB::fit");

    for n_samples in [100_usize, 1000_usize, 10000_usize].iter() {
        for n_features in [10_usize, 100_usize, 1000_usize].iter() {
            let x = DenseMatrix::<f64>::rand(*n_samples, *n_features);
            let y: Vec<f64> = (0..*n_samples)
                .map(|i| (i % *n_samples / 5_usize) as f64)
                .collect::<Vec<f64>>();
            group.bench_with_input(
                BenchmarkId::from_parameter(format!(
                    "n_samples: {}, n_features: {}",
                    n_samples, n_features
                )),
                n_samples,
                |b, _| {
                    b.iter(|| {
                        GaussianNB::fit(&x, &y, Default::default()).unwrap();
                    })
                },
            );
        }
    }
    group.finish();
}

pub fn gaussian_naive_matrix_datastructure(c: &mut Criterion) {
    let mut group = c.benchmark_group("GaussianNB");
    let classes = (0..10000).map(|i| (i % 25) as f64).collect::<Vec<f64>>();

    group.bench_function("DenseMatrix", |b| {
        let x = DenseMatrix::<f64>::rand(10000, 500);
        let y = <DenseMatrix<f64> as BaseMatrix<f64>>::RowVector::from_array(&classes);

        b.iter(|| {
            GaussianNB::fit(&x, &y, Default::default()).unwrap();
        })
    });

    group.bench_function("ndarray", |b| {
        let x = Array2::<f64>::rand(10000, 500);
        let y = <Array2<f64> as BaseMatrix<f64>>::RowVector::from_array(&classes);

        b.iter(|| {
            GaussianNB::fit(&x, &y, Default::default()).unwrap();
        })
    });

    group.bench_function("ndalgebra", |b| {
        let x = DMatrix::<f64>::rand(10000, 500);
        let y = <DMatrix<f64> as BaseMatrix<f64>>::RowVector::from_array(&classes);

        b.iter(|| {
            GaussianNB::fit(&x, &y, Default::default()).unwrap();
        })
    });
}
criterion_group!(
    benches,
    gaussian_naive_bayes_fit_benchmark,
    gaussian_naive_matrix_datastructure
);
criterion_main!(benches);
