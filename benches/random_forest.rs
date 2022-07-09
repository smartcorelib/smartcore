use criterion::{criterion_group, criterion_main, Criterion};
use smartcore::{
    dataset::digits,
    ensemble::random_forest_classifier::{
        RandomForestClassifier, RandomForestClassifierParameters,
    },
    linalg::naive::dense_matrix::DenseMatrix,
    tree::decision_tree_classifier::SplitCriterion,
};

fn random_forest_classifier_for_digits(c: &mut Criterion) {
    let dataset_digits = digits::load_dataset();
    let nrows = dataset_digits.target.len();
    let ncols = dataset_digits.num_features;
    let values = dataset_digits.data;
    let x = DenseMatrix::from_vec(nrows, ncols, &values);
    let y = dataset_digits.target;

    c.bench_function("Benchmarking random forest fitting on digits", |b| {
        b.iter(|| {
            RandomForestClassifier::fit(
                &x,
                &y,
                RandomForestClassifierParameters {
                    criterion: SplitCriterion::Gini,
                    max_depth: None,
                    min_samples_leaf: 1,
                    min_samples_split: 2,
                    n_trees: 100,
                    m: Option::None,
                    keep_samples: false,
                    base_seed: 87,
                    num_threads: 0,
                },
            )
        })
    });
}

criterion_group!(benches, random_forest_classifier_for_digits);
criterion_main!(benches);
