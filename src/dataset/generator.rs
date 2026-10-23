//! # Dataset Generators
//!
//! The generators get their RNG from `get_rng_impl(None)`. With the `std_rand`
//! feature the RNG takes its seed from OS entropy. Without `std_rand` the RNG
//! uses a fixed seed (or `getrandom` with the `js` feature), thus the
//! `datasets` feature does not need `rand/std`.
use rand::distr::Distribution;
use rand::distr::Uniform;

use crate::dataset::Dataset;
use crate::rand_custom::get_rng_impl;

/// Sample from N(mean, std) via Box-Muller transform using only rand 0.10
#[inline]
fn sample_normal(mean: f32, std: f32, rng: &mut impl rand::Rng) -> f32 {
    let unit = Uniform::new(f32::EPSILON, 1.0f32).unwrap();
    let u1 = unit.sample(rng);
    let u2 = unit.sample(rng);
    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
    mean + std * z
}

/// Generate `num_centers` clusters of normally distributed points
///
/// The RNG comes from `get_rng_impl(None)`. With the `std_rand` feature the output
/// is random for each call. Without `std_rand` (and without `js`) the RNG uses a
/// fixed seed, thus each call gives the same output. Use [`make_blobs_with_seed`] to
/// select the seed.
#[must_use]
pub fn make_blobs(
    num_samples: usize,
    num_features: usize,
    num_centers: usize,
) -> Dataset<f32, f32> {
    make_blobs_with_seed(num_samples, num_features, num_centers, None)
}

/// Same as [`make_blobs`], but uses `seed` for the RNG. With `Some(seed)` the output is
/// reproducible with all features. With `None` the behaviour is the same as [`make_blobs`].
#[must_use]
pub fn make_blobs_with_seed(
    num_samples: usize,
    num_features: usize,
    num_centers: usize,
    seed: Option<u64>,
) -> Dataset<f32, f32> {
    let center_box = Uniform::new(-10.0f32, 10.0f32).expect("Invalid uniform range");
    let cluster_std = 1.0f32;
    let mut rng = get_rng_impl(seed);

    // Pre-compute cluster centers (one mean per feature per cluster)
    let centers: Vec<Vec<f32>> = (0..num_centers)
        .map(|_| {
            (0..num_features)
                .map(|_| center_box.sample(&mut rng))
                .collect()
        })
        .collect();

    let mut y: Vec<f32> = Vec::with_capacity(num_samples);
    let mut x: Vec<f32> = Vec::with_capacity(num_samples * num_features);

    for i in 0..num_samples {
        let label = i % num_centers;
        y.push(label as f32);
        for j in 0..num_features {
            x.push(sample_normal(centers[label][j], cluster_std, &mut rng));
        }
    }

    Dataset {
        data: x,
        target: y,
        num_samples,
        num_features,
        feature_names: (0..num_features).map(|n| n.to_string()).collect(),
        target_names: vec!["label".to_string()],
        description: "Isotropic Gaussian blobs".to_string(),
    }
}

/// Make a large circle containing a smaller circle in 2d.
///
/// The RNG comes from `get_rng_impl(None)`. With the `std_rand` feature the output
/// is random for each call. Without `std_rand` (and without `js`) the RNG uses a
/// fixed seed, thus each call gives the same output. Use [`make_circles_with_seed`] to
/// select the seed.
#[must_use]
pub fn make_circles(num_samples: usize, factor: f32, noise: f32) -> Dataset<f32, u32> {
    make_circles_with_seed(num_samples, factor, noise, None)
}

/// Same as [`make_circles`], but uses `seed` for the RNG. With `Some(seed)` the output is
/// reproducible with all features. With `None` the behaviour is the same as [`make_circles`].
#[must_use]
pub fn make_circles_with_seed(
    num_samples: usize,
    factor: f32,
    noise: f32,
    seed: Option<u64>,
) -> Dataset<f32, u32> {
    if !(0.0..1.0).contains(&factor) {
        panic!("'factor' has to be between 0 and 1.");
    }

    let num_samples_out = num_samples / 2;
    let num_samples_in = num_samples - num_samples_out;

    let linspace_out = linspace(0.0, 2.0 * std::f32::consts::PI, num_samples_out);
    let linspace_in = linspace(0.0, 2.0 * std::f32::consts::PI, num_samples_in);

    let mut rng = get_rng_impl(seed);

    let mut x: Vec<f32> = Vec::with_capacity(num_samples * 2);
    let mut y: Vec<f32> = Vec::with_capacity(num_samples);

    for v in linspace_out {
        x.push(v.cos() + sample_normal(0.0, noise, &mut rng));
        x.push(v.sin() + sample_normal(0.0, noise, &mut rng));
        y.push(0.0);
    }

    for v in linspace_in {
        x.push(v.cos() * factor + sample_normal(0.0, noise, &mut rng));
        x.push(v.sin() * factor + sample_normal(0.0, noise, &mut rng));
        y.push(1.0);
    }

    Dataset {
        data: x,
        target: y.into_iter().map(|x| x as u32).collect(),
        num_samples,
        num_features: 2,
        feature_names: (0..2).map(|n| n.to_string()).collect(),
        target_names: vec!["label".to_string()],
        description: "Large circle containing a smaller circle in 2d".to_string(),
    }
}

/// Make two interleaving half circles in 2d
///
/// The RNG comes from `get_rng_impl(None)`. With the `std_rand` feature the output
/// is random for each call. Without `std_rand` (and without `js`) the RNG uses a
/// fixed seed, thus each call gives the same output. Use [`make_moons_with_seed`] to
/// select the seed.
#[must_use]
pub fn make_moons(num_samples: usize, noise: f32) -> Dataset<f32, u32> {
    make_moons_with_seed(num_samples, noise, None)
}

/// Same as [`make_moons`], but uses `seed` for the RNG. With `Some(seed)` the output is
/// reproducible with all features. With `None` the behaviour is the same as [`make_moons`].
#[must_use]
pub fn make_moons_with_seed(
    num_samples: usize,
    noise: f32,
    seed: Option<u64>,
) -> Dataset<f32, u32> {
    let num_samples_out = num_samples / 2;
    let num_samples_in = num_samples - num_samples_out;

    let linspace_out = linspace(0.0, std::f32::consts::PI, num_samples_out);
    let linspace_in = linspace(0.0, std::f32::consts::PI, num_samples_in);

    let mut rng = get_rng_impl(seed);

    let mut x: Vec<f32> = Vec::with_capacity(num_samples * 2);
    let mut y: Vec<f32> = Vec::with_capacity(num_samples);

    for v in linspace_out {
        x.push(v.cos() + sample_normal(0.0, noise, &mut rng));
        x.push(v.sin() + sample_normal(0.0, noise, &mut rng));
        y.push(0.0);
    }

    for v in linspace_in {
        x.push(1.0 - v.cos() + sample_normal(0.0, noise, &mut rng));
        x.push(1.0 - v.sin() + sample_normal(0.0, noise, &mut rng) - 0.5);
        y.push(1.0);
    }

    Dataset {
        data: x,
        target: y.into_iter().map(|x| x as u32).collect(),
        num_samples,
        num_features: 2,
        feature_names: (0..2).map(|n| n.to_string()).collect(),
        target_names: vec!["label".to_string()],
        description: "Two interleaving half circles in 2d".to_string(),
    }
}

fn linspace(start: f32, stop: f32, num: usize) -> Vec<f32> {
    let div = num as f32;
    let delta = stop - start;
    let step = delta / div;
    (0..num).map(|v| v as f32 * step).collect()
}

#[cfg(test)]
mod tests {

    use super::*;

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn test_make_blobs() {
        let dataset = make_blobs(10, 2, 3);
        assert_eq!(
            dataset.data.len(),
            dataset.num_features * dataset.num_samples
        );
        assert_eq!(dataset.target.len(), dataset.num_samples);
        assert_eq!(dataset.num_features, 2);
        assert_eq!(dataset.num_samples, 10);
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn test_make_circles() {
        let dataset = make_circles(10, 0.5, 0.05);
        assert_eq!(
            dataset.data.len(),
            dataset.num_features * dataset.num_samples
        );
        assert_eq!(dataset.target.len(), dataset.num_samples);
        assert_eq!(dataset.num_features, 2);
        assert_eq!(dataset.num_samples, 10);
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn test_make_moons() {
        let dataset = make_moons(10, 0.05);
        assert_eq!(
            dataset.data.len(),
            dataset.num_features * dataset.num_samples
        );
        assert_eq!(dataset.target.len(), dataset.num_samples);
        assert_eq!(dataset.num_features, 2);
        assert_eq!(dataset.num_samples, 10);
    }

    #[cfg(not(any(feature = "std_rand", feature = "js")))]
    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn test_generators_without_std_rand_are_reproducible() {
        // without std_rand the generators use a fixed-seed SmallRng
        assert_eq!(make_blobs(10, 2, 3).data, make_blobs(10, 2, 3).data);
        assert_eq!(
            make_circles(10, 0.5, 0.05).data,
            make_circles(10, 0.5, 0.05).data
        );
        assert_eq!(make_moons(10, 0.05).data, make_moons(10, 0.05).data);
    }

    #[cfg(feature = "std_rand")]
    #[test]
    fn test_generators_with_std_rand_are_random() {
        // with std_rand the generators take the seed from OS entropy
        assert_ne!(make_blobs(100, 2, 3).data, make_blobs(100, 2, 3).data);
        assert_ne!(
            make_circles(100, 0.5, 0.05).data,
            make_circles(100, 0.5, 0.05).data
        );
        assert_ne!(make_moons(100, 0.05).data, make_moons(100, 0.05).data);
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn test_generators_with_seed_are_reproducible() {
        assert_eq!(
            make_blobs_with_seed(10, 2, 3, Some(42)).data,
            make_blobs_with_seed(10, 2, 3, Some(42)).data
        );
        assert_ne!(
            make_blobs_with_seed(10, 2, 3, Some(1)).data,
            make_blobs_with_seed(10, 2, 3, Some(2)).data
        );
        assert_eq!(
            make_circles_with_seed(10, 0.5, 0.05, Some(42)).data,
            make_circles_with_seed(10, 0.5, 0.05, Some(42)).data
        );
        assert_eq!(
            make_moons_with_seed(10, 0.05, Some(42)).data,
            make_moons_with_seed(10, 0.05, Some(42)).data
        );
    }
}
