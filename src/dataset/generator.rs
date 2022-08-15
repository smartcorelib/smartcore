//! # Dataset Generators
//!
use rand::distributions::Uniform;
use rand::prelude::*;
use rand_distr::Normal;

use crate::dataset::Dataset;

/// Generate `num_centers` clusters of normally distributed points
pub fn make_blobs(
    num_samples: usize,
    num_features: usize,
    num_centers: usize,
) -> Dataset<f32, f32> {
    let center_box = Uniform::from(-10.0..10.0);
    let cluster_std = 1.0;
    let mut centers: Vec<Vec<Normal<f32>>> = Vec::with_capacity(num_centers);

    let mut rng = rand::thread_rng();
    for _ in 0..num_centers {
        centers.push(
            (0..num_features)
                .map(|_| Normal::new(center_box.sample(&mut rng), cluster_std).unwrap())
                .collect(),
        );
    }

    let mut y: Vec<f32> = Vec::with_capacity(num_samples);
    let mut x: Vec<f32> = Vec::with_capacity(num_samples);

    for i in 0..num_samples {
        let label = i % num_centers;
        y.push(label as f32);
        for j in 0..num_features {
            x.push(centers[label][j].sample(&mut rng));
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
pub fn make_circles(num_samples: usize, factor: f32, noise: f32) -> Dataset<f32, f32> {
    if !(0.0..1.0).contains(&factor) {
        panic!("'factor' has to be between 0 and 1.");
    }

    let num_samples_out = num_samples / 2;
    let num_samples_in = num_samples - num_samples_out;

    let linspace_out = linspace(0.0, 2.0 * std::f32::consts::PI, num_samples_out);
    let linspace_in = linspace(0.0, 2.0 * std::f32::consts::PI, num_samples_in);

    let noise = Normal::new(0.0, noise).unwrap();
    let mut rng = rand::thread_rng();

    let mut x: Vec<f32> = Vec::with_capacity(num_samples * 2);
    let mut y: Vec<f32> = Vec::with_capacity(num_samples);

    for v in linspace_out {
        x.push(v.cos() + noise.sample(&mut rng));
        x.push(v.sin() + noise.sample(&mut rng));
        y.push(0.0);
    }

    for v in linspace_in {
        x.push(v.cos() * factor + noise.sample(&mut rng));
        x.push(v.sin() * factor + noise.sample(&mut rng));
        y.push(1.0);
    }

    Dataset {
        data: x,
        target: y,
        num_samples,
        num_features: 2,
        feature_names: (0..2).map(|n| n.to_string()).collect(),
        target_names: vec!["label".to_string()],
        description: "Large circle containing a smaller circle in 2d".to_string(),
    }
}

/// Make two interleaving half circles in 2d
pub fn make_moons(num_samples: usize, noise: f32) -> Dataset<f32, f32> {
    let num_samples_out = num_samples / 2;
    let num_samples_in = num_samples - num_samples_out;

    let linspace_out = linspace(0.0, std::f32::consts::PI, num_samples_out);
    let linspace_in = linspace(0.0, std::f32::consts::PI, num_samples_in);

    let noise = Normal::new(0.0, noise).unwrap();
    let mut rng = rand::thread_rng();

    let mut x: Vec<f32> = Vec::with_capacity(num_samples * 2);
    let mut y: Vec<f32> = Vec::with_capacity(num_samples);

    for v in linspace_out {
        x.push(v.cos() + noise.sample(&mut rng));
        x.push(v.sin() + noise.sample(&mut rng));
        y.push(0.0);
    }

    for v in linspace_in {
        x.push(1.0 - v.cos() + noise.sample(&mut rng));
        x.push(1.0 - v.sin() + noise.sample(&mut rng) - 0.5);
        y.push(1.0);
    }

    Dataset {
        data: x,
        target: y,
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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
}
