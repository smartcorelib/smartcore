//! Stage 5: dataset loader & generator edge-case tests.
//!
//! Gated on `#[cfg(feature = "datasets")]`.
//! Tracking issue: #396 / #391.

#[cfg(all(test, feature = "datasets"))]
mod dataset_tests {
    // ── Loaders ──────────────────────────────────────────────────────────────

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn boston_loader_shape_and_features() {
        use crate::dataset::boston::load_dataset;
        let ds = load_dataset();
        assert_eq!(ds.num_features, 13, "Boston: expected 13 features");
        assert_eq!(ds.num_samples, 506, "Boston: expected 506 samples");
        assert_eq!(ds.data.len(), 506 * 13);
        assert_eq!(ds.target.len(), 506);
        assert!(!ds.feature_names.is_empty(), "Boston: feature_names empty");
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn breast_cancer_loader_shape_and_binary_target() {
        use crate::dataset::breast_cancer::load_dataset;
        let ds = load_dataset();
        assert_eq!(ds.num_features, 30, "BreastCancer: expected 30 features");
        assert_eq!(ds.num_samples, 569, "BreastCancer: expected 569 samples");
        // Target is binary: only 0s and 1s
        let unique_targets: std::collections::HashSet<u32> = ds.target.iter().cloned().collect();
        assert!(unique_targets.len() <= 2, "BreastCancer: more than 2 target classes");
        assert!(unique_targets.contains(&0) || unique_targets.contains(&1));
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn diabetes_loader_shape() {
        use crate::dataset::diabetes::load_dataset;
        let ds = load_dataset();
        assert_eq!(ds.num_features, 10, "Diabetes: expected 10 features");
        assert_eq!(ds.num_samples, 442, "Diabetes: expected 442 samples");
        assert_eq!(ds.data.len(), 442 * 10);
        assert_eq!(ds.target.len(), 442);
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn digits_loader_shape_and_target_cardinality() {
        use crate::dataset::digits::load_dataset;
        let ds = load_dataset();
        assert_eq!(ds.num_features, 64, "Digits: expected 64 features");
        assert_eq!(ds.num_samples, 1797, "Digits: expected 1797 samples");
        let unique_targets: std::collections::HashSet<u32> = ds.target.iter().cloned().collect();
        assert_eq!(unique_targets.len(), 10, "Digits: expected 10 classes (0-9)");
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn iris_loader_shape_and_target_names() {
        use crate::dataset::iris::load_dataset;
        let ds = load_dataset();
        assert_eq!(ds.num_features, 4, "Iris: expected 4 features");
        assert_eq!(ds.num_samples, 150, "Iris: expected 150 samples");
        assert_eq!(ds.feature_names.len(), 4);
        assert_eq!(ds.target_names.len(), 3, "Iris: expected 3 target names");
        let unique_targets: std::collections::HashSet<u32> = ds.target.iter().cloned().collect();
        assert_eq!(unique_targets.len(), 3, "Iris: expected 3 target classes");
    }

    // ── Generators ───────────────────────────────────────────────────────────

    use crate::dataset::generator::{make_blobs, make_circles, make_moons};

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn make_blobs_standard() {
        let ds = make_blobs(100, 4, 3);
        assert_eq!(ds.num_samples, 100);
        assert_eq!(ds.num_features, 4);
        assert_eq!(ds.data.len(), 400);
        assert_eq!(ds.target.len(), 100);
        // Labels should be in {0.0, 1.0, 2.0}
        let unique: std::collections::HashSet<u32> =
            ds.target.iter().map(|&v| v as u32).collect();
        assert_eq!(unique.len(), 3);
    }

    /// n_samples=1 edge-case: must not panic.
    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn make_blobs_single_sample() {
        let ds = make_blobs(1, 2, 1);
        assert_eq!(ds.num_samples, 1);
        assert_eq!(ds.data.len(), 2);
        assert_eq!(ds.target.len(), 1);
    }

    /// noise=0 circles: all data should be finite.
    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn make_circles_zero_noise() {
        let ds = make_circles(20, 0.5, 0.0);
        assert_eq!(ds.num_samples, 20);
        assert_eq!(ds.num_features, 2);
        assert!(ds.data.iter().all(|v| v.is_finite()), "non-finite value in circles data");
    }

    /// noise=0 moons: all data should be finite.
    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn make_moons_zero_noise() {
        let ds = make_moons(20, 0.0);
        assert_eq!(ds.num_samples, 20);
        assert_eq!(ds.num_features, 2);
        assert!(ds.data.iter().all(|v| v.is_finite()), "non-finite value in moons data");
    }

    /// make_circles with n_samples=2 (minimum viable: 1 outer + 1 inner).
    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn make_circles_minimal_samples() {
        let ds = make_circles(2, 0.5, 0.01);
        assert_eq!(ds.num_samples, 2);
        assert_eq!(ds.target.len(), 2);
    }

    /// make_moons with n_samples=2.
    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn make_moons_minimal_samples() {
        let ds = make_moons(2, 0.0);
        assert_eq!(ds.num_samples, 2);
        assert_eq!(ds.target.len(), 2);
    }

    /// Description field is always non-empty.
    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn dataset_description_non_empty() {
        assert!(!make_blobs(10, 2, 2).description.is_empty());
        assert!(!make_circles(10, 0.5, 0.05).description.is_empty());
        assert!(!make_moons(10, 0.05).description.is_empty());
    }
}
