//! # Optical Recognition of Handwritten Digits Dataset
//!
//! | Number of Instances | Number of Attributes | Missing Values? | Associated Tasks: |
//! |-|-|-|-|
//! | 1797 | 64 | No | Classification, Clusteing |
//!
//! [Digits dataset](https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits) contains normalized bitmaps of handwritten digits (0-9) from a preprinted form.
//! This multivariate dataset is frequently used to demonstrate various machine learning algorithms.
//!
//! All input attributes are integers in the range 0..16.
//!
use crate::dataset::deserialize_data;
use crate::dataset::Dataset;

/// Get dataset
pub fn load_dataset() -> Dataset<f32, f32> {
    let (x, y, num_samples, num_features) = match deserialize_data(std::include_bytes!("digits.xy"))
    {
        Err(why) => panic!("Can't deserialize digits.xy. {}", why),
        Ok((x, y, num_samples, num_features)) => (x, y, num_samples, num_features),
    };

    Dataset {
        data: x,
        target: y,
        num_samples,
        num_features,
        feature_names: vec![
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect(),
        target_names: vec!["setosa", "versicolor", "virginica"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        description: "Digits dataset: https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits".to_string(),
    }
}

#[cfg(test)]
mod tests {

    #[cfg(not(target_arch = "wasm32"))]
    use super::super::*;
    use super::*;

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    #[ignore]
    fn refresh_digits_dataset() {
        // run this test to generate digits.xy file.
        let dataset = load_dataset();
        assert!(serialize_data(&dataset, "digits.xy").is_ok());
    }
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn digits_dataset() {
        let dataset = load_dataset();
        assert_eq!(dataset.data.len(), 1797 * 64);
        assert_eq!(dataset.target.len(), 1797);
        assert_eq!(dataset.num_features, 64);
        assert_eq!(dataset.num_samples, 1797);
    }
}
