//! # Breast Cancer Wisconsin (Diagnostic) Data Set
//!
//! Diagnostic Wisconsin Breast Cancer database
//!
//! | Number of Instances | Number of Attributes | Missing Values? | Associated Tasks: |
//! |-|-|-|-|
//! | 569 | 30 | No | Classification |
//!
//! [Breast Cancer](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29) was collected by Dr. William H. Wolberg, W. Nick Street and Olvi L. Mangasarian.
//! Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass and describe characteristics of the cell nuclei present in the image.
//! The dataset has following attributes:
//!
//! | Predictor | Data Type | Target? |
//! |-|-|-|
//! | Radius (mean of distances from center to points on the perimeter) | Numerical | No |
//! | Texture (standard deviation of gray-scale values) | Numerical | No |
//! | Perimeter | Numerical | No |
//! | Area | Numerical | No |
//! | Smoothness (local variation in radius lengths) | Numerical | No |
//! | Compactness (perimeter^2 / area - 1.0) | Numerical | No |
//! | Concavity (severity of concave portions of the contour) | Numerical | No |
//! | Concave points (number of concave portions of the contour) | Numerical | No |
//! | Symmetry | Numerical | No |
//! | Fractal dimension ("coastline approximation" - 1) | Numerical | No |
//! | Has cancer | Nominal | Yes |
//!
//! The mean, standard error, and "worst" or largest (mean of the three worst/largest values) of these features were computed for each image, resulting in 30 features.
//! For instance, field 0 is Mean Radius, field 10 is Radius SE, field 20 is Worst Radius.
use crate::dataset::deserialize_data;
use crate::dataset::Dataset;

/// Get dataset
pub fn load_dataset() -> Dataset<f32, u32> {
    let (x, y, num_samples, num_features) =
        match deserialize_data(std::include_bytes!("breast_cancer.xy")) {
            Err(why) => panic!("Can't deserialize breast_cancer.xy. {why}"),
            Ok((x, y, num_samples, num_features)) => (
                x,
                y.into_iter().map(|x| x as u32).collect(),
                num_samples,
                num_features,
            ),
        };

    Dataset {
        data: x,
        target: y,
        num_samples,
        num_features,
        feature_names: vec![
            "mean radius", "mean texture", "mean perimeter", "mean area",
            "mean smoothness", "mean compactness", "mean concavity",
            "mean concave points", "mean symmetry", "mean fractal dimension",
            "radius error", "texture error", "perimeter error", "area error",
            "smoothness error", "compactness error", "concavity error",
            "concave points error", "symmetry error",
            "fractal dimension error", "worst radius", "worst texture",
            "worst perimeter", "worst area", "worst smoothness",
            "worst compactness", "worst concavity", "worst concave points",
            "worst symmetry", "worst fractal dimension",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect(),
        target_names: vec!["malignant or benign [0, 1]".to_string()],
        description: "Breast Cancer Wisconsin (Diagnostic) Data Set: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29"
            .to_string(),
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    // TODO: implement serialization
    // #[test]
    // #[ignore]
    // #[cfg(not(target_arch = "wasm32"))]
    // fn refresh_cancer_dataset() {
    //     // run this test to generate breast_cancer.xy file.
    //     let dataset = load_dataset();
    //     assert!(serialize_data(&dataset, "breast_cancer.xy").is_ok());
    // }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn cancer_dataset() {
        let dataset = load_dataset();
        assert_eq!(
            dataset.data.len(),
            dataset.num_features * dataset.num_samples
        );
        assert_eq!(dataset.target.len(), dataset.num_samples);
        assert_eq!(dataset.num_features, 30);
        assert_eq!(dataset.num_samples, 569);
    }
}
