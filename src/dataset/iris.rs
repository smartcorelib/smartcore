//! # The Iris flower dataset
//!
//! | Number of Instances | Number of Attributes | Missing Values? | Associated Tasks: |
//! |-|-|-|-|
//! | 150 | 4 | No | Classification |
//!
//! [Fisher's Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris) is a multivariate dataset that was published in 1936 by Ronald Fisher.
//! This multivariate dataset is frequently used to demonstrate various machine learning algorithms. The dataset has following attributes:
//!
//! | Predictor | Data Type | Target? |
//! |-|-|-|
//! | Sepal length | Numerical | No |
//! | Sepal width | Numerical | No |
//! | Petal length | Numerical | No |
//! | Petal width | Numerical | No |
//! | Class | Nominal | Yes |
//!
use crate::dataset::deserialize_data;
use crate::dataset::Dataset;

/// Get dataset
pub fn load_dataset() -> Dataset<f32, u32> {
    let (x, y, num_samples, num_features): (Vec<f32>, Vec<u32>, usize, usize) =
        match deserialize_data(std::include_bytes!("iris.xy")) {
            Err(why) => panic!("Can't deserialize iris.xy. {why}"),
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
        feature_names: [
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect(),
        target_names: ["setosa", "versicolor", "virginica"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        description: "Iris dataset: https://archive.ics.uci.edu/ml/datasets/iris".to_string(),
    }
}

#[cfg(test)]
mod tests {

    // #[cfg(not(target_arch = "wasm32"))]
    // use super::super::*;
    use super::*;

    // TODO: fix serialization
    // #[cfg(not(target_arch = "wasm32"))]
    // #[test]
    // #[ignore]
    // fn refresh_iris_dataset() {
    //     // run this test to generate iris.xy file.
    //     let dataset = load_dataset();
    //     assert!(serialize_data(&dataset, "iris.xy").is_ok());
    // }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn iris_dataset() {
        let dataset = load_dataset();
        assert_eq!(dataset.data.len(), 50 * 3 * 4);
        assert_eq!(dataset.target.len(), 50 * 3);
        assert_eq!(dataset.num_features, 4);
        assert_eq!(dataset.num_samples, 50 * 3);
    }
}
