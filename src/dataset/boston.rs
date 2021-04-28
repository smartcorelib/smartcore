//! # The Boston Housing Dataset
//!
//! | Number of Instances | Number of Attributes | Missing Values? | Associated Tasks: |
//! |-|-|-|-|
//! | 506 | 13 | No | Regression |
//!
//! [The Boston house-price data](http://lib.stat.cmu.edu/datasets/boston) is derived from information collected by the U.S. Census Service concerning housing in the area of Boston, MA.
//! The dataset has following attributes:
//!
//! | Predictor | Data Type | Target? |
//! |-|-|-|
//! | CRIM, per capita crime rate by town | Numerical | No |
//! | ZN, proportion of residential land zoned for lots over 25,000 sq.ft. | Numerical | No |
//! | INDUS, proportion of non-retail business acres per town. | Numerical | No |
//! | CHAS, Charles River dummy variable (1 if tract bounds river; 0 otherwise) | Nominal | No |
//! | NOX, nitric oxides concentration (parts per 10 million) | Numerical | No |
//! | RM, average number of rooms per dwelling | Numerical | No |
//! | AGE, proportion of owner-occupied units built prior to 1940 | Numerical | No |
//! | DIS, weighted distances to five Boston employment centres | Numerical | No |
//! | RAD, index of accessibility to radial highways | Ordinal | No |
//! | TAX, full-value property-tax rate per $10,000 | Numerical | No |
//! | PTRATIO, pupil-teacher ratio by town | Numerical | No |
//! | B, 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town | Numerical | No |
//! | LSTAT, % lower status of the population | Numerical | No |
//! | MEDV, Median value of owner-occupied homes in $1000's | Numerical | Yes |
//!
use crate::dataset::deserialize_data;
use crate::dataset::Dataset;

/// Get dataset
pub fn load_dataset() -> Dataset<f32, f32> {
    let (x, y, num_samples, num_features) = match deserialize_data(std::include_bytes!("boston.xy"))
    {
        Err(why) => panic!("Can't deserialize boston.xy. {}", why),
        Ok((x, y, num_samples, num_features)) => (x, y, num_samples, num_features),
    };

    Dataset {
        data: x,
        target: y,
        num_samples,
        num_features,
        feature_names: vec![
            "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B",
            "LSTAT",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect(),
        target_names: vec!["price".to_string()],
        description: "The Boston house-price data: http://lib.stat.cmu.edu/datasets/boston"
            .to_string(),
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[cfg(test)]
mod tests {

    use super::super::*;
    use super::*;

    #[test]
    #[ignore]
    fn refresh_boston_dataset() {
        // run this test to generate boston.xy file.
        let dataset = load_dataset();
        assert!(serialize_data(&dataset, "boston.xy").is_ok());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn boston_dataset() {
        let dataset = load_dataset();
        assert_eq!(
            dataset.data.len(),
            dataset.num_features * dataset.num_samples
        );
        assert_eq!(dataset.target.len(), dataset.num_samples);
        assert_eq!(dataset.num_features, 13);
        assert_eq!(dataset.num_samples, 506);
    }
}
