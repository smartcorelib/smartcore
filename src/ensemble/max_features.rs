//! # Max Features
//! At each node, a forest of decision trees tests only a random subset of the features.
//! [`MaxFeatures`] sets the size of this subset. The forest calculates the size from _p_,
//! the number of features in the training data.
//!
//! | Variant          | Number of features            | Condition         |
//! |------------------|-------------------------------|-------------------|
//! | `Sqrt` (default) | \\(\max(1, \lfloor\sqrt{p}\rfloor)\\) |           |
//! | `Fraction(f)`    | \\(\max(1, \lfloor f p \rfloor)\\)    | \\(0 < f \le 1\\) |
//! | `Count(n)`       | \\(n\\)                       | \\(1 \le n \le p\\) |
//! | `All`            | \\(p\\)                       |                   |
//!
//! A small subset makes the trees less correlated, so the forest has a lower variance.
//! A large subset makes each tree stronger. \\(\sqrt{p}\\) is the usual value for classification.
//! With `All`, a random forest is the same as bagging.
//!
//! If the value is not valid for the data (for example, `Count(n)` with \\(n > p\\)), `fit` returns an error.
//!
//! The four forest estimators use this type: [random forest classifier](../random_forest_classifier/index.html),
//! [random forest regressor](../random_forest_regressor/index.html),
//! [extra trees classifier](../extra_trees_classifier/index.html) and
//! [extra trees regressor](../extra_trees_regressor/index.html).
//!
//! ## References:
//!
//! * ["Random Forests", Breiman L., Machine Learning 45(1), 5-32, 2001](https://doi.org/10.1023/A:1010933404324)
//! * ["Extremely randomized trees", Geurts P., Ernst D., Wehenkel L., Machine Learning 63(1), 3-42, 2006](https://doi.org/10.1007/s10994-006-6226-1)
//! * ["An Introduction to Statistical Learning", James G., Witten D., Hastie T., Tibshirani R., 8.2 Bagging, Random Forests, Boosting](http://faculty.marshall.usc.edu/gareth-james/ISL/)
//!
//! Example:
//!
//! ```
//! use smartcore::ensemble::max_features::MaxFeatures;
//! use smartcore::ensemble::random_forest_classifier::RandomForestClassifierParameters;
//!
//! // Use the default: the square root of the number of features.
//! let params = RandomForestClassifierParameters::default();
//! assert_eq!(params.m, MaxFeatures::Sqrt);
//!
//! // Use half of the features at each node.
//! let params = RandomForestClassifierParameters::default()
//!     .with_m(MaxFeatures::Fraction(0.5));
//!
//! // Use exactly 3 features at each node. A plain `usize` gives `Count`.
//! let params = RandomForestClassifierParameters::default().with_m(3);
//! assert_eq!(params.m, MaxFeatures::Count(3));
//! ```
use crate::error::{Failed, FailedError};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
/// Specify the number of features to use in each node while fitting the tree.
pub enum MaxFeatures {
    #[default]
    /// Consider sqrt(n_features) in each node
    Sqrt, // max(1, floor(sqrt(p))) — default
    /// Consider the given fraction of the features in each node
    Fraction(f64), // max(1, floor(f * p)), 0 < f <= 1
    /// Consider the given number of features in each node
    Count(usize), // n, 1 <= n <= p
    /// Consider all the features in each node
    All, // p
}

impl MaxFeatures {
    pub(crate) fn resolve(&self, n_features: usize) -> Result<usize, Failed> {
        match *self {
            MaxFeatures::Sqrt => Ok(1.max(n_features.isqrt())),
            MaxFeatures::Fraction(f) => {
                if 0.0 < f && f <= 1.0 {
                    Ok(1.max((f * n_features as f64) as usize))
                } else {
                    Err(Failed::because(
                        FailedError::ParametersError,
                        &format!(
                            "fraction must be between 0 (exclusive) and 1 (inclusive), got {f}."
                        ),
                    ))
                }
            }
            MaxFeatures::Count(count) => {
                if (1..=n_features).contains(&count) {
                    Ok(count)
                } else {
                    Err(Failed::because(
                        FailedError::ParametersError,
                        &format!(
                            "number of features to use must be between 1 and {n_features}, got {count}."
                        ),
                    ))
                }
            }
            MaxFeatures::All => Ok(n_features),
        }
    }
}

impl From<usize> for MaxFeatures {
    fn from(n: usize) -> Self {
        MaxFeatures::Count(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn default_is_sqrt() {
        assert_eq!(MaxFeatures::default(), MaxFeatures::Sqrt);
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn from_usize_gives_count() {
        assert_eq!(MaxFeatures::from(3), MaxFeatures::Count(3));
        let m: MaxFeatures = 7.into();
        assert_eq!(m, MaxFeatures::Count(7));
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn sqrt_rounds_down() {
        // (n_features, expected)
        let cases = [
            (1, 1),
            (2, 1),
            (3, 1),
            (4, 2),
            (8, 2),
            (9, 3),
            (10, 3),
            (100, 10),
        ];
        for (p, expected) in cases {
            assert_eq!(MaxFeatures::Sqrt.resolve(p).unwrap(), expected, "p = {p}");
        }
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn fraction_rounds_down() {
        assert_eq!(MaxFeatures::Fraction(0.5).resolve(7).unwrap(), 3);
        assert_eq!(MaxFeatures::Fraction(0.25).resolve(8).unwrap(), 2);
        assert_eq!(MaxFeatures::Fraction(1.0).resolve(8).unwrap(), 8);
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn small_fraction_gives_at_least_one() {
        assert_eq!(MaxFeatures::Fraction(0.01).resolve(10).unwrap(), 1);
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn fraction_out_of_range_is_error() {
        for f in [0.0, -0.5, 1.5, f64::NAN, f64::INFINITY] {
            assert!(MaxFeatures::Fraction(f).resolve(10).is_err(), "f = {f}");
        }
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn fraction_error_message() {
        let expected = Failed::because(
            FailedError::ParametersError,
            "fraction must be between 0 (exclusive) and 1 (inclusive), got 1.5.",
        );
        assert_eq!(MaxFeatures::Fraction(1.5).resolve(10).err(), Some(expected));
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn count_in_range_is_unchanged() {
        assert_eq!(MaxFeatures::Count(1).resolve(5).unwrap(), 1);
        assert_eq!(MaxFeatures::Count(3).resolve(5).unwrap(), 3);
        assert_eq!(MaxFeatures::Count(5).resolve(5).unwrap(), 5);
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn count_out_of_range_is_error() {
        assert!(MaxFeatures::Count(0).resolve(5).is_err());
        assert!(MaxFeatures::Count(6).resolve(5).is_err());

        let expected = Failed::because(
            FailedError::ParametersError,
            "number of features to use must be between 1 and 5, got 6.",
        );
        assert_eq!(MaxFeatures::Count(6).resolve(5).err(), Some(expected));
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn all_gives_every_feature() {
        assert_eq!(MaxFeatures::All.resolve(1).unwrap(), 1);
        assert_eq!(MaxFeatures::All.resolve(12).unwrap(), 12);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_round_trip() {
        for m in [
            MaxFeatures::Sqrt,
            MaxFeatures::Fraction(0.5),
            MaxFeatures::Count(3),
            MaxFeatures::All,
        ] {
            let json = serde_json::to_string(&m).unwrap();
            let back: MaxFeatures = serde_json::from_str(&json).unwrap();
            assert_eq!(back, m, "json = {json}");
        }
    }
}
