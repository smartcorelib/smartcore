//! # Dimension Reduction Methods
//! Dimension reduction is a popular approach for deriving a low-dimensional set of features from a large set of variables.
//!
//! High Dimensional Data (a lot of input features) often degrade performance of machine learning algorithms due to [curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality).
//! The more dimensions you have in a data set, the more difficult it becomes to predict certain quantities. While it seems that the more explanatory variables the better,
//! when it comes to adding variables, the opposite is true. Each added variable results in an exponential decrease in predictive power.
//! Therefore, it is often desirable to reduce the number of input features.
//!
//! Dimension reduction is also used for the purposes of data visualization.
//!
//! ## References
//! * ["An Introduction to Statistical Learning", James G., Witten D., Hastie T., Tibshirani R., 6.3 Dimension Reduction Methods](http://faculty.marshall.usc.edu/gareth-james/ISL/)

/// PCA is a popular approach for deriving a low-dimensional set of features from a large set of variables.
pub mod pca;
