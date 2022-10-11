//! # Linear Models
//! Linear models describe a continuous response variable as a function of one or more predictor variables.
//! The model describes the relationship between a dependent variable y (also called the response) as a function of one or more independent, or explanatory variables \\(X_i\\). The general equation for a linear model is:
//! \\[y = \beta_0 + \sum_{i=1}^n \beta_iX_i + \epsilon\\]
//!
//! where \\(\beta_0 \\) is the intercept term (the expected value of Y when X = 0), \\(\epsilon \\) is an error term that is is independent of X and \\(\beta_i \\)
//! is the average increase in y associated with a one-unit increase in \\(X_i\\)
//!
//! Model assumptions:
//! * _Linearity_. The relationship between X and the mean of y is linear.
//! * _Constant variance_. The variance of residual is the same for any value of X.
//! * _Normality_. For any fixed value of X, Y is normally distributed.
//! * _Independence_. Observations are independent of each other.
//!
//! ## References:
//!
//! * ["An Introduction to Statistical Learning", James G., Witten D., Hastie T., Tibshirani R., 3. Linear Regression](http://faculty.marshall.usc.edu/gareth-james/ISL/)
//! * ["The Statistical Sleuth, A Course in Methods of Data Analysis", Ramsey F.L., Schafer D.W., Ch 7, 8, 3rd edition, 2013](http://www.statisticalsleuth.com/)
//!
//! <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
//! <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

pub mod bg_solver;
pub mod elastic_net;
pub mod lasso;
pub mod lasso_optimizer;
pub mod linear_regression;
pub mod logistic_regression;
pub mod ridge_regression;
