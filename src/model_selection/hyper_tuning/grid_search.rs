// TODO: missing documentation

use crate::{
    api::{Predictor, SupervisedEstimator},
    error::{Failed, FailedError},
    linalg::basic::arrays::{Array1, Array2},
    numbers::basenum::Number,
    numbers::realnum::RealNumber,
};

use crate::model_selection::{cross_validate, BaseKFold, CrossValidationResult};

/// Parameters for GridSearchCV
#[derive(Debug)]
pub struct GridSearchCVParameters<
    T: Number,
    M: Array2<T>,
    C: Clone,
    I: Iterator<Item = C>,
    E: Predictor<M, M::RowVector>,
    F: Fn(&M, &M::RowVector, C) -> Result<E, Failed>,
    K: BaseKFold,
    S: Fn(&M::RowVector, &M::RowVector) -> T,
> {
    _phantom: std::marker::PhantomData<(T, M)>,

    parameters_search: I,
    estimator: F,
    score: S,
    cv: K,
}

impl<
        T: RealNumber,
        M: Array2<T>,
        C: Clone,
        I: Iterator<Item = C>,
        E: Predictor<M, M::RowVector>,
        F: Fn(&M, &M::RowVector, C) -> Result<E, Failed>,
        K: BaseKFold,
        S: Fn(&M::RowVector, &M::RowVector) -> T,
    > GridSearchCVParameters<T, M, C, I, E, F, K, S>
{
    /// Create new GridSearchCVParameters
    pub fn new(parameters_search: I, estimator: F, score: S, cv: K) -> Self {
        GridSearchCVParameters {
            _phantom: std::marker::PhantomData,
            parameters_search,
            estimator,
            score,
            cv,
        }
    }
}
/// Exhaustive search over specified parameter values for an estimator.
#[derive(Debug)]
pub struct GridSearchCV<T: RealNumber, M: Array2<T>, C: Clone, E: Predictor<M, M::RowVector>> {
    _phantom: std::marker::PhantomData<(T, M)>,
    predictor: E,
    /// Cross validation results.
    pub cross_validation_result: CrossValidationResult<T>,
    /// best parameter
    pub best_parameter: C,
}

impl<T: RealNumber, M: Array2<T>, E: Predictor<M, M::RowVector>, C: Clone>
    GridSearchCV<T, M, C, E>
{
    ///  Search for the best estimator by testing all possible combinations with cross-validation using given metric.
    /// * `x` - features, matrix of size _NxM_ where _N_ is number of samples and _M_ is number of attributes.
    /// * `y` - target values, should be of size _N_
    /// * `gs_parameters` - GridSearchCVParameters struct
    pub fn fit<
        I: Iterator<Item = C>,
        K: BaseKFold,
        F: Fn(&M, &M::RowVector, C) -> Result<E, Failed>,
        S: Fn(&M::RowVector, &M::RowVector) -> T,
    >(
        x: &M,
        y: &M::RowVector,
        gs_parameters: GridSearchCVParameters<T, M, C, I, E, F, K, S>,
    ) -> Result<Self, Failed> {
        let mut best_result: Option<CrossValidationResult<T>> = None;
        let mut best_parameters = None;
        let parameters_search = gs_parameters.parameters_search;
        let estimator = gs_parameters.estimator;
        let cv = gs_parameters.cv;
        let score = gs_parameters.score;

        for parameters in parameters_search {
            let result = cross_validate(&estimator, x, y, &parameters, &cv, &score)?;
            if best_result.is_none()
                || result.mean_test_score() > best_result.as_ref().unwrap().mean_test_score()
            {
                best_parameters = Some(parameters);
                best_result = Some(result);
            }
        }

        if let (Some(best_parameter), Some(cross_validation_result)) =
            (best_parameters, best_result)
        {
            let predictor = estimator(x, y, best_parameter.clone())?;
            Ok(Self {
                _phantom: gs_parameters._phantom,
                predictor,
                cross_validation_result,
                best_parameter,
            })
        } else {
            Err(Failed::because(
                FailedError::FindFailed,
                "there were no parameter sets found",
            ))
        }
    }

    /// Return grid search cross validation results
    pub fn cv_results(&self) -> &CrossValidationResult<T> {
        &self.cross_validation_result
    }

    /// Return best parameters found
    pub fn best_parameters(&self) -> &C {
        &self.best_parameter
    }

    /// Call predict on the estimator with the best found parameters
    pub fn predict(&self, x: &M) -> Result<M::RowVector, Failed> {
        self.predictor.predict(x)
    }
}

impl<
        T: RealNumber,
        M: Array2<T>,
        C: Clone,
        I: Iterator<Item = C>,
        E: Predictor<M, M::RowVector>,
        F: Fn(&M, &M::RowVector, C) -> Result<E, Failed>,
        K: BaseKFold,
        S: Fn(&M::RowVector, &M::RowVector) -> T,
    > SupervisedEstimator<M, M::RowVector, GridSearchCVParameters<T, M, C, I, E, F, K, S>>
    for GridSearchCV<T, M, C, E>
{
    fn fit(
        x: &M,
        y: &M::RowVector,
        parameters: GridSearchCVParameters<T, M, C, I, E, F, K, S>,
    ) -> Result<Self, Failed> {
        GridSearchCV::fit(x, y, parameters)
    }
}

impl<T: RealNumber, M: Array2<T>, C: Clone, E: Predictor<M, M::RowVector>>
    Predictor<M, M::RowVector> for GridSearchCV<T, M, C, E>
{
    fn predict(&self, x: &M) -> Result<M::RowVector, Failed> {
        self.predict(x)
    }
}

#[cfg(test)]
mod tests {

    use crate::{
        linalg::naive::dense_matrix::DenseMatrix,
        linear::logistic_regression::{LogisticRegression, LogisticRegressionSearchParameters},
        metrics::accuracy,
        model_selection::{
            hyper_tuning::grid_search::{self, GridSearchCVParameters},
            KFold,
        },
    };
    use grid_search::GridSearchCV;

    #[test]
    fn test_grid_search() {
        let x = DenseMatrix::from_2d_array(&[
            &[5.1, 3.5, 1.4, 0.2],
            &[4.9, 3.0, 1.4, 0.2],
            &[4.7, 3.2, 1.3, 0.2],
            &[4.6, 3.1, 1.5, 0.2],
            &[5.0, 3.6, 1.4, 0.2],
            &[5.4, 3.9, 1.7, 0.4],
            &[4.6, 3.4, 1.4, 0.3],
            &[5.0, 3.4, 1.5, 0.2],
            &[4.4, 2.9, 1.4, 0.2],
            &[4.9, 3.1, 1.5, 0.1],
            &[7.0, 3.2, 4.7, 1.4],
            &[6.4, 3.2, 4.5, 1.5],
            &[6.9, 3.1, 4.9, 1.5],
            &[5.5, 2.3, 4.0, 1.3],
            &[6.5, 2.8, 4.6, 1.5],
            &[5.7, 2.8, 4.5, 1.3],
            &[6.3, 3.3, 4.7, 1.6],
            &[4.9, 2.4, 3.3, 1.0],
            &[6.6, 2.9, 4.6, 1.3],
            &[5.2, 2.7, 3.9, 1.4],
        ]);
        let y = vec![
            0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        ];

        let cv = KFold {
            n_splits: 5,
            ..KFold::default()
        };

        let parameters = LogisticRegressionSearchParameters {
            alpha: vec![0., 1.],
            ..Default::default()
        };

        let grid_search = GridSearchCV::fit(
            &x,
            &y,
            GridSearchCVParameters {
                estimator: LogisticRegression::fit,
                score: accuracy,
                cv,
                parameters_search: parameters.into_iter(),
                _phantom: Default::default(),
            },
        )
        .unwrap();
        let best_parameters = grid_search.best_parameters();

        assert!([1.].contains(&best_parameters.alpha));

        let cv_results = grid_search.cv_results();

        assert_eq!(cv_results.mean_test_score(), 0.9);

        let x = DenseMatrix::from_2d_array(&[&[5., 3., 1., 0.]]);
        let result = grid_search.predict(&x).unwrap();
        assert_eq!(result, vec![0.]);
    }
}
