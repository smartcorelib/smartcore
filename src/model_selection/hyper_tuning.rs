/// grid search results.
#[derive(Clone, Debug)]
pub struct GridSearchResult<T: RealNumber, I: Clone> {
    /// Vector with test scores on each cv split
    pub cross_validation_result: CrossValidationResult<T>,
    /// Vector with training scores on each cv split
    pub parameters: I,
}

/// Search for the best estimator by testing all possible combinations with cross-validation using given metric.
/// * `fit_estimator` - a `fit` function of an estimator
/// * `x` - features, matrix of size _NxM_ where _N_ is number of samples and _M_ is number of attributes.
/// * `y` - target values, should be of size _N_
/// * `parameter_search` - an iterator for parameters that will be tested.
/// * `cv` - the cross-validation splitting strategy, should be an instance of [`BaseKFold`](./trait.BaseKFold.html)
/// * `score` - a metric to use for evaluation, see [metrics](../metrics/index.html)
pub fn grid_search<T, M, I, E, K, F, S>(
    fit_estimator: F,
    x: &M,
    y: &M::RowVector,
    parameter_search: I,
    cv: K,
    score: S,
) -> Result<GridSearchResult<T, I::Item>, Failed>
where
    T: RealNumber,
    M: Matrix<T>,
    I: Iterator,
    I::Item: Clone,
    E: Predictor<M, M::RowVector>,
    K: BaseKFold,
    F: Fn(&M, &M::RowVector, I::Item) -> Result<E, Failed>,
    S: Fn(&M::RowVector, &M::RowVector) -> T,
{
    let mut best_result: Option<CrossValidationResult<T>> = None;
    let mut best_parameters = None;

    for parameters in parameter_search {
        let result = cross_validate(&fit_estimator, x, y, &parameters, &cv, &score)?;
        if best_result.is_none()
            || result.mean_test_score() > best_result.as_ref().unwrap().mean_test_score()
        {
            best_parameters = Some(parameters);
            best_result = Some(result);
        }
    }

    if let (Some(parameters), Some(cross_validation_result)) = (best_parameters, best_result) {
        Ok(GridSearchResult {
            cross_validation_result,
            parameters,
        })
    } else {
        Err(Failed::because(
            FailedError::FindFailed,
            "there were no parameter sets found",
        ))
    }
}

#[cfg(test)]
mod tests {
  use crate::linear::logistic_regression::{
    LogisticRegression, LogisticRegressionSearchParameters,
};

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

      let results = grid_search(
          LogisticRegression::fit,
          &x,
          &y,
          parameters.into_iter(),
          cv,
          &accuracy,
      )
      .unwrap();

      assert!([0., 1.].contains(&results.parameters.alpha));
  }
}
