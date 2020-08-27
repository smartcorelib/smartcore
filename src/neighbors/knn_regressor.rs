use serde::{Deserialize, Serialize};

use crate::linalg::{row_iter, BaseVector, Matrix};
use crate::math::distance::Distance;
use crate::math::num::FloatExt;
use crate::neighbors::{KNNAlgorithm, KNNAlgorithmName};

#[derive(Serialize, Deserialize, Debug)]
pub struct KNNRegressorParameters {
    pub algorithm: KNNAlgorithmName,
    pub k: usize,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct KNNRegressor<T: FloatExt, D: Distance<Vec<T>, T>> {
    y: Vec<T>,
    knn_algorithm: KNNAlgorithm<T, D>,
    k: usize,
}

impl Default for KNNRegressorParameters {
    fn default() -> Self {
        KNNRegressorParameters {
            algorithm: KNNAlgorithmName::CoverTree,
            k: 3,
        }
    }
}

impl<T: FloatExt, D: Distance<Vec<T>, T>> PartialEq for KNNRegressor<T, D> {
    fn eq(&self, other: &Self) -> bool {
        if self.k != other.k || self.y.len() != other.y.len() {
            return false;
        } else {
            for i in 0..self.y.len() {
                if (self.y[i] - other.y[i]).abs() > T::epsilon() {
                    return false;
                }
            }
            true
        }
    }
}

impl<T: FloatExt, D: Distance<Vec<T>, T>> KNNRegressor<T, D> {
    pub fn fit<M: Matrix<T>>(
        x: &M,
        y: &M::RowVector,
        distance: D,
        parameters: KNNRegressorParameters,
    ) -> KNNRegressor<T, D> {
        let y_m = M::from_row_vector(y.clone());

        let (_, y_n) = y_m.shape();
        let (x_n, _) = x.shape();

        let data = row_iter(x).collect();

        assert!(
            x_n == y_n,
            format!(
                "Size of x should equal size of y; |x|=[{}], |y|=[{}]",
                x_n, y_n
            )
        );

        assert!(
            parameters.k > 1,
            format!("k should be > 1, k=[{}]", parameters.k)
        );

        KNNRegressor {
            y: y.to_vec(),
            k: parameters.k,
            knn_algorithm: parameters.algorithm.fit(data, distance),
        }
    }

    pub fn predict<M: Matrix<T>>(&self, x: &M) -> M::RowVector {
        let mut result = M::zeros(1, x.shape().0);

        row_iter(x)
            .enumerate()
            .for_each(|(i, x)| result.set(0, i, self.predict_for_row(x)));

        result.to_row_vector()
    }

    fn predict_for_row(&self, x: Vec<T>) -> T {
        let idxs = self.knn_algorithm.find(&x, self.k);
        let mut result = T::zero();
        for i in idxs {
            result = result + self.y[i];
        }

        result / T::from_usize(self.k).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::naive::dense_matrix::DenseMatrix;
    use crate::math::distance::Distances;

    #[test]
    fn knn_fit_predict() {
        let x = DenseMatrix::from_array(&[&[1., 2.], &[3., 4.], &[5., 6.], &[7., 8.], &[9., 10.]]);
        let y: Vec<f64> = vec![1., 2., 3., 4., 5.];
        let y_exp = vec![2., 2., 3., 4., 4.];
        let knn = KNNRegressor::fit(
            &x,
            &y,
            Distances::euclidian(),
            KNNRegressorParameters {
                k: 3,
                algorithm: KNNAlgorithmName::LinearSearch,
            },
        );
        let y_hat = knn.predict(&x);
        assert_eq!(5, Vec::len(&y_hat));
        for i in 0..y_hat.len() {
            assert!((y_hat[i] - y_exp[i]).abs() < std::f64::EPSILON);
        }
    }

    #[test]
    fn serde() {
        let x = DenseMatrix::from_array(&[&[1., 2.], &[3., 4.], &[5., 6.], &[7., 8.], &[9., 10.]]);
        let y = vec![1., 2., 3., 4., 5.];

        let knn = KNNRegressor::fit(&x, &y, Distances::euclidian(), Default::default());

        let deserialized_knn = bincode::deserialize(&bincode::serialize(&knn).unwrap()).unwrap();

        assert_eq!(knn, deserialized_knn);
    }
}
