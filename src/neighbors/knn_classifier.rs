use serde::{Deserialize, Serialize};

use crate::linalg::{row_iter, Matrix};
use crate::math::distance::Distance;
use crate::math::num::FloatExt;
use crate::neighbors::{KNNAlgorithm, KNNAlgorithmName};

#[derive(Serialize, Deserialize, Debug)]
pub struct KNNClassifierParameters {
    pub algorithm: KNNAlgorithmName,
    pub k: usize,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct KNNClassifier<T: FloatExt, D: Distance<Vec<T>, T>> {
    classes: Vec<T>,
    y: Vec<usize>,
    knn_algorithm: KNNAlgorithm<T, D>,
    k: usize,
}

impl Default for KNNClassifierParameters {
    fn default() -> Self {
        KNNClassifierParameters {
            algorithm: KNNAlgorithmName::CoverTree,
            k: 3,
        }
    }
}

impl<T: FloatExt, D: Distance<Vec<T>, T>> PartialEq for KNNClassifier<T, D> {
    fn eq(&self, other: &Self) -> bool {
        if self.classes.len() != other.classes.len()
            || self.k != other.k
            || self.y.len() != other.y.len()
        {
            return false;
        } else {
            for i in 0..self.classes.len() {
                if (self.classes[i] - other.classes[i]).abs() > T::epsilon() {
                    return false;
                }
            }
            for i in 0..self.y.len() {
                if self.y[i] != other.y[i] {
                    return false;
                }
            }
            true
        }
    }
}

impl<T: FloatExt, D: Distance<Vec<T>, T>> KNNClassifier<T, D> {
    pub fn fit<M: Matrix<T>>(
        x: &M,
        y: &M::RowVector,
        distance: D,
        parameters: KNNClassifierParameters,
    ) -> KNNClassifier<T, D> {
        let y_m = M::from_row_vector(y.clone());

        let (_, y_n) = y_m.shape();
        let (x_n, _) = x.shape();

        let data = row_iter(x).collect();

        let mut yi: Vec<usize> = vec![0; y_n];
        let classes = y_m.unique();

        for i in 0..y_n {
            let yc = y_m.get(0, i);
            yi[i] = classes.iter().position(|c| yc == *c).unwrap();
        }

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

        KNNClassifier {
            classes: classes,
            y: yi,
            k: parameters.k,
            knn_algorithm: parameters.algorithm.fit(data, distance),
        }
    }

    pub fn predict<M: Matrix<T>>(&self, x: &M) -> M::RowVector {
        let mut result = M::zeros(1, x.shape().0);

        row_iter(x)
            .enumerate()
            .for_each(|(i, x)| result.set(0, i, self.classes[self.predict_for_row(x)]));

        result.to_row_vector()
    }

    fn predict_for_row(&self, x: Vec<T>) -> usize {
        let idxs = self.knn_algorithm.find(&x, self.k);
        let mut c = vec![0; self.classes.len()];
        let mut max_c = 0;
        let mut max_i = 0;
        for i in idxs {
            c[self.y[i]] += 1;
            if c[self.y[i]] > max_c {
                max_c = c[self.y[i]];
                max_i = self.y[i];
            }
        }

        max_i
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
        let y = vec![2., 2., 2., 3., 3.];
        let knn = KNNClassifier::fit(
            &x,
            &y,
            Distances::euclidian(),
            KNNClassifierParameters {
                k: 3,
                algorithm: KNNAlgorithmName::LinearSearch,
            },
        );
        let r = knn.predict(&x);
        assert_eq!(5, Vec::len(&r));
        assert_eq!(y.to_vec(), r);
    }

    #[test]
    fn serde() {
        let x = DenseMatrix::from_array(&[&[1., 2.], &[3., 4.], &[5., 6.], &[7., 8.], &[9., 10.]]);
        let y = vec![2., 2., 2., 3., 3.];

        let knn = KNNClassifier::fit(&x, &y, Distances::euclidian(), Default::default());

        let deserialized_knn = bincode::deserialize(&bincode::serialize(&knn).unwrap()).unwrap();

        assert_eq!(knn, deserialized_knn);
    }
}
