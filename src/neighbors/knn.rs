use serde::{Deserialize, Serialize};

use crate::algorithm::neighbour::cover_tree::CoverTree;
use crate::algorithm::neighbour::linear_search::LinearKNNSearch;
use crate::linalg::{row_iter, Matrix};
use crate::math::distance::Distance;
use crate::math::num::FloatExt;

#[derive(Serialize, Deserialize, Debug)]
pub struct KNNClassifier<T: FloatExt, D: Distance<Vec<T>, T>> {
    classes: Vec<T>,
    y: Vec<usize>,
    knn_algorithm: KNNAlgorithmV<T, D>,
    k: usize,
}

pub enum KNNAlgorithmName {
    LinearSearch,
    CoverTree,
}

#[derive(Serialize, Deserialize, Debug)]
pub enum KNNAlgorithmV<T: FloatExt, D: Distance<Vec<T>, T>> {
    LinearSearch(LinearKNNSearch<Vec<T>, T, D>),
    CoverTree(CoverTree<Vec<T>, T, D>),
}

impl KNNAlgorithmName {
    fn fit<T: FloatExt, D: Distance<Vec<T>, T>>(
        &self,
        data: Vec<Vec<T>>,
        distance: D,
    ) -> KNNAlgorithmV<T, D> {
        match *self {
            KNNAlgorithmName::LinearSearch => {
                KNNAlgorithmV::LinearSearch(LinearKNNSearch::new(data, distance))
            }
            KNNAlgorithmName::CoverTree => KNNAlgorithmV::CoverTree(CoverTree::new(data, distance)),
        }
    }
}

impl<T: FloatExt, D: Distance<Vec<T>, T>> KNNAlgorithmV<T, D> {
    fn find(&self, from: &Vec<T>, k: usize) -> Vec<usize> {
        match *self {
            KNNAlgorithmV::LinearSearch(ref linear) => linear.find(from, k),
            KNNAlgorithmV::CoverTree(ref cover) => cover.find(from, k),
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
        k: usize,
        distance: D,
        algorithm: KNNAlgorithmName,
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

        assert!(k > 1, format!("k should be > 1, k=[{}]", k));

        KNNClassifier {
            classes: classes,
            y: yi,
            k: k,
            knn_algorithm: algorithm.fit(data, distance),
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
            3,
            Distances::euclidian(),
            KNNAlgorithmName::LinearSearch,
        );
        let r = knn.predict(&x);
        assert_eq!(5, Vec::len(&r));
        assert_eq!(y.to_vec(), r);
    }

    #[test]
    fn serde() {
        let x = DenseMatrix::from_array(&[&[1., 2.], &[3., 4.], &[5., 6.], &[7., 8.], &[9., 10.]]);
        let y = vec![2., 2., 2., 3., 3.];

        let knn = KNNClassifier::fit(
            &x,
            &y,
            3,
            Distances::euclidian(),
            KNNAlgorithmName::CoverTree,
        );

        let deserialized_knn = bincode::deserialize(&bincode::serialize(&knn).unwrap()).unwrap();

        assert_eq!(knn, deserialized_knn);
    }
}
