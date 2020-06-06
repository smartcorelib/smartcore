extern crate rand;

use std::default::Default;
use std::fmt::Debug;

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::linalg::Matrix;
use crate::math::num::FloatExt;
use crate::tree::decision_tree_regressor::{
    DecisionTreeRegressor, DecisionTreeRegressorParameters,
};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RandomForestRegressorParameters {
    pub max_depth: Option<u16>,
    pub min_samples_leaf: usize,
    pub min_samples_split: usize,
    pub n_trees: usize,
    pub mtry: Option<usize>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RandomForestRegressor<T: FloatExt> {
    parameters: RandomForestRegressorParameters,
    trees: Vec<DecisionTreeRegressor<T>>,
}

impl Default for RandomForestRegressorParameters {
    fn default() -> Self {
        RandomForestRegressorParameters {
            max_depth: None,
            min_samples_leaf: 1,
            min_samples_split: 2,
            n_trees: 10,
            mtry: Option::None,
        }
    }
}

impl<T: FloatExt> PartialEq for RandomForestRegressor<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.trees.len() != other.trees.len() {
            return false;
        } else {
            for i in 0..self.trees.len() {
                if self.trees[i] != other.trees[i] {
                    return false;
                }
            }
            true
        }
    }
}

impl<T: FloatExt> RandomForestRegressor<T> {
    pub fn fit<M: Matrix<T>>(
        x: &M,
        y: &M::RowVector,
        parameters: RandomForestRegressorParameters,
    ) -> RandomForestRegressor<T> {
        let (n_rows, num_attributes) = x.shape();

        let mtry = parameters
            .mtry
            .unwrap_or((num_attributes as f64).sqrt().floor() as usize);

        let mut trees: Vec<DecisionTreeRegressor<T>> = Vec::new();

        for _ in 0..parameters.n_trees {
            let samples = RandomForestRegressor::<T>::sample_with_replacement(n_rows);
            let params = DecisionTreeRegressorParameters {
                max_depth: parameters.max_depth,
                min_samples_leaf: parameters.min_samples_leaf,
                min_samples_split: parameters.min_samples_split,
            };
            let tree = DecisionTreeRegressor::fit_weak_learner(x, y, samples, mtry, params);
            trees.push(tree);
        }

        RandomForestRegressor {
            parameters: parameters,
            trees: trees,
        }
    }

    pub fn predict<M: Matrix<T>>(&self, x: &M) -> M::RowVector {
        let mut result = M::zeros(1, x.shape().0);

        let (n, _) = x.shape();

        for i in 0..n {
            result.set(0, i, self.predict_for_row(x, i));
        }

        result.to_row_vector()
    }

    fn predict_for_row<M: Matrix<T>>(&self, x: &M, row: usize) -> T {
        let n_trees = self.trees.len();

        let mut result = T::zero();

        for tree in self.trees.iter() {
            result = result + tree.predict_for_row(x, row);
        }

        result / T::from(n_trees).unwrap()
    }

    fn sample_with_replacement(nrows: usize) -> Vec<usize> {
        let mut rng = rand::thread_rng();
        let mut samples = vec![0; nrows];
        for _ in 0..nrows {
            let xi = rng.gen_range(0, nrows);
            samples[xi] += 1;
        }
        samples
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::naive::dense_matrix::DenseMatrix;
    use ndarray::{arr1, arr2};

    #[test]
    fn fit_longley() {
        let x = DenseMatrix::from_array(&[
            &[234.289, 235.6, 159., 107.608, 1947., 60.323],
            &[259.426, 232.5, 145.6, 108.632, 1948., 61.122],
            &[258.054, 368.2, 161.6, 109.773, 1949., 60.171],
            &[284.599, 335.1, 165., 110.929, 1950., 61.187],
            &[328.975, 209.9, 309.9, 112.075, 1951., 63.221],
            &[346.999, 193.2, 359.4, 113.27, 1952., 63.639],
            &[365.385, 187., 354.7, 115.094, 1953., 64.989],
            &[363.112, 357.8, 335., 116.219, 1954., 63.761],
            &[397.469, 290.4, 304.8, 117.388, 1955., 66.019],
            &[419.18, 282.2, 285.7, 118.734, 1956., 67.857],
            &[442.769, 293.6, 279.8, 120.445, 1957., 68.169],
            &[444.546, 468.1, 263.7, 121.95, 1958., 66.513],
            &[482.704, 381.3, 255.2, 123.366, 1959., 68.655],
            &[502.601, 393.1, 251.4, 125.368, 1960., 69.564],
            &[518.173, 480.6, 257.2, 127.852, 1961., 69.331],
            &[554.894, 400.7, 282.7, 130.081, 1962., 70.551],
        ]);
        let y = vec![
            83.0, 88.5, 88.2, 89.5, 96.2, 98.1, 99.0, 100.0, 101.2, 104.6, 108.4, 110.8, 112.6,
            114.2, 115.7, 116.9,
        ];

        let expected_y: Vec<f64> = vec![
            85., 88., 88., 89., 97., 98., 99., 99., 102., 104., 109., 110., 113., 114., 115., 116.,
        ];

        let y_hat = RandomForestRegressor::fit(
            &x,
            &y,
            RandomForestRegressorParameters {
                max_depth: None,
                min_samples_leaf: 1,
                min_samples_split: 2,
                n_trees: 1000,
                mtry: Option::None,
            },
        )
        .predict(&x);

        for i in 0..y_hat.len() {
            assert!((y_hat[i] - expected_y[i]).abs() < 1.0);
        }
    }

    #[test]
    fn my_fit_longley_ndarray() {
        let x = arr2(&[
            [234.289, 235.6, 159., 107.608, 1947., 60.323],
            [259.426, 232.5, 145.6, 108.632, 1948., 61.122],
            [258.054, 368.2, 161.6, 109.773, 1949., 60.171],
            [284.599, 335.1, 165., 110.929, 1950., 61.187],
            [328.975, 209.9, 309.9, 112.075, 1951., 63.221],
            [346.999, 193.2, 359.4, 113.27, 1952., 63.639],
            [365.385, 187., 354.7, 115.094, 1953., 64.989],
            [363.112, 357.8, 335., 116.219, 1954., 63.761],
            [397.469, 290.4, 304.8, 117.388, 1955., 66.019],
            [419.18, 282.2, 285.7, 118.734, 1956., 67.857],
            [442.769, 293.6, 279.8, 120.445, 1957., 68.169],
            [444.546, 468.1, 263.7, 121.95, 1958., 66.513],
            [482.704, 381.3, 255.2, 123.366, 1959., 68.655],
            [502.601, 393.1, 251.4, 125.368, 1960., 69.564],
            [518.173, 480.6, 257.2, 127.852, 1961., 69.331],
            [554.894, 400.7, 282.7, 130.081, 1962., 70.551],
        ]);
        let y = arr1(&[
            83.0, 88.5, 88.2, 89.5, 96.2, 98.1, 99.0, 100.0, 101.2, 104.6, 108.4, 110.8, 112.6,
            114.2, 115.7, 116.9,
        ]);

        let expected_y: Vec<f64> = vec![
            85., 88., 88., 89., 97., 98., 99., 99., 102., 104., 109., 110., 113., 114., 115., 116.,
        ];

        let y_hat = RandomForestRegressor::fit(
            &x,
            &y,
            RandomForestRegressorParameters {
                max_depth: None,
                min_samples_leaf: 1,
                min_samples_split: 2,
                n_trees: 1000,
                mtry: Option::None,
            },
        )
        .predict(&x);

        for i in 0..y_hat.len() {
            assert!((y_hat[i] - expected_y[i]).abs() < 1.0);
        }
    }

    #[test]
    fn serde() {
        let x = DenseMatrix::from_array(&[
            &[234.289, 235.6, 159., 107.608, 1947., 60.323],
            &[259.426, 232.5, 145.6, 108.632, 1948., 61.122],
            &[258.054, 368.2, 161.6, 109.773, 1949., 60.171],
            &[284.599, 335.1, 165., 110.929, 1950., 61.187],
            &[328.975, 209.9, 309.9, 112.075, 1951., 63.221],
            &[346.999, 193.2, 359.4, 113.27, 1952., 63.639],
            &[365.385, 187., 354.7, 115.094, 1953., 64.989],
            &[363.112, 357.8, 335., 116.219, 1954., 63.761],
            &[397.469, 290.4, 304.8, 117.388, 1955., 66.019],
            &[419.18, 282.2, 285.7, 118.734, 1956., 67.857],
            &[442.769, 293.6, 279.8, 120.445, 1957., 68.169],
            &[444.546, 468.1, 263.7, 121.95, 1958., 66.513],
            &[482.704, 381.3, 255.2, 123.366, 1959., 68.655],
            &[502.601, 393.1, 251.4, 125.368, 1960., 69.564],
            &[518.173, 480.6, 257.2, 127.852, 1961., 69.331],
            &[554.894, 400.7, 282.7, 130.081, 1962., 70.551],
        ]);
        let y = vec![
            83.0, 88.5, 88.2, 89.5, 96.2, 98.1, 99.0, 100.0, 101.2, 104.6, 108.4, 110.8, 112.6,
            114.2, 115.7, 116.9,
        ];

        let forest = RandomForestRegressor::fit(&x, &y, Default::default());

        let deserialized_forest: RandomForestRegressor<f64> =
            bincode::deserialize(&bincode::serialize(&forest).unwrap()).unwrap();

        assert_eq!(forest, deserialized_forest);
    }
}
