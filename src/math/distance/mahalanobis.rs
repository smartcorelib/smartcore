#![allow(non_snake_case)]

use std::marker::PhantomData;

use serde::{Deserialize, Serialize};

use crate::math::num::FloatExt;

use super::Distance;
use crate::linalg::Matrix;

#[derive(Serialize, Deserialize, Debug)]
pub struct Mahalanobis<T: FloatExt, M: Matrix<T>> {
    pub sigma: M,
    pub sigmaInv: M,
    t: PhantomData<T>,
}

impl<T: FloatExt, M: Matrix<T>> Mahalanobis<T, M> {
    pub fn new(data: &M) -> Mahalanobis<T, M> {
        let sigma = data.cov();
        let sigmaInv = sigma.lu().inverse();
        Mahalanobis {
            sigma: sigma,
            sigmaInv: sigmaInv,
            t: PhantomData,
        }
    }

    pub fn new_from_covariance(cov: &M) -> Mahalanobis<T, M> {
        let sigma = cov.clone();
        let sigmaInv = sigma.lu().inverse();
        Mahalanobis {
            sigma: sigma,
            sigmaInv: sigmaInv,
            t: PhantomData,
        }
    }
}

impl<T: FloatExt, M: Matrix<T>> Distance<Vec<T>, T> for Mahalanobis<T, M> {
    fn distance(&self, x: &Vec<T>, y: &Vec<T>) -> T {
        let (nrows, ncols) = self.sigma.shape();
        if x.len() != nrows {
            panic!(
                "Array x[{}] has different dimension with Sigma[{}][{}].",
                x.len(),
                nrows,
                ncols
            );
        }

        if y.len() != nrows {
            panic!(
                "Array y[{}] has different dimension with Sigma[{}][{}].",
                y.len(),
                nrows,
                ncols
            );
        }

        println!("{}", self.sigmaInv);

        let n = x.len();
        let mut z = vec![T::zero(); n];
        for i in 0..n {
            z[i] = x[i] - y[i];
        }

        // np.dot(np.dot((a-b),VI),(a-b).T)
        let mut s = T::zero();
        for j in 0..n {
            for i in 0..n {
                s = s + self.sigmaInv.get(i, j) * z[i] * z[j];
            }
        }

        s.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::naive::dense_matrix::*;

    #[test]
    fn mahalanobis_distance() {
        let data = DenseMatrix::from_array(&[
            &[64., 580., 29.],
            &[66., 570., 33.],
            &[68., 590., 37.],
            &[69., 660., 46.],
            &[73., 600., 55.],
        ]);

        let a = data.column_mean();
        let b = vec![66., 640., 44.];

        let mahalanobis = Mahalanobis::new(&data);

        println!("{}", mahalanobis.distance(&a, &b));
    }
}
