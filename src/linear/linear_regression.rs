use std::fmt::Debug;

use serde::{Deserialize, Serialize};

use crate::linalg::Matrix;
use crate::math::num::FloatExt;

#[derive(Serialize, Deserialize, Debug)]
pub enum LinearRegressionSolverName {
    QR,
    SVD,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct LinearRegressionParameters {
    solver: LinearRegressionSolverName,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct LinearRegression<T: FloatExt, M: Matrix<T>> {
    coefficients: M,
    intercept: T,
    solver: LinearRegressionSolverName,
}

impl Default for LinearRegressionParameters {
    fn default() -> Self {
        LinearRegressionParameters {
            solver: LinearRegressionSolverName::SVD,
        }
    }
}

impl<T: FloatExt, M: Matrix<T>> PartialEq for LinearRegression<T, M> {
    fn eq(&self, other: &Self) -> bool {
        self.coefficients == other.coefficients
            && (self.intercept - other.intercept).abs() <= T::epsilon()
    }
}

impl<T: FloatExt, M: Matrix<T>> LinearRegression<T, M> {
    pub fn fit(
        x: &M,
        y: &M::RowVector,
        parameters: LinearRegressionParameters,
    ) -> LinearRegression<T, M> {
        let y_m = M::from_row_vector(y.clone());
        let b = y_m.transpose();
        let (x_nrows, num_attributes) = x.shape();
        let (y_nrows, _) = b.shape();

        if x_nrows != y_nrows {
            panic!("Number of rows of X doesn't match number of rows of Y");
        }

        let a = x.v_stack(&M::ones(x_nrows, 1));

        let w = match parameters.solver {
            LinearRegressionSolverName::QR => a.qr_solve_mut(b),
            LinearRegressionSolverName::SVD => a.svd_solve_mut(b),
        };

        let wights = w.slice(0..num_attributes, 0..1);

        LinearRegression {
            intercept: w.get(num_attributes, 0),
            coefficients: wights,
            solver: parameters.solver,
        }
    }

    pub fn predict(&self, x: &M) -> M::RowVector {
        let (nrows, _) = x.shape();
        let mut y_hat = x.dot(&self.coefficients);
        y_hat.add_mut(&M::fill(nrows, 1, self.intercept));
        y_hat.transpose().to_row_vector()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::naive::dense_matrix::*;
    use nalgebra::{DMatrix, RowDVector};

    #[test]
    fn ols_fit_predict() {
        let x = DMatrix::from_row_slice(
            16,
            6,
            &[
                234.289, 235.6, 159.0, 107.608, 1947., 60.323, 259.426, 232.5, 145.6, 108.632,
                1948., 61.122, 258.054, 368.2, 161.6, 109.773, 1949., 60.171, 284.599, 335.1,
                165.0, 110.929, 1950., 61.187, 328.975, 209.9, 309.9, 112.075, 1951., 63.221,
                346.999, 193.2, 359.4, 113.270, 1952., 63.639, 365.385, 187.0, 354.7, 115.094,
                1953., 64.989, 363.112, 357.8, 335.0, 116.219, 1954., 63.761, 397.469, 290.4,
                304.8, 117.388, 1955., 66.019, 419.180, 282.2, 285.7, 118.734, 1956., 67.857,
                442.769, 293.6, 279.8, 120.445, 1957., 68.169, 444.546, 468.1, 263.7, 121.950,
                1958., 66.513, 482.704, 381.3, 255.2, 123.366, 1959., 68.655, 502.601, 393.1,
                251.4, 125.368, 1960., 69.564, 518.173, 480.6, 257.2, 127.852, 1961., 69.331,
                554.894, 400.7, 282.7, 130.081, 1962., 70.551,
            ],
        );

        let y: RowDVector<f64> = RowDVector::from_vec(vec![
            83.0, 88.5, 88.2, 89.5, 96.2, 98.1, 99.0, 100.0, 101.2, 104.6, 108.4, 110.8, 112.6,
            114.2, 115.7, 116.9,
        ]);

        let y_hat_qr = LinearRegression::fit(
            &x,
            &y,
            LinearRegressionParameters {
                solver: LinearRegressionSolverName::QR,
            },
        )
        .predict(&x);

        let y_hat_svd = LinearRegression::fit(&x, &y, Default::default()).predict(&x);

        assert!(y
            .iter()
            .zip(y_hat_qr.iter())
            .all(|(&a, &b)| (a - b).abs() <= 5.0));
        assert!(y
            .iter()
            .zip(y_hat_svd.iter())
            .all(|(&a, &b)| (a - b).abs() <= 5.0));
    }

    #[test]
    fn ols_fit_predict_nalgebra() {
        let x = DenseMatrix::from_array(&[
            &[234.289, 235.6, 159.0, 107.608, 1947., 60.323],
            &[259.426, 232.5, 145.6, 108.632, 1948., 61.122],
            &[258.054, 368.2, 161.6, 109.773, 1949., 60.171],
            &[284.599, 335.1, 165.0, 110.929, 1950., 61.187],
            &[328.975, 209.9, 309.9, 112.075, 1951., 63.221],
            &[346.999, 193.2, 359.4, 113.270, 1952., 63.639],
            &[365.385, 187.0, 354.7, 115.094, 1953., 64.989],
            &[363.112, 357.8, 335.0, 116.219, 1954., 63.761],
            &[397.469, 290.4, 304.8, 117.388, 1955., 66.019],
            &[419.180, 282.2, 285.7, 118.734, 1956., 67.857],
            &[442.769, 293.6, 279.8, 120.445, 1957., 68.169],
            &[444.546, 468.1, 263.7, 121.950, 1958., 66.513],
            &[482.704, 381.3, 255.2, 123.366, 1959., 68.655],
            &[502.601, 393.1, 251.4, 125.368, 1960., 69.564],
            &[518.173, 480.6, 257.2, 127.852, 1961., 69.331],
            &[554.894, 400.7, 282.7, 130.081, 1962., 70.551],
        ]);

        let y: Vec<f64> = vec![
            83.0, 88.5, 88.2, 89.5, 96.2, 98.1, 99.0, 100.0, 101.2, 104.6, 108.4, 110.8, 112.6,
            114.2, 115.7, 116.9,
        ];

        let y_hat_qr = LinearRegression::fit(
            &x,
            &y,
            LinearRegressionParameters {
                solver: LinearRegressionSolverName::QR,
            },
        )
        .predict(&x);

        let y_hat_svd = LinearRegression::fit(&x, &y, Default::default()).predict(&x);

        assert!(y
            .iter()
            .zip(y_hat_qr.iter())
            .all(|(&a, &b)| (a - b).abs() <= 5.0));
        assert!(y
            .iter()
            .zip(y_hat_svd.iter())
            .all(|(&a, &b)| (a - b).abs() <= 5.0));
    }

    #[test]
    fn serde() {
        let x = DenseMatrix::from_array(&[
            &[234.289, 235.6, 159.0, 107.608, 1947., 60.323],
            &[259.426, 232.5, 145.6, 108.632, 1948., 61.122],
            &[258.054, 368.2, 161.6, 109.773, 1949., 60.171],
            &[284.599, 335.1, 165.0, 110.929, 1950., 61.187],
            &[328.975, 209.9, 309.9, 112.075, 1951., 63.221],
            &[346.999, 193.2, 359.4, 113.270, 1952., 63.639],
            &[365.385, 187.0, 354.7, 115.094, 1953., 64.989],
            &[363.112, 357.8, 335.0, 116.219, 1954., 63.761],
            &[397.469, 290.4, 304.8, 117.388, 1955., 66.019],
            &[419.180, 282.2, 285.7, 118.734, 1956., 67.857],
            &[442.769, 293.6, 279.8, 120.445, 1957., 68.169],
            &[444.546, 468.1, 263.7, 121.950, 1958., 66.513],
            &[482.704, 381.3, 255.2, 123.366, 1959., 68.655],
            &[502.601, 393.1, 251.4, 125.368, 1960., 69.564],
            &[518.173, 480.6, 257.2, 127.852, 1961., 69.331],
            &[554.894, 400.7, 282.7, 130.081, 1962., 70.551],
        ]);

        let y = vec![
            83.0, 88.5, 88.2, 89.5, 96.2, 98.1, 99.0, 100.0, 101.2, 104.6, 108.4, 110.8, 112.6,
            114.2, 115.7, 116.9,
        ];

        let lr = LinearRegression::fit(&x, &y, Default::default());

        let deserialized_lr: LinearRegression<f64, DenseMatrix<f64>> =
            serde_json::from_str(&serde_json::to_string(&lr).unwrap()).unwrap();

        assert_eq!(lr, deserialized_lr);
    }
}
