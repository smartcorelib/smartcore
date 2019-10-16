use crate::linalg::Matrix;
use crate::regression::Regression;
use std::fmt::Debug;

#[derive(Debug)]
pub enum LinearRegressionSolver {
    QR,
    SVD
}

#[derive(Debug)]
pub struct LinearRegression<M: Matrix> {    
    coefficients: M,
    intercept: f64,
    solver: LinearRegressionSolver
}

impl<M: Matrix> LinearRegression<M> {

    pub fn fit(x: &M, y: &M, solver: LinearRegressionSolver) -> LinearRegression<M>{

        let (x_nrows, num_attributes) = x.shape();
        let (y_nrows, _) = y.shape();

        if x_nrows != y_nrows {
            panic!("Number of rows of X doesn't match number of rows of Y");
        }

        // let b = y.v_stack(&M::ones(1, 1));
        let b = y.clone();
        let mut a = x.h_stack(&M::ones(x_nrows, 1));

        let w = match solver {
            LinearRegressionSolver::QR => a.qr_solve_mut(b),
            LinearRegressionSolver::SVD => a.svd_solve_mut(b)
        };

        let wights = w.slice(0..num_attributes, 0..1);  

        LinearRegression {
            intercept: w.get(num_attributes, 0),
            coefficients: wights,
            solver: solver
        }
    }

}

impl<M: Matrix> Regression<M> for LinearRegression<M> {


    fn predict(&self, x: &M) -> M {
        let (nrows, _) = x.shape();
        let mut y_hat = x.dot(&self.coefficients);
        y_hat.add_mut(&M::fill(nrows, 1, self.intercept));
        y_hat
    }

}

#[cfg(test)]
mod tests {    
    use super::*; 
    use crate::linalg::naive::dense_matrix::DenseMatrix;

    #[test]
    fn knn_fit_predict() { 

            let x = DenseMatrix::from_2d_array(&[
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
                &[554.894, 400.7, 282.7, 130.081, 1962., 70.551]]);
            let y = DenseMatrix::from_array(16, 1, &[83.0,  88.5,  88.2,  89.5,  96.2,  98.1,  99.0, 100.0, 101.2, 104.6, 108.4, 110.8, 112.6, 114.2, 115.7, 116.9]);        

            let y_hat_qr = LinearRegression::fit(&x, &y, LinearRegressionSolver::QR).predict(&x);                        

            let y_hat_svd = LinearRegression::fit(&x, &y, LinearRegressionSolver::SVD).predict(&x);

            assert!(y.approximate_eq(&y_hat_qr, 5.));
            assert!(y.approximate_eq(&y_hat_svd, 5.));


    }
}