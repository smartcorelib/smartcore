use std::fmt::Debug;
use std::marker::PhantomData;

use serde::{Deserialize, Serialize};

use crate::linalg::Matrix;
use crate::math::num::RealNumber;
use crate::optimization::first_order::lbfgs::LBFGS;
use crate::optimization::first_order::{FirstOrderOptimizer, OptimizerResult};
use crate::optimization::line_search::Backtracking;
use crate::optimization::FunctionOrder;

#[derive(Serialize, Deserialize, Debug)]
pub struct LogisticRegression<T: RealNumber, M: Matrix<T>> {
    weights: M,
    classes: Vec<T>,
    num_attributes: usize,
    num_classes: usize,
}

trait ObjectiveFunction<T: RealNumber, M: Matrix<T>> {
    fn f(&self, w_bias: &M) -> T;
    fn df(&self, g: &mut M, w_bias: &M);

    fn partial_dot(w: &M, x: &M, v_col: usize, m_row: usize) -> T {
        let mut sum = T::zero();
        let p = x.shape().1;
        for i in 0..p {
            sum = sum + x.get(m_row, i) * w.get(0, i + v_col);
        }

        sum + w.get(0, p + v_col)
    }
}

struct BinaryObjectiveFunction<'a, T: RealNumber, M: Matrix<T>> {
    x: &'a M,
    y: Vec<usize>,
    phantom: PhantomData<&'a T>,
}

impl<T: RealNumber, M: Matrix<T>> PartialEq for LogisticRegression<T, M> {
    fn eq(&self, other: &Self) -> bool {
        if self.num_classes != other.num_classes
            || self.num_attributes != other.num_attributes
            || self.classes.len() != other.classes.len()
        {
            return false;
        } else {
            for i in 0..self.classes.len() {
                if (self.classes[i] - other.classes[i]).abs() > T::epsilon() {
                    return false;
                }
            }

            return self.weights == other.weights;
        }
    }
}

impl<'a, T: RealNumber, M: Matrix<T>> ObjectiveFunction<T, M>
    for BinaryObjectiveFunction<'a, T, M>
{
    fn f(&self, w_bias: &M) -> T {
        let mut f = T::zero();
        let (n, _) = self.x.shape();

        for i in 0..n {
            let wx = BinaryObjectiveFunction::partial_dot(w_bias, self.x, 0, i);
            f = f + (wx.ln_1pe() - (T::from(self.y[i]).unwrap()) * wx);
        }

        f
    }

    fn df(&self, g: &mut M, w_bias: &M) {
        g.copy_from(&M::zeros(1, g.shape().1));

        let (n, p) = self.x.shape();

        for i in 0..n {
            let wx = BinaryObjectiveFunction::partial_dot(w_bias, self.x, 0, i);

            let dyi = (T::from(self.y[i]).unwrap()) - wx.sigmoid();
            for j in 0..p {
                g.set(0, j, g.get(0, j) - dyi * self.x.get(i, j));
            }
            g.set(0, p, g.get(0, p) - dyi);
        }
    }
}

struct MultiClassObjectiveFunction<'a, T: RealNumber, M: Matrix<T>> {
    x: &'a M,
    y: Vec<usize>,
    k: usize,
    phantom: PhantomData<&'a T>,
}

impl<'a, T: RealNumber, M: Matrix<T>> ObjectiveFunction<T, M>
    for MultiClassObjectiveFunction<'a, T, M>
{
    fn f(&self, w_bias: &M) -> T {
        let mut f = T::zero();
        let mut prob = M::zeros(1, self.k);
        let (n, p) = self.x.shape();
        for i in 0..n {
            for j in 0..self.k {
                prob.set(
                    0,
                    j,
                    MultiClassObjectiveFunction::partial_dot(w_bias, self.x, j * (p + 1), i),
                );
            }
            prob.softmax_mut();
            f = f - prob.get(0, self.y[i]).ln();
        }

        f
    }

    fn df(&self, g: &mut M, w: &M) {
        g.copy_from(&M::zeros(1, g.shape().1));

        let mut prob = M::zeros(1, self.k);
        let (n, p) = self.x.shape();

        for i in 0..n {
            for j in 0..self.k {
                prob.set(
                    0,
                    j,
                    MultiClassObjectiveFunction::partial_dot(w, self.x, j * (p + 1), i),
                );
            }

            prob.softmax_mut();

            for j in 0..self.k {
                let yi = (if self.y[i] == j { T::one() } else { T::zero() }) - prob.get(0, j);

                for l in 0..p {
                    let pos = j * (p + 1);
                    g.set(0, pos + l, g.get(0, pos + l) - yi * self.x.get(i, l));
                }
                g.set(0, j * (p + 1) + p, g.get(0, j * (p + 1) + p) - yi);
            }
        }
    }
}

impl<T: RealNumber, M: Matrix<T>> LogisticRegression<T, M> {
    pub fn fit(x: &M, y: &M::RowVector) -> LogisticRegression<T, M> {
        let y_m = M::from_row_vector(y.clone());
        let (x_nrows, num_attributes) = x.shape();
        let (_, y_nrows) = y_m.shape();

        if x_nrows != y_nrows {
            panic!("Number of rows of X doesn't match number of rows of Y");
        }

        let classes = y_m.unique();

        let k = classes.len();

        let mut yi: Vec<usize> = vec![0; y_nrows];

        for i in 0..y_nrows {
            let yc = y_m.get(0, i);
            yi[i] = classes.iter().position(|c| yc == *c).unwrap();
        }

        if k < 2 {
            panic!("Incorrect number of classes: {}", k);
        } else if k == 2 {
            let x0 = M::zeros(1, num_attributes + 1);

            let objective = BinaryObjectiveFunction {
                x: x,
                y: yi,
                phantom: PhantomData,
            };

            let result = LogisticRegression::minimize(x0, objective);

            LogisticRegression {
                weights: result.x,
                classes: classes,
                num_attributes: num_attributes,
                num_classes: k,
            }
        } else {
            let x0 = M::zeros(1, (num_attributes + 1) * k);

            let objective = MultiClassObjectiveFunction {
                x: x,
                y: yi,
                k: k,
                phantom: PhantomData,
            };

            let result = LogisticRegression::minimize(x0, objective);

            let weights = result.x.reshape(k, num_attributes + 1);

            LogisticRegression {
                weights: weights,
                classes: classes,
                num_attributes: num_attributes,
                num_classes: k,
            }
        }
    }

    pub fn predict(&self, x: &M) -> M::RowVector {
        let n = x.shape().0;
        let mut result = M::zeros(1, n);
        if self.num_classes == 2 {
            let (nrows, _) = x.shape();
            let x_and_bias = x.v_stack(&M::ones(nrows, 1));
            let y_hat: Vec<T> = x_and_bias.dot(&self.weights.transpose()).to_raw_vector();
            for i in 0..n {
                result.set(
                    0,
                    i,
                    self.classes[if y_hat[i].sigmoid() > T::half() { 1 } else { 0 }],
                );
            }
        } else {
            let (nrows, _) = x.shape();
            let x_and_bias = x.v_stack(&M::ones(nrows, 1));
            let y_hat = x_and_bias.dot(&self.weights.transpose());
            let class_idxs = y_hat.argmax();
            for i in 0..n {
                result.set(0, i, self.classes[class_idxs[i]]);
            }
        }
        result.to_row_vector()
    }

    pub fn coefficients(&self) -> M {
        self.weights
            .slice(0..self.num_classes, 0..self.num_attributes)
    }

    pub fn intercept(&self) -> M {
        self.weights.slice(
            0..self.num_classes,
            self.num_attributes..self.num_attributes + 1,
        )
    }

    fn minimize(x0: M, objective: impl ObjectiveFunction<T, M>) -> OptimizerResult<T, M> {
        let f = |w: &M| -> T { objective.f(w) };

        let df = |g: &mut M, w: &M| objective.df(g, w);

        let mut ls: Backtracking<T> = Default::default();
        ls.order = FunctionOrder::THIRD;
        let optimizer: LBFGS<T> = Default::default();

        optimizer.optimize(&f, &df, &x0, &ls)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::naive::dense_matrix::*;

    #[test]
    fn multiclass_objective_f() {
        let x = DenseMatrix::from_array(&[
            &[1., -5.],
            &[2., 5.],
            &[3., -2.],
            &[1., 2.],
            &[2., 0.],
            &[6., -5.],
            &[7., 5.],
            &[6., -2.],
            &[7., 2.],
            &[6., 0.],
            &[8., -5.],
            &[9., 5.],
            &[10., -2.],
            &[8., 2.],
            &[9., 0.],
        ]);

        let y = vec![0, 0, 1, 1, 2, 1, 1, 0, 0, 2, 1, 1, 0, 0, 1];

        let objective = MultiClassObjectiveFunction {
            x: &x,
            y: y,
            k: 3,
            phantom: PhantomData,
        };

        let mut g: DenseMatrix<f64> = DenseMatrix::zeros(1, 9);

        objective.df(
            &mut g,
            &DenseMatrix::vector_from_array(&[1., 2., 3., 4., 5., 6., 7., 8., 9.]),
        );
        objective.df(
            &mut g,
            &DenseMatrix::vector_from_array(&[1., 2., 3., 4., 5., 6., 7., 8., 9.]),
        );

        assert!((g.get(0, 0) + 33.000068218163484).abs() < std::f64::EPSILON);

        let f = objective.f(&DenseMatrix::vector_from_array(&[
            1., 2., 3., 4., 5., 6., 7., 8., 9.,
        ]));

        assert!((f - 408.0052230582765).abs() < std::f64::EPSILON);
    }

    #[test]
    fn binary_objective_f() {
        let x = DenseMatrix::from_array(&[
            &[1., -5.],
            &[2., 5.],
            &[3., -2.],
            &[1., 2.],
            &[2., 0.],
            &[6., -5.],
            &[7., 5.],
            &[6., -2.],
            &[7., 2.],
            &[6., 0.],
            &[8., -5.],
            &[9., 5.],
            &[10., -2.],
            &[8., 2.],
            &[9., 0.],
        ]);

        let y = vec![0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1];

        let objective = BinaryObjectiveFunction {
            x: &x,
            y: y,
            phantom: PhantomData,
        };

        let mut g: DenseMatrix<f64> = DenseMatrix::zeros(1, 3);

        objective.df(&mut g, &DenseMatrix::vector_from_array(&[1., 2., 3.]));
        objective.df(&mut g, &DenseMatrix::vector_from_array(&[1., 2., 3.]));

        assert!((g.get(0, 0) - 26.051064349381285).abs() < std::f64::EPSILON);
        assert!((g.get(0, 1) - 10.239000702928523).abs() < std::f64::EPSILON);
        assert!((g.get(0, 2) - 3.869294270156324).abs() < std::f64::EPSILON);

        let f = objective.f(&DenseMatrix::vector_from_array(&[1., 2., 3.]));

        assert!((f - 59.76994756647412).abs() < std::f64::EPSILON);
    }

    #[test]
    fn lr_fit_predict() {
        let x = DenseMatrix::from_array(&[
            &[1., -5.],
            &[2., 5.],
            &[3., -2.],
            &[1., 2.],
            &[2., 0.],
            &[6., -5.],
            &[7., 5.],
            &[6., -2.],
            &[7., 2.],
            &[6., 0.],
            &[8., -5.],
            &[9., 5.],
            &[10., -2.],
            &[8., 2.],
            &[9., 0.],
        ]);
        let y: Vec<f64> = vec![0., 0., 1., 1., 2., 1., 1., 0., 0., 2., 1., 1., 0., 0., 1.];

        let lr = LogisticRegression::fit(&x, &y);

        assert_eq!(lr.coefficients().shape(), (3, 2));
        assert_eq!(lr.intercept().shape(), (3, 1));

        assert!((lr.coefficients().get(0, 0) - 0.0435).abs() < 1e-4);
        assert!((lr.intercept().get(0, 0) - 0.1250).abs() < 1e-4);

        let y_hat = lr.predict(&x);

        assert_eq!(
            y_hat,
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        );
    }

    #[test]
    fn serde() {
        let x = DenseMatrix::from_array(&[
            &[1., -5.],
            &[2., 5.],
            &[3., -2.],
            &[1., 2.],
            &[2., 0.],
            &[6., -5.],
            &[7., 5.],
            &[6., -2.],
            &[7., 2.],
            &[6., 0.],
            &[8., -5.],
            &[9., 5.],
            &[10., -2.],
            &[8., 2.],
            &[9., 0.],
        ]);
        let y: Vec<f64> = vec![0., 0., 1., 1., 2., 1., 1., 0., 0., 2., 1., 1., 0., 0., 1.];

        let lr = LogisticRegression::fit(&x, &y);

        let deserialized_lr: LogisticRegression<f64, DenseMatrix<f64>> =
            serde_json::from_str(&serde_json::to_string(&lr).unwrap()).unwrap();

        assert_eq!(lr, deserialized_lr);
    }

    #[test]
    fn lr_fit_predict_iris() {
        let x = DenseMatrix::from_array(&[
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
        let y: Vec<f64> = vec![
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        ];

        let lr = LogisticRegression::fit(&x, &y);

        let y_hat = lr.predict(&x);

        let error: f64 = y
            .into_iter()
            .zip(y_hat.into_iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        assert!(error <= 1.0);
    }
}
