#![allow(non_snake_case)]

use std::fmt::Debug;
use std::marker::PhantomData;

use crate::linalg::BaseMatrix;
use crate::math::num::RealNumber;

#[derive(Debug, Clone)]
pub struct LU<T: RealNumber, M: BaseMatrix<T>> {
    LU: M,
    pivot: Vec<usize>,
    pivot_sign: i8,
    singular: bool,
    phantom: PhantomData<T>,
}

impl<T: RealNumber, M: BaseMatrix<T>> LU<T, M> {
    pub fn new(LU: M, pivot: Vec<usize>, pivot_sign: i8) -> LU<T, M> {
        let (_, n) = LU.shape();

        let mut singular = false;
        for j in 0..n {
            if LU.get(j, j) == T::zero() {
                singular = true;
                break;
            }
        }

        LU {
            LU: LU,
            pivot: pivot,
            pivot_sign: pivot_sign,
            singular: singular,
            phantom: PhantomData,
        }
    }

    pub fn L(&self) -> M {
        let (n_rows, n_cols) = self.LU.shape();
        let mut L = M::zeros(n_rows, n_cols);

        for i in 0..n_rows {
            for j in 0..n_cols {
                if i > j {
                    L.set(i, j, self.LU.get(i, j));
                } else if i == j {
                    L.set(i, j, T::one());
                } else {
                    L.set(i, j, T::zero());
                }
            }
        }

        L
    }

    pub fn U(&self) -> M {
        let (n_rows, n_cols) = self.LU.shape();
        let mut U = M::zeros(n_rows, n_cols);

        for i in 0..n_rows {
            for j in 0..n_cols {
                if i <= j {
                    U.set(i, j, self.LU.get(i, j));
                } else {
                    U.set(i, j, T::zero());
                }
            }
        }

        U
    }

    pub fn pivot(&self) -> M {
        let (_, n) = self.LU.shape();
        let mut piv = M::zeros(n, n);

        for i in 0..n {
            piv.set(i, self.pivot[i], T::one());
        }

        piv
    }

    pub fn inverse(&self) -> M {
        let (m, n) = self.LU.shape();

        if m != n {
            panic!("Matrix is not square: {}x{}", m, n);
        }

        let mut inv = M::zeros(n, n);

        for i in 0..n {
            inv.set(i, i, T::one());
        }

        inv = self.solve(inv);
        return inv;
    }

    fn solve(&self, mut b: M) -> M {
        let (m, n) = self.LU.shape();
        let (b_m, b_n) = b.shape();

        if b_m != m {
            panic!(
                "Row dimensions do not agree: A is {} x {}, but B is {} x {}",
                m, n, b_m, b_n
            );
        }

        if self.singular {
            panic!("Matrix is singular.");
        }

        let mut X = M::zeros(b_m, b_n);

        for j in 0..b_n {
            for i in 0..m {
                X.set(i, j, b.get(self.pivot[i], j));
            }
        }

        for k in 0..n {
            for i in k + 1..n {
                for j in 0..b_n {
                    X.sub_element_mut(i, j, X.get(k, j) * self.LU.get(i, k));
                }
            }
        }

        for k in (0..n).rev() {
            for j in 0..b_n {
                X.div_element_mut(k, j, self.LU.get(k, k));
            }

            for i in 0..k {
                for j in 0..b_n {
                    X.sub_element_mut(i, j, X.get(k, j) * self.LU.get(i, k));
                }
            }
        }

        for j in 0..b_n {
            for i in 0..m {
                b.set(i, j, X.get(i, j));
            }
        }

        b
    }
}

pub trait LUDecomposableMatrix<T: RealNumber>: BaseMatrix<T> {
    fn lu(&self) -> LU<T, Self> {
        self.clone().lu_mut()
    }

    fn lu_mut(mut self) -> LU<T, Self> {
        let (m, n) = self.shape();

        let mut piv = vec![0; m];
        for i in 0..m {
            piv[i] = i;
        }

        let mut pivsign = 1;
        let mut LUcolj = vec![T::zero(); m];

        for j in 0..n {
            for i in 0..m {
                LUcolj[i] = self.get(i, j);
            }

            for i in 0..m {
                let kmax = usize::min(i, j);
                let mut s = T::zero();
                for k in 0..kmax {
                    s = s + self.get(i, k) * LUcolj[k];
                }

                LUcolj[i] = LUcolj[i] - s;
                self.set(i, j, LUcolj[i]);
            }

            let mut p = j;
            for i in j + 1..m {
                if LUcolj[i].abs() > LUcolj[p].abs() {
                    p = i;
                }
            }
            if p != j {
                for k in 0..n {
                    let t = self.get(p, k);
                    self.set(p, k, self.get(j, k));
                    self.set(j, k, t);
                }
                let k = piv[p];
                piv[p] = piv[j];
                piv[j] = k;
                pivsign = -pivsign;
            }

            if j < m && self.get(j, j) != T::zero() {
                for i in j + 1..m {
                    self.div_element_mut(i, j, self.get(j, j));
                }
            }
        }

        LU::new(self, piv, pivsign)
    }

    fn lu_solve_mut(self, b: Self) -> Self {
        self.lu_mut().solve(b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::naive::dense_matrix::*;

    #[test]
    fn decompose() {
        let a = DenseMatrix::from_array(&[&[1., 2., 3.], &[0., 1., 5.], &[5., 6., 0.]]);
        let expected_L = DenseMatrix::from_array(&[&[1., 0., 0.], &[0., 1., 0.], &[0.2, 0.8, 1.]]);
        let expected_U = DenseMatrix::from_array(&[&[5., 6., 0.], &[0., 1., 5.], &[0., 0., -1.]]);
        let expected_pivot =
            DenseMatrix::from_array(&[&[0., 0., 1.], &[0., 1., 0.], &[1., 0., 0.]]);
        let lu = a.lu();
        assert!(lu.L().approximate_eq(&expected_L, 1e-4));
        assert!(lu.U().approximate_eq(&expected_U, 1e-4));
        assert!(lu.pivot().approximate_eq(&expected_pivot, 1e-4));
    }

    #[test]
    fn inverse() {
        let a = DenseMatrix::from_array(&[&[1., 2., 3.], &[0., 1., 5.], &[5., 6., 0.]]);
        let expected =
            DenseMatrix::from_array(&[&[-6.0, 3.6, 1.4], &[5.0, -3.0, -1.0], &[-1.0, 0.8, 0.2]]);
        let a_inv = a.lu().inverse();
        println!("{}", a_inv);
        assert!(a_inv.approximate_eq(&expected, 1e-4));
    }
}
