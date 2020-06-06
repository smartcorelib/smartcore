#![allow(non_snake_case)]

use crate::linalg::BaseMatrix;
use crate::math::num::FloatExt;
use std::fmt::Debug;

#[derive(Debug, Clone)]
pub struct SVD<T: FloatExt, M: SVDDecomposableMatrix<T>> {
    pub U: M,
    pub V: M,
    pub s: Vec<T>,
    full: bool,
    m: usize,
    n: usize,
    tol: T,
}

pub trait SVDDecomposableMatrix<T: FloatExt>: BaseMatrix<T> {
    fn svd_solve_mut(self, b: Self) -> Self {
        self.svd_mut().solve(b)
    }

    fn svd_solve(&self, b: Self) -> Self {
        self.svd().solve(b)
    }

    fn svd(&self) -> SVD<T, Self> {
        self.clone().svd_mut()
    }

    fn svd_mut(self) -> SVD<T, Self> {
        let mut U = self;

        let (m, n) = U.shape();

        let (mut l, mut nm) = (0usize, 0usize);
        let (mut anorm, mut g, mut scale) = (T::zero(), T::zero(), T::zero());

        let mut v = Self::zeros(n, n);
        let mut w = vec![T::zero(); n];
        let mut rv1 = vec![T::zero(); n];

        for i in 0..n {
            l = i + 2;
            rv1[i] = scale * g;
            g = T::zero();
            let mut s = T::zero();
            scale = T::zero();

            if i < m {
                for k in i..m {
                    scale = scale + U.get(k, i).abs();
                }

                if scale.abs() > T::epsilon() {
                    for k in i..m {
                        U.div_element_mut(k, i, scale);
                        s = s + U.get(k, i) * U.get(k, i);
                    }

                    let mut f = U.get(i, i);
                    g = -s.sqrt().copysign(f);
                    let h = f * g - s;
                    U.set(i, i, f - g);
                    for j in l - 1..n {
                        s = T::zero();
                        for k in i..m {
                            s = s + U.get(k, i) * U.get(k, j);
                        }
                        f = s / h;
                        for k in i..m {
                            U.add_element_mut(k, j, f * U.get(k, i));
                        }
                    }
                    for k in i..m {
                        U.mul_element_mut(k, i, scale);
                    }
                }
            }

            w[i] = scale * g;
            g = T::zero();
            let mut s = T::zero();
            scale = T::zero();

            if i + 1 <= m && i + 1 != n {
                for k in l - 1..n {
                    scale = scale + U.get(i, k).abs();
                }

                if scale.abs() > T::epsilon() {
                    for k in l - 1..n {
                        U.div_element_mut(i, k, scale);
                        s = s + U.get(i, k) * U.get(i, k);
                    }

                    let f = U.get(i, l - 1);
                    g = -s.sqrt().copysign(f);
                    let h = f * g - s;
                    U.set(i, l - 1, f - g);

                    for k in l - 1..n {
                        rv1[k] = U.get(i, k) / h;
                    }

                    for j in l - 1..m {
                        s = T::zero();
                        for k in l - 1..n {
                            s = s + U.get(j, k) * U.get(i, k);
                        }

                        for k in l - 1..n {
                            U.add_element_mut(j, k, s * rv1[k]);
                        }
                    }

                    for k in l - 1..n {
                        U.mul_element_mut(i, k, scale);
                    }
                }
            }

            anorm = T::max(anorm, w[i].abs() + rv1[i].abs());
        }

        for i in (0..n).rev() {
            if i < n - 1 {
                if g != T::zero() {
                    for j in l..n {
                        v.set(j, i, (U.get(i, j) / U.get(i, l)) / g);
                    }
                    for j in l..n {
                        let mut s = T::zero();
                        for k in l..n {
                            s = s + U.get(i, k) * v.get(k, j);
                        }
                        for k in l..n {
                            v.add_element_mut(k, j, s * v.get(k, i));
                        }
                    }
                }
                for j in l..n {
                    v.set(i, j, T::zero());
                    v.set(j, i, T::zero());
                }
            }
            v.set(i, i, T::one());
            g = rv1[i];
            l = i;
        }

        for i in (0..usize::min(m, n)).rev() {
            l = i + 1;
            g = w[i];
            for j in l..n {
                U.set(i, j, T::zero());
            }

            if g.abs() > T::epsilon() {
                g = T::one() / g;
                for j in l..n {
                    let mut s = T::zero();
                    for k in l..m {
                        s = s + U.get(k, i) * U.get(k, j);
                    }
                    let f = (s / U.get(i, i)) * g;
                    for k in i..m {
                        U.add_element_mut(k, j, f * U.get(k, i));
                    }
                }
                for j in i..m {
                    U.mul_element_mut(j, i, g);
                }
            } else {
                for j in i..m {
                    U.set(j, i, T::zero());
                }
            }

            U.add_element_mut(i, i, T::one());
        }

        for k in (0..n).rev() {
            for iteration in 0..30 {
                let mut flag = true;
                l = k;
                while l != 0 {
                    if l == 0 || rv1[l].abs() <= T::epsilon() * anorm {
                        flag = false;
                        break;
                    }
                    nm = l - 1;
                    if w[nm].abs() <= T::epsilon() * anorm {
                        break;
                    }
                    l -= 1;
                }

                if flag {
                    let mut c = T::zero();
                    let mut s = T::one();
                    for i in l..k + 1 {
                        let f = s * rv1[i];
                        rv1[i] = c * rv1[i];
                        if f.abs() <= T::epsilon() * anorm {
                            break;
                        }
                        g = w[i];
                        let mut h = f.hypot(g);
                        w[i] = h;
                        h = T::one() / h;
                        c = g * h;
                        s = -f * h;
                        for j in 0..m {
                            let y = U.get(j, nm);
                            let z = U.get(j, i);
                            U.set(j, nm, y * c + z * s);
                            U.set(j, i, z * c - y * s);
                        }
                    }
                }

                let z = w[k];
                if l == k {
                    if z < T::zero() {
                        w[k] = -z;
                        for j in 0..n {
                            v.set(j, k, -v.get(j, k));
                        }
                    }
                    break;
                }

                if iteration == 29 {
                    panic!("no convergence in 30 iterations");
                }

                let mut x = w[l];
                nm = k - 1;
                let mut y = w[nm];
                g = rv1[nm];
                let mut h = rv1[k];
                let mut f = ((y - z) * (y + z) + (g - h) * (g + h)) / (T::two() * h * y);
                g = f.hypot(T::one());
                f = ((x - z) * (x + z) + h * ((y / (f + g.copysign(f))) - h)) / x;
                let mut c = T::one();
                let mut s = T::one();

                for j in l..=nm {
                    let i = j + 1;
                    g = rv1[i];
                    y = w[i];
                    h = s * g;
                    g = c * g;
                    let mut z = f.hypot(h);
                    rv1[j] = z;
                    c = f / z;
                    s = h / z;
                    f = x * c + g * s;
                    g = g * c - x * s;
                    h = y * s;
                    y = y * c;

                    for jj in 0..n {
                        x = v.get(jj, j);
                        z = v.get(jj, i);
                        v.set(jj, j, x * c + z * s);
                        v.set(jj, i, z * c - x * s);
                    }

                    z = f.hypot(h);
                    w[j] = z;
                    if z.abs() > T::epsilon() {
                        z = T::one() / z;
                        c = f * z;
                        s = h * z;
                    }

                    f = c * g + s * y;
                    x = c * y - s * g;
                    for jj in 0..m {
                        y = U.get(jj, j);
                        z = U.get(jj, i);
                        U.set(jj, j, y * c + z * s);
                        U.set(jj, i, z * c - y * s);
                    }
                }

                rv1[l] = T::zero();
                rv1[k] = f;
                w[k] = x;
            }
        }

        let mut inc = 1usize;
        let mut su = vec![T::zero(); m];
        let mut sv = vec![T::zero(); n];

        loop {
            inc *= 3;
            inc += 1;
            if inc > n {
                break;
            }
        }

        loop {
            inc /= 3;
            for i in inc..n {
                let sw = w[i];
                for k in 0..m {
                    su[k] = U.get(k, i);
                }
                for k in 0..n {
                    sv[k] = v.get(k, i);
                }
                let mut j = i;
                while w[j - inc] < sw {
                    w[j] = w[j - inc];
                    for k in 0..m {
                        U.set(k, j, U.get(k, j - inc));
                    }
                    for k in 0..n {
                        v.set(k, j, v.get(k, j - inc));
                    }
                    j -= inc;
                    if j < inc {
                        break;
                    }
                }
                w[j] = sw;
                for k in 0..m {
                    U.set(k, j, su[k]);
                }
                for k in 0..n {
                    v.set(k, j, sv[k]);
                }
            }
            if inc <= 1 {
                break;
            }
        }

        for k in 0..n {
            let mut s = 0.;
            for i in 0..m {
                if U.get(i, k) < T::zero() {
                    s += 1.;
                }
            }
            for j in 0..n {
                if v.get(j, k) < T::zero() {
                    s += 1.;
                }
            }
            if s > (m + n) as f64 / 2. {
                for i in 0..m {
                    U.set(i, k, -U.get(i, k));
                }
                for j in 0..n {
                    v.set(j, k, -v.get(j, k));
                }
            }
        }

        SVD::new(U, v, w)
    }
}

impl<T: FloatExt, M: SVDDecomposableMatrix<T>> SVD<T, M> {
    pub fn new(U: M, V: M, s: Vec<T>) -> SVD<T, M> {
        let m = U.shape().0;
        let n = V.shape().0;
        let full = s.len() == m.min(n);
        let tol = T::half() * (T::from(m + n).unwrap() + T::one()).sqrt() * s[0] * T::epsilon();
        SVD {
            U: U,
            V: V,
            s: s,
            full: full,
            m: m,
            n: n,
            tol: tol,
        }
    }

    pub fn solve(&self, mut b: M) -> M {
        let p = b.shape().1;

        if self.U.shape().0 != b.shape().0 {
            panic!(
                "Dimensions do not agree. U.nrows should equal b.nrows but is {}, {}",
                self.U.shape().0,
                b.shape().0
            );
        }

        for k in 0..p {
            let mut tmp = vec![T::zero(); self.n];
            for j in 0..self.n {
                let mut r = T::zero();
                if self.s[j] > self.tol {
                    for i in 0..self.m {
                        r = r + self.U.get(i, j) * b.get(i, k);
                    }
                    r = r / self.s[j];
                }
                tmp[j] = r;
            }

            for j in 0..self.n {
                let mut r = T::zero();
                for jj in 0..self.n {
                    r = r + self.V.get(j, jj) * tmp[jj];
                }
                b.set(j, k, r);
            }
        }

        b
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::naive::dense_matrix::DenseMatrix;

    #[test]
    fn decompose_symmetric() {
        let A = DenseMatrix::from_array(&[
            &[0.9000, 0.4000, 0.7000],
            &[0.4000, 0.5000, 0.3000],
            &[0.7000, 0.3000, 0.8000],
        ]);

        let s: Vec<f64> = vec![1.7498382, 0.3165784, 0.1335834];

        let U = DenseMatrix::from_array(&[
            &[0.6881997, -0.07121225, 0.7220180],
            &[0.3700456, 0.89044952, -0.2648886],
            &[0.6240573, -0.44947578, -0.639158],
        ]);

        let V = DenseMatrix::from_array(&[
            &[0.6881997, -0.07121225, 0.7220180],
            &[0.3700456, 0.89044952, -0.2648886],
            &[0.6240573, -0.44947578, -0.6391588],
        ]);

        let svd = A.svd();

        assert!(V.abs().approximate_eq(&svd.V.abs(), 1e-4));
        assert!(U.abs().approximate_eq(&svd.U.abs(), 1e-4));
        for i in 0..s.len() {
            assert!((s[i] - svd.s[i]).abs() < 1e-4);
        }
    }

    #[test]
    fn decompose_asymmetric() {
        let A = DenseMatrix::from_array(&[
            &[
                1.19720880,
                -1.8391378,
                0.3019585,
                -1.1165701,
                -1.7210814,
                0.4918882,
                -0.04247433,
            ],
            &[
                0.06605075,
                1.0315583,
                0.8294362,
                -0.3646043,
                -1.6038017,
                -0.9188110,
                -0.63760340,
            ],
            &[
                -1.02637715,
                1.0747931,
                -0.8089055,
                -0.4726863,
                -0.2064826,
                -0.3325532,
                0.17966051,
            ],
            &[
                -1.45817729,
                -0.8942353,
                0.3459245,
                1.5068363,
                -2.0180708,
                -0.3696350,
                -1.19575563,
            ],
            &[
                -0.07318103,
                -0.2783787,
                1.2237598,
                0.1995332,
                0.2545336,
                -0.1392502,
                -1.88207227,
            ],
            &[
                0.88248425, -0.9360321, 0.1393172, 0.1393281, -0.3277873, -0.5553013, 1.63805985,
            ],
            &[
                0.12641406,
                -0.8710055,
                -0.2712301,
                0.2296515,
                1.1781535,
                -0.2158704,
                -0.27529472,
            ],
        ]);

        let s: Vec<f64> = vec![
            3.8589375, 3.4396766, 2.6487176, 2.2317399, 1.5165054, 0.8109055, 0.2706515,
        ];

        let U = DenseMatrix::from_array(&[
            &[
                -0.3082776,
                0.77676231,
                0.01330514,
                0.23231424,
                -0.47682758,
                0.13927109,
                0.02640713,
            ],
            &[
                -0.4013477,
                -0.09112050,
                0.48754440,
                0.47371793,
                0.40636608,
                0.24600706,
                -0.37796295,
            ],
            &[
                0.0599719,
                -0.31406586,
                0.45428229,
                -0.08071283,
                -0.38432597,
                0.57320261,
                0.45673993,
            ],
            &[
                -0.7694214,
                -0.12681435,
                -0.05536793,
                -0.62189972,
                -0.02075522,
                -0.01724911,
                -0.03681864,
            ],
            &[
                -0.3319069,
                -0.17984404,
                -0.54466777,
                0.45335157,
                0.19377726,
                0.12333423,
                0.55003852,
            ],
            &[
                0.1259351,
                0.49087824,
                0.16349687,
                -0.32080176,
                0.64828744,
                0.20643772,
                0.38812467,
            ],
            &[
                0.1491884,
                0.01768604,
                -0.47884363,
                -0.14108924,
                0.03922507,
                0.73034065,
                -0.43965505,
            ],
        ]);

        let V = DenseMatrix::from_array(&[
            &[
                -0.2122609,
                -0.54650056,
                0.08071332,
                -0.43239135,
                -0.2925067,
                0.1414550,
                0.59769207,
            ],
            &[
                -0.1943605,
                0.63132116,
                -0.54059857,
                -0.37089970,
                -0.1363031,
                0.2892641,
                0.17774114,
            ],
            &[
                0.3031265,
                -0.06182488,
                0.18579097,
                -0.38606409,
                -0.5364911,
                0.2983466,
                -0.58642548,
            ],
            &[
                0.1844063, 0.24425278, 0.25923756, 0.59043765, -0.4435443, 0.3959057, 0.37019098,
            ],
            &[
                -0.7164205,
                0.30694911,
                0.58264743,
                -0.07458095,
                -0.1142140,
                -0.1311972,
                -0.13124764,
            ],
            &[
                -0.1103067,
                -0.10633600,
                0.18257905,
                -0.03638501,
                0.5722925,
                0.7784398,
                -0.09153611,
            ],
            &[
                -0.5156083,
                -0.36573746,
                -0.47613340,
                0.41342817,
                -0.2659765,
                0.1654796,
                -0.32346758,
            ],
        ]);

        let svd = A.svd();

        assert!(V.abs().approximate_eq(&svd.V.abs(), 1e-4));
        assert!(U.abs().approximate_eq(&svd.U.abs(), 1e-4));
        for i in 0..s.len() {
            assert!((s[i] - svd.s[i]).abs() < 1e-4);
        }
    }

    #[test]
    fn solve() {
        let a = DenseMatrix::from_array(&[&[0.9, 0.4, 0.7], &[0.4, 0.5, 0.3], &[0.7, 0.3, 0.8]]);
        let b = DenseMatrix::from_array(&[&[0.5, 0.2], &[0.5, 0.8], &[0.5, 0.3]]);
        let expected_w = DenseMatrix::from_array(&[&[-0.20, -1.28], &[0.87, 2.22], &[0.47, 0.66]]);
        let w = a.svd_solve_mut(b);
        assert!(w.approximate_eq(&expected_w, 1e-2));
    }
}
