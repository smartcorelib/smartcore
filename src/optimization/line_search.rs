use crate::optimization::FunctionOrder;
use num_traits::Float;

pub trait LineSearchMethod<T: Float> {
    fn search(
        &self,
        f: &(dyn Fn(T) -> T),
        df: &(dyn Fn(T) -> T),
        alpha: T,
        f0: T,
        df0: T,
    ) -> LineSearchResult<T>;
}

#[derive(Debug, Clone)]
pub struct LineSearchResult<T: Float> {
    pub alpha: T,
    pub f_x: T,
}

pub struct Backtracking<T: Float> {
    pub c1: T,
    pub max_iterations: usize,
    pub max_infinity_iterations: usize,
    pub phi: T,
    pub plo: T,
    pub order: FunctionOrder,
}

impl<T: Float> Default for Backtracking<T> {
    fn default() -> Self {
        Backtracking {
            c1: T::from(1e-4).unwrap(),
            max_iterations: 1000,
            max_infinity_iterations: (-T::epsilon().log2()).to_usize().unwrap(),
            phi: T::from(0.5).unwrap(),
            plo: T::from(0.1).unwrap(),
            order: FunctionOrder::SECOND,
        }
    }
}

impl<T: Float> LineSearchMethod<T> for Backtracking<T> {
    fn search(
        &self,
        f: &(dyn Fn(T) -> T),
        _: &(dyn Fn(T) -> T),
        alpha: T,
        f0: T,
        df0: T,
    ) -> LineSearchResult<T> {
        let two = T::from(2.).unwrap();
        let three = T::from(3.).unwrap();

        let (mut a1, mut a2) = (alpha, alpha);
        let (mut fx0, mut fx1) = (f0, f(a1));

        let mut iterfinite = 0;
        while !fx1.is_finite() && iterfinite < self.max_infinity_iterations {
            iterfinite += 1;
            a1 = a2;
            a2 = a1 / two;

            fx1 = f(a2);
        }

        let mut iteration = 0;

        while fx1 > f0 + self.c1 * a2 * df0 {
            if iteration > self.max_iterations {
                panic!("Linesearch failed to converge, reached maximum iterations.");
            }

            let a_tmp;

            if self.order == FunctionOrder::SECOND || iteration == 0 {
                a_tmp = -(df0 * a2.powf(two)) / (two * (fx1 - f0 - df0 * a2))
            } else {
                let div = T::one() / (a1.powf(two) * a2.powf(two) * (a2 - a1));
                let a = (a1.powf(two) * (fx1 - f0 - df0 * a2)
                    - a2.powf(two) * (fx0 - f0 - df0 * a1))
                    * div;
                let b = (-a1.powf(three) * (fx1 - f0 - df0 * a2)
                    + a2.powf(three) * (fx0 - f0 - df0 * a1))
                    * div;

                if (a - T::zero()).powf(two).sqrt() <= T::epsilon() {
                    a_tmp = df0 / (two * b);
                } else {
                    let d = T::max(b.powf(two) - three * a * df0, T::zero());
                    a_tmp = (-b + d.sqrt()) / (three * a); //root of quadratic equation
                }
            }

            a1 = a2;
            a2 = T::max(T::min(a_tmp, a2 * self.phi), a2 * self.plo);

            fx0 = fx1;
            fx1 = f(a2);

            iteration += 1;
        }

        LineSearchResult {
            alpha: a2,
            f_x: fx1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backtracking() {
        let f = |x: f64| -> f64 { x.powf(2.) + x };

        let df = |x: f64| -> f64 { 2. * x + 1. };

        let ls: Backtracking<f64> = Default::default();

        let mut x = -3.;
        let mut alpha = 1.;

        for _ in 0..10 {
            let result = ls.search(&f, &df, alpha, f(x), df(x));
            alpha = result.alpha;
            x += alpha;
        }

        assert!(f(x).abs() < 0.01);
    }
}
