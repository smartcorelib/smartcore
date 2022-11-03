//! # Real Number
//! Most algorithms in SmartCore rely on basic linear algebra operations like dot product, matrix decomposition and other subroutines that are defined for a set of real numbers, ℝ.
//! This module defines real number and some useful functions that are used in [Linear Algebra](../../linalg/index.html) module.

use num_traits::Float;

use crate::numbers::basenum::Number;

/// Defines real number
/// <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS_CHTML"></script>
pub trait RealNumber: Number + Float {
    /// Copy sign from `sign` - another real number
    fn copysign(self, sign: Self) -> Self;

    /// Calculates natural \\( \ln(1+e^x) \\) without overflow.
    fn ln_1pe(self) -> Self;

    /// Efficient implementation of Sigmoid function, \\( S(x) = \frac{1}{1 + e^{-x}} \\), see [Sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function)
    fn sigmoid(self) -> Self;

    /// Returns pseudorandom number between 0 and 1
    fn rand() -> Self;

    /// Returns 2
    fn two() -> Self;

    /// Returns .5
    fn half() -> Self;

    /// Returns \\( x^2 \\)
    fn square(self) -> Self {
        self * self
    }

    /// Raw transmutation to u32
    fn to_f32_bits(self) -> u32;

    /// Raw transmutation to u64
    fn to_f64_bits(self) -> u64;
}

impl RealNumber for f64 {
    fn copysign(self, sign: Self) -> Self {
        self.copysign(sign)
    }

    fn ln_1pe(self) -> f64 {
        if self > 15. {
            self
        } else {
            self.exp().ln_1p()
        }
    }

    fn sigmoid(self) -> f64 {
        if self < -40. {
            0.
        } else if self > 40. {
            1.
        } else {
            1. / (1. + f64::exp(-self))
        }
    }

    fn rand() -> f64 {
        // TODO: to be implemented, see issue smartcore#214
        1.0
    }

    fn two() -> Self {
        2f64
    }

    fn half() -> Self {
        0.5f64
    }

    fn to_f32_bits(self) -> u32 {
        self.to_bits() as u32
    }

    fn to_f64_bits(self) -> u64 {
        self.to_bits()
    }
}

impl RealNumber for f32 {
    fn copysign(self, sign: Self) -> Self {
        self.copysign(sign)
    }

    fn ln_1pe(self) -> f32 {
        if self > 15. {
            self
        } else {
            self.exp().ln_1p()
        }
    }

    fn sigmoid(self) -> f32 {
        if self < -40. {
            0.
        } else if self > 40. {
            1.
        } else {
            1. / (1. + f32::exp(-self))
        }
    }

    fn rand() -> f32 {
        1.0
    }

    fn two() -> Self {
        2f32
    }

    fn half() -> Self {
        0.5f32
    }

    fn to_f32_bits(self) -> u32 {
        self.to_bits()
    }

    fn to_f64_bits(self) -> u64 {
        self.to_bits() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn sigmoid() {
        assert_eq!(1.0.sigmoid(), 0.7310585786300049);
        assert_eq!(41.0.sigmoid(), 1.);
        assert_eq!((-41.0).sigmoid(), 0.);
    }

    #[test]
    fn f32_from_string() {
        assert_eq!(f32::from_str("1.111111").unwrap(), 1.111111)
    }

    #[test]
    fn f64_from_string() {
        assert_eq!(f64::from_str("1.111111111").unwrap(), 1.111111111)
    }
}
