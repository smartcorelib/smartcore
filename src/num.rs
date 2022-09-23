//! # Float Number
//! Most algorithms in SmartCore rely on basic linear algebra operations like dot product, matrix decomposition and other subroutines that are defined for a set of float numbers.
//! This module defines number sets and some useful functions that are used in [Linear Algebra](../../linalg/index.html) module.

use num_traits::{Bounded, Float, FromPrimitive, Num, NumCast, Signed, ToPrimitive};
use rand::prelude::*;
use std::fmt::{Debug, Display};
use std::iter::{Product, Sum};
use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};

/// Define a `Number` set that acquires traits from `num_traits` to make available a base trait  
/// to be used by other usable sets like `FloatNumber`.
pub trait Number:
    Num
    + FromPrimitive
    + ToPrimitive
    + Debug
    + Display
    + Copy
    + Sum
    + Product
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + Bounded
    + NumCast
{
}

/// Defines float number
/// <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS_CHTML"></script>
pub trait FloatNumber: Number + Float + Signed {
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

    /// Raw transmutation to u64
    fn to_f32_bits(self) -> u32;
}

impl Number for f64 {}
impl Number for f32 {}
impl Number for i8 {}
impl Number for i16 {}
impl Number for i32 {}
impl Number for i64 {}
impl Number for u8 {}
impl Number for u16 {}
impl Number for u32 {}
impl Number for u64 {}
impl Number for usize {}

impl FloatNumber for f64 {
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
        let mut rng = rand::thread_rng();
        rng.gen()
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
}

impl FloatNumber for f32 {
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
        let mut rng = rand::thread_rng();
        rng.gen()
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid() {
        assert_eq!(1.0.sigmoid(), 0.7310585786300049);
        assert_eq!(41.0.sigmoid(), 1.);
        assert_eq!((-41.0).sigmoid(), 0.);
    }
}
