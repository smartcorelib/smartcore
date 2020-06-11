use num_traits::{Float, FromPrimitive};
use rand::prelude::*;
use std::fmt::{Debug, Display};

pub trait FloatExt: Float + FromPrimitive + Debug + Display + Copy {
    fn copysign(self, sign: Self) -> Self;

    fn ln_1pe(self) -> Self;

    fn sigmoid(self) -> Self;

    fn rand() -> Self;

    fn two() -> Self;

    fn half() -> Self;

    fn square(self) -> Self {
        self * self
    }
}

impl FloatExt for f64 {
    fn copysign(self, sign: Self) -> Self {
        self.copysign(sign)
    }

    fn ln_1pe(self) -> f64 {
        if self > 15. {
            return self;
        } else {
            return self.exp().ln_1p();
        }
    }

    fn sigmoid(self) -> f64 {
        if self < -40. {
            return 0.;
        } else if self > 40. {
            return 1.;
        } else {
            return 1. / (1. + f64::exp(-self));
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
}

impl FloatExt for f32 {
    fn copysign(self, sign: Self) -> Self {
        self.copysign(sign)
    }

    fn ln_1pe(self) -> f32 {
        if self > 15. {
            return self;
        } else {
            return self.exp().ln_1p();
        }
    }

    fn sigmoid(self) -> f32 {
        if self < -40. {
            return 0.;
        } else if self > 40. {
            return 1.;
        } else {
            return 1. / (1. + f32::exp(-self));
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
