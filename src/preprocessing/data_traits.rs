//! Traits to indicate that float variables can be viewed as categorical
//! This module assumes

use crate::math::num::RealNumber;

pub type CategoricalFloat = u16;

// pub struct CategoricalFloat(u16);
const ERROR_MARGIN: f64 = 0.001;

pub trait Categorizable: RealNumber {
    type A;

    fn to_category(self) -> CategoricalFloat;

    fn is_valid(self) -> bool;
}

impl Categorizable for f32 {
    type A = CategoricalFloat;

    fn to_category(self) -> CategoricalFloat {
        self as CategoricalFloat
    }

    fn is_valid(self) -> bool {
        let a = self.to_category();
        (a as f32 - self).abs() < (ERROR_MARGIN as f32)
    }
}

impl Categorizable for f64 {
    type A = CategoricalFloat;

    fn to_category(self) -> CategoricalFloat {
        self as CategoricalFloat
    }

    fn is_valid(self) -> bool {
        let a = self.to_category();
        (a as f64 - self).abs() < ERROR_MARGIN
    }
}
