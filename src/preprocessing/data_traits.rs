//! Traits to indicate that float variables can be viewed as categorical
//! This module assumes 

use crate::math::num::RealNumber;

pub type CategoricalFloat = u16;

// pub struct CategoricalFloat(u16);

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
        a as f32 == self
    }
}

impl Categorizable for f64 {

    type A = CategoricalFloat;

    fn to_category(self) ->CategoricalFloat {
        self as CategoricalFloat
    }

    fn is_valid(self) -> bool {
        let a = self.to_category();
        a as f64 == self
    }
}