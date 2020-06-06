use ndarray::ScalarOperand;
use num_traits::{FromPrimitive, Num, One, ToPrimitive, Zero};
use std::fmt::Debug;
use std::hash::Hash;

pub trait AnyNumber: Num + ScalarOperand + ToPrimitive + FromPrimitive {}

pub trait Nominal:
    PartialEq + Zero + One + Eq + Hash + ToPrimitive + FromPrimitive + Debug + 'static + Clone
{
}

impl<T> AnyNumber for T where T: Num + ScalarOperand + ToPrimitive + FromPrimitive {}

impl<T> Nominal for T where
    T: PartialEq + Zero + One + Eq + Hash + ToPrimitive + Debug + FromPrimitive + 'static + Clone
{
}
