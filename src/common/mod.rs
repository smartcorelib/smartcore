use num_traits::{Num, ToPrimitive, FromPrimitive, Zero, One};
use ndarray::{ScalarOperand};
use std::hash::Hash;
use std::fmt::Debug;

pub trait AnyNumber: Num + ScalarOperand + ToPrimitive + FromPrimitive{}

pub trait Nominal: PartialEq + Zero + One + Eq + Hash + ToPrimitive + FromPrimitive + Debug + 'static + Clone{}


impl<T> AnyNumber for T where T: Num + ScalarOperand + ToPrimitive + FromPrimitive {}

impl<T> Nominal for T where T: PartialEq + Zero + One + Eq + Hash + ToPrimitive + Debug + FromPrimitive + 'static + Clone {}