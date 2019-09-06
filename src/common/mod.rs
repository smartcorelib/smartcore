use num_traits::{Num, ToPrimitive, FromPrimitive};
use ndarray::{ScalarOperand};

pub trait AnyNumber: Num + ScalarOperand + ToPrimitive + FromPrimitive{}


impl<T> AnyNumber for T where T: Num + ScalarOperand + ToPrimitive + FromPrimitive {}