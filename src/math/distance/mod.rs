pub mod euclidian;

use crate::math::num::FloatExt;

pub trait Distance<T, F: FloatExt>{
    fn distance(a: &T, b: &T) -> F;
}

pub struct Distances{    
}

impl Distances {
    pub fn euclidian() -> euclidian::Euclidian{
        euclidian::Euclidian {}
    }
}