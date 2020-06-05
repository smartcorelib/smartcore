pub mod euclidian;
pub mod minkowski;
pub mod manhattan;
pub mod hamming;
pub mod mahalanobis;

use crate::math::num::FloatExt;

pub trait Distance<T, F: FloatExt>{
    fn distance(&self, a: &T, b: &T) -> F;
}

pub struct Distances{    
}

impl Distances {
    pub fn euclidian() -> euclidian::Euclidian{
        euclidian::Euclidian {}
    }

    pub fn minkowski<T: FloatExt>(p: T) -> minkowski::Minkowski<T>{
        minkowski::Minkowski {p: p}
    }

    pub fn manhattan() -> manhattan::Manhattan{
        manhattan::Manhattan {}
    }

    pub fn hamming() -> hamming::Hamming{
        hamming::Hamming {}
    }
}