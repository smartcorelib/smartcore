pub mod euclidian;

use num_traits::Float;

pub trait Distance<T>
{
    fn distance(a: &T, b: &T) -> f64;
}