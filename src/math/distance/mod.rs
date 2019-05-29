pub mod euclidian;

use num_traits::Float;

pub trait Distance<T, A>
where
    A: Float
{
    fn distance(a: &T, b: &T) -> A;
}