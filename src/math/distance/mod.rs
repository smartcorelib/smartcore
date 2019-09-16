pub mod euclidian;

pub trait Distance<T> {

    fn distance_to(&self, other: &Self) -> f64;

    fn distance(a: &Self, b: &T) -> f64;     

}