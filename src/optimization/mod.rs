pub mod first_order;
pub mod line_search;

use crate::linalg::Vector;

type F<X: Vector> = dyn Fn(&X) -> f64;
type DF<X: Vector> = dyn Fn(&mut X, &X);

#[derive(Debug)]
pub enum FunctionOrder {
    FIRST,
    SECOND,
    THIRD
}