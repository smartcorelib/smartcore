pub mod first_order;
pub mod line_search;

pub type F<'a, X> = dyn for<'b> Fn(&'b X) -> f64 + 'a;
pub type DF<'a, X> = dyn for<'b> Fn(&'b mut X, &'b X) + 'a;

#[derive(Debug, PartialEq)]
pub enum FunctionOrder {
    FIRST,
    SECOND,
    THIRD
}