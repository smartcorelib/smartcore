///
pub mod first_order;
///
pub mod line_search;

///
pub type F<'a, T, X> = dyn for<'b> Fn(&'b X) -> T + 'a;
///
pub type DF<'a, X> = dyn for<'b> Fn(&'b mut X, &'b X) + 'a;

/// Function order
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, PartialEq, Eq)]
pub enum FunctionOrder {
    /// Second order
    SECOND,
    /// Third order
    THIRD,
}
