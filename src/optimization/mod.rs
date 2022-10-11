// TODO: missing documentation

///
pub mod first_order;
///
pub mod line_search;

///
pub type F<'a, T, X> = dyn for<'b> Fn(&'b X) -> T + 'a;
///
pub type DF<'a, X> = dyn for<'b> Fn(&'b mut X, &'b X) + 'a;

///
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, PartialEq, Eq)]
pub enum FunctionOrder {
    ///
    SECOND,
    ///
    THIRD,
}
