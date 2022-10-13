use num_traits::{Bounded, FromPrimitive, Num, NumCast, ToPrimitive};
use std::fmt::{Debug, Display};
use std::iter::{Product, Sum};
use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};

/// Define a `Number` set that acquires traits from `num_traits` to make available a base trait  
/// to be used by other usable sets like `FloatNumber`.
pub trait Number:
    Num
    + FromPrimitive
    + ToPrimitive
    + Debug
    + Display
    + Copy
    + Sum
    + Product
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + Bounded
    + NumCast
{
}

impl Number for f64 {}
impl Number for f32 {}
impl Number for i8 {}
impl Number for i16 {}
impl Number for i32 {}
impl Number for i64 {}
impl Number for u8 {}
impl Number for u16 {}
impl Number for u32 {}
impl Number for u64 {}
impl Number for usize {}

/// Integers represented as ordered numbers, mostly used for labels (y)
pub trait IntNumber:
    Number
    + Ord
{}

impl IntNumber for i8 {}
impl IntNumber for i16 {}
impl IntNumber for i32 {}
impl IntNumber for i64 {}
impl IntNumber for u8 {}
impl IntNumber for u16 {}
impl IntNumber for u32 {}
impl IntNumber for u64 {}
impl IntNumber for usize {}


#[cfg(test)]
mod tests {
    use std::str::FromStr;

    #[test]
    fn i32_from_string() {
        assert_eq!(i32::from_str("1").unwrap(), 1)
    }

    #[test]
    fn i8_from_string() {
        assert_eq!(i8::from_str("1").unwrap(), 1)
    }
}