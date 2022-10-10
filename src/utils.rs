#[cfg(test)]
pub mod vec_utils {

    use crate::numbers::floatnum::FloatNumber;

    pub fn approx_eq<T: FloatNumber>(a: &[T], b: &[T], tol: T) -> bool {
        a.iter().zip(b.iter()).all(|(&a, &b)| (a - b).abs() < tol)
    }
}
