use crate::math::distance::Distance;
use ndarray::{ArrayBase, Data, Dimension};
use crate::common::AnyNumber;

impl<A, S1, S2, D> Distance<ArrayBase<S2, D>> for ArrayBase<S1, D>
where
        A: AnyNumber,
        S1: Data<Elem = A>,        
        S2: Data<Elem = A>,        
        D: Dimension
{
    fn distance_to(&self, other: &Self) -> f64 
    {
        Self::distance(self, other)
    }

    fn distance(a: &Self, b: &ArrayBase<S2, D>) -> f64 
    {
        if a.len() != b.len() {
            panic!("vectors a and b have different length");
        } else {     
            ((a - b)*(a - b)).sum().to_f64().unwrap().sqrt()            
        }
    }


}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, ArrayView1, arr1};    

    #[test]
    fn measure_simple_euclidian_distance() {
        let a = arr1(&[1, 2, 3]);
        let b = arr1(&[4, 5, 6]);             
        
        let d_arr = Array1::distance(&a, &b);
        let d_view = ArrayView1::distance(&a.view(), &b.view());

        assert!((d_arr - 5.19615242).abs() < 1e-8);
        assert!((d_view - 5.19615242).abs() < 1e-8);
    }    

    #[test]
    fn measure_simple_euclidian_distance_static() {
        let a = arr1(&[-2.1968219, -0.9559913, -0.0431738,  1.0567679,  0.3853515]);
        let b = arr1(&[-1.7781325, -0.6659839,  0.9526148, -0.9460919, -0.3925300]);    

        let d = Array1::distance(&a, &b);

        assert!((d - 2.422302).abs() < 1e-6);
    }
}