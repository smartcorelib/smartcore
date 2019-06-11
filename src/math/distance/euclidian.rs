use super::Distance;
use ndarray::{ArrayBase, Data, Dimension};
use num_traits::{Num, ToPrimitive};
use ndarray::{ScalarOperand};

pub struct EuclidianDistance{}

impl<A, S, D> Distance<ArrayBase<S, D>> for EuclidianDistance
where
    A: Num + ScalarOperand + ToPrimitive,      
    S: Data<Elem = A>,
    D: Dimension
{

    fn distance(a: &ArrayBase<S, D>, b: &ArrayBase<S, D>) -> f64 {
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
    use ndarray::{arr1, Array};    

    #[test]
    fn measure_simple_euclidian_distance() {
        let a = arr1(&[1, 2, 3]);
        let b = arr1(&[4, 5, 6]);        

        let d = EuclidianDistance::distance(&a, &b);

        assert!((d - 5.19615242).abs() < 1e-8);
    }

    #[test]
    fn measure_simple_euclidian_distance_static() {
        let a = arr1(&[-2.1968219, -0.9559913, -0.0431738,  1.0567679,  0.3853515]);
        let b = arr1(&[-1.7781325, -0.6659839,  0.9526148, -0.9460919, -0.3925300]);    

        let d = EuclidianDistance::distance(&a, &b);

        assert!((d - 2.422302).abs() < 1e-6);
    }
}