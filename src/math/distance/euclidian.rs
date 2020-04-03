use serde::{Serialize, Deserialize};

use crate::math::num::FloatExt;

use super::Distance;

#[derive(Serialize, Deserialize, Debug)]
pub struct Euclidian {    
}

impl Euclidian {
    pub fn squared_distance<T: FloatExt>(x: &Vec<T>,y: &Vec<T>) -> T {
        if x.len() != y.len() {
            panic!("Input vector sizes are different.");
        }
    
        let mut sum = T::zero();
        for i in 0..x.len() {
            sum = sum + (x[i] - y[i]).powf(T::two());
        }
    
        sum
    }

    pub fn distance<T: FloatExt>(x: &Vec<T>, y: &Vec<T>) -> T {    
        Euclidian::squared_distance(x, y).sqrt()
    }
    
}

impl<T: FloatExt> Distance<Vec<T>, T> for Euclidian {

    fn distance(x: &Vec<T>, y: &Vec<T>) -> T {    
        Self::distance(x, y)
    } 

}


#[cfg(test)]
mod tests {
    use super::*;    

    #[test]
    fn squared_distance() {
        let a = vec![1., 2., 3.];
        let b = vec![4., 5., 6.];             
        
        let d_arr: f64 = Euclidian::distance(&a, &b);        

        assert!((d_arr - 5.19615242).abs() < 1e-8);        
    }    

}