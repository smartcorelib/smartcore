use crate::math::num::FloatExt;

pub fn distance<T: FloatExt>(x: &Vec<T>, y: &Vec<T>) -> T {    
    return squared_distance(x, y).sqrt();
}

pub fn squared_distance<T: FloatExt>(x: &Vec<T>,y: &Vec<T>) -> T {
    if x.len() != y.len() {
        panic!("Input vector sizes are different.");
    }

    let mut sum = T::zero();
    for i in 0..x.len() {
        sum = sum + (x[i] - y[i]).powf(T::two());
    }

    return sum;
}


#[cfg(test)]
mod tests {
    use super::*;    

    #[test]
    fn squared_distance() {
        let a = vec![1., 2., 3.];
        let b = vec![4., 5., 6.];             
        
        let d_arr: f64 = distance(&a, &b);        

        assert!((d_arr - 5.19615242).abs() < 1e-8);        
    }    

}