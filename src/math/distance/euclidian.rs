pub fn distance(x: &Vec<f64>, y: &Vec<f64>) -> f64 {    
    return squared_distance(x, y).sqrt();
}

pub fn squared_distance(x: &Vec<f64>,y: &Vec<f64>) -> f64 {
    if x.len() != y.len() {
        panic!("Input vector sizes are different.");
    }

    let mut sum = 0f64;
    for i in 0..x.len() {
        sum += (x[i] - y[i]).powf(2.);
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
        
        let d_arr = distance(&a, &b);        

        assert!((d_arr - 5.19615242).abs() < 1e-8);        
    }    

}