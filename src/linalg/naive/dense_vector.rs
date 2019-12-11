use crate::linalg::{Vector, Matrix};
use crate::math;
use crate::linalg::naive::dense_matrix::DenseMatrix;

#[derive(Debug, Clone)]
pub struct DenseVector {
    
    size: usize,
    values: Vec<f64> 

}

impl Into<Vec<f64>> for DenseVector {
    fn into(self) -> Vec<f64> {
        self.values
    }
}

impl PartialEq for DenseVector {
    fn eq(&self, other: &Self) -> bool {
        if self.size != other.size {
            return false
        }

        let len = self.values.len();
        let other_len = other.values.len();

        if len != other_len {
            return false;
        }

        for i in 0..len {
            if (self.values[i] - other.values[i]).abs() > math::EPSILON {
                return false;
            }
        }

        true
    }
}

impl Vector for DenseVector {

    fn from_array(values: &[f64]) -> Self {
        DenseVector::from_vec(&Vec::from(values)) 
     }
 
    fn from_vec(values: &Vec<f64>) -> Self {
        DenseVector {
            size: values.len(),
            values: values.clone()
        }
    }

    fn get(&self, i: usize) -> f64 {
        self.values[i]
    }

    fn set(&mut self, i: usize, value: f64) {
        self.values[i] = value;
    }

    fn zeros(size: usize) -> Self {
        DenseVector::fill(size, 0f64)
    }

    fn ones(size: usize) -> Self {
        DenseVector::fill(size, 1f64)
    }

    fn fill(size: usize, value: f64) -> Self {
        DenseVector::from_vec(&vec![value; size])
    }

    fn shape(&self) -> (usize, usize) {
        (1, self.size)
    }

    fn add_mut(&mut self, other: &Self) -> &Self {
        if self.size != other.size {
            panic!("A and B should have the same shape");
        }        
        for i in 0..self.size {
            self.values[i] += other.values[i];           
        }

        self
    }

    fn mul_mut(&mut self, other: &Self) -> &Self {
        if self.size != other.size {
            panic!("A and B should have the same shape");
        }        
        for i in 0..self.size {
            self.values[i] *= other.values[i];           
        }

        self
    }

    fn sub_mut(&mut self, other: &Self) -> &Self {
        if self.size != other.size {
            panic!("A and B should have the same shape");
        }        
        for i in 0..self.size {
            self.values[i] -= other.values[i];           
        }

        self
    }

    fn div_mut(&mut self, other: &Self) -> &Self {
        if self.size != other.size {
            panic!("A and B should have the same shape");
        }        
        for i in 0..self.size {
            self.values[i] /= other.values[i];           
        }

        self
    }

    fn dot(&self, other: &Self) -> f64 {
        if self.size != other.size {
            panic!("A and B should be of the same size");
        }        

        let mut result = 0f64;
        for i in 0..self.size {
            result += self.get(i) * other.get(i);            
        }

        result
    }   

    fn norm2(&self) -> f64 {
        let mut norm = 0f64;

        for xi in self.values.iter() {
            norm += xi * xi;
        }

        norm.sqrt()
    }

    fn norm(&self, p:f64) -> f64 {

        if p.is_infinite() && p.is_sign_positive() {
            self.values.iter().map(|x| x.abs()).fold(std::f64::NEG_INFINITY, |a, b| a.max(b))
        } else if p.is_infinite() && p.is_sign_negative() {
            self.values.iter().map(|x| x.abs()).fold(std::f64::INFINITY, |a, b| a.min(b))
        } else {

            let mut norm = 0f64;

            for xi in self.values.iter() {
                norm += xi.abs().powf(p);
            }

            norm.powf(1.0/p)
        }
    }

    fn add_scalar_mut(&mut self, scalar: f64) -> &Self {
        for i in 0..self.values.len() {
            self.values[i] += scalar;
        }
        self
    }

    fn sub_scalar_mut(&mut self, scalar: f64) -> &Self {
        for i in 0..self.values.len() {
            self.values[i] -= scalar;
        }
        self
    }

    fn mul_scalar_mut(&mut self, scalar: f64) -> &Self {
        for i in 0..self.values.len() {
            self.values[i] *= scalar;
        }
        self
    }

    fn div_scalar_mut(&mut self, scalar: f64) -> &Self {
        for i in 0..self.values.len() {
            self.values[i] /= scalar;
        }
        self
    }

    fn negative_mut(&mut self) -> &Self {
        for i in 0..self.values.len() {
            self.values[i] = -self.values[i];
        }
        self
    }

    fn abs_mut(&mut self) -> &Self{
        for i in 0..self.values.len() {
            self.values[i] = self.values[i].abs();
        }
        self
    }

    fn pow_mut(&mut self, p: f64) -> &Self{
        for i in 0..self.values.len() {
            self.values[i] = self.values[i].powf(p);
        }
        self
    }

    fn sum(&self) -> f64 {
        let mut sum = 0.;
        for i in 0..self.values.len() {
            sum += self.values[i];
        }
        sum
    }

    fn negative(&self) -> Self {
        let mut result = DenseVector {
            size: self.size,
            values: self.values.clone()
        };
        for i in 0..self.values.len() {
            result.values[i] = -self.values[i];
        }
        result
    }

    fn copy_from(&mut self, other: &Self) {
        for i in 0..self.values.len() {
            self.values[i] = other.values[i];
        }
    }

    fn max_diff(&self, other: &Self) -> f64{
        let mut max_diff = 0f64;
        for i in 0..self.values.len() {
            max_diff = max_diff.max((self.values[i] - other.values[i]).abs());
        }
        max_diff

    }

    fn softmax_mut(&mut self) {
        let max = self.values.iter().map(|x| x.abs()).fold(std::f64::NEG_INFINITY, |a, b| a.max(b));
        let mut z = 0.;
        for i in 0..self.size {
            let p = (self.values[i] - max).exp();
            self.values[i] = p;
            z += p;
        }
        for i in 0..self.size {
            self.values[i] /= z;
        }
    }

    fn unique(&self) -> Vec<f64> {
        let mut result = self.values.clone();
        result.sort_by(|a, b| a.partial_cmp(b).unwrap());
        result.dedup();
        result
    }

}

#[cfg(test)]
mod tests {    
    use super::*; 

    #[test]
    fn qr_solve_mut() { 

            let v = DenseVector::from_array(&[3., -2., 6.]);            
            assert_eq!(v.norm(1.), 11.);
            assert_eq!(v.norm(2.), 7.);
            assert_eq!(v.norm(std::f64::INFINITY), 6.);
            assert_eq!(v.norm(std::f64::NEG_INFINITY), 2.);
    }

    #[test]
    fn copy_from() { 

            let mut a = DenseVector::from_array(&[0., 0., 0.]);  
            let b = DenseVector::from_array(&[-1., 0., 2.]);    
            a.copy_from(&b);
            assert_eq!(a.get(0), b.get(0));     
            assert_eq!(a.get(1), b.get(1));     
            assert_eq!(a.get(2), b.get(2));            
    }

    #[test]
    fn softmax_mut() { 

            let mut prob = DenseVector::from_array(&[1., 2., 3.]);  
            prob.softmax_mut();            
            assert!((prob.get(0) - 0.09).abs() < 0.01);     
            assert!((prob.get(1) - 0.24).abs() < 0.01);     
            assert!((prob.get(2) - 0.66).abs() < 0.01);            
    }

}