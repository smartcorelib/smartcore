use crate::linalg::Vector;

#[derive(Debug, Clone)]
pub struct DenseVector {
    
    size: usize,
    values: Vec<f64> 

}

impl DenseVector {

    pub fn from_array(values: &[f64]) -> DenseVector {
       DenseVector::from_vec(Vec::from(values)) 
    }

    pub fn from_vec(values: Vec<f64>) -> DenseVector {
        DenseVector {
            size: values.len(),
            values: values
        }
    }
    
}

impl Into<Vec<f64>> for DenseVector {
    fn into(self) -> Vec<f64> {
        self.values
    }
}

impl Vector for DenseVector {

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
        DenseVector::from_vec(vec![value; size])
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

}