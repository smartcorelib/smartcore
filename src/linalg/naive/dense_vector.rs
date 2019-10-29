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

}