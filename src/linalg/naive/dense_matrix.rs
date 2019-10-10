use std::ops::Range;
use crate::linalg::Matrix;
use crate::math;

#[derive(Debug)]
pub struct DenseMatrix {

    ncols: usize,
    nrows: usize,
    values: Vec<f64> 

}

impl DenseMatrix {

    pub fn from_2d_array(values: &[&[f64]]) -> DenseMatrix {
        DenseMatrix::from_2d_vec(&values.into_iter().map(|row| Vec::from(*row)).collect())
    }

    pub fn from_2d_vec(values: &Vec<Vec<f64>>) -> DenseMatrix {
        let nrows = values.len();
        let ncols = values.first().unwrap_or_else(|| panic!("Cannot create 2d matrix from an empty vector")).len();
        let mut m = DenseMatrix {
            ncols: ncols,
            nrows: nrows,
            values: vec![0f64; ncols*nrows]
        };
        for row in 0..nrows {
            for col in 0..ncols {
                m.set(row, col, values[row][col]);
            }
        }
        m
    }

    pub fn from_array(nrows: usize, ncols: usize, values: &[f64]) -> DenseMatrix {
       DenseMatrix::from_vec(nrows, ncols, Vec::from(values)) 
    }

    pub fn from_vec(nrows: usize, ncols: usize, values: Vec<f64>) -> DenseMatrix {
        DenseMatrix {
            ncols: ncols,
            nrows: nrows,
            values: values
        }
    }

    pub fn div_mut(&mut self, b: DenseMatrix) -> () {
        if self.nrows != b.nrows || self.ncols != b.ncols {
            panic!("Can't divide matrices of different sizes.");
        }

        for i in 0..self.values.len() {
            self.values[i] /= b.values[i];
        }
    }

    fn set(&mut self, row: usize, col: usize, x: f64) {
        self.values[col*self.nrows + row] = x;        
    }

    fn div_element_mut(&mut self, row: usize, col: usize, x: f64) {
        self.values[col*self.nrows + row] /= x;
    }

    fn add_element_mut(&mut self, row: usize, col: usize, x: f64) {
        self.values[col*self.nrows + row] += x
    }

    fn sub_element_mut(&mut self, row: usize, col: usize, x: f64) {
        self.values[col*self.nrows + row] -= x;
    }
    
}

impl PartialEq for DenseMatrix {
    fn eq(&self, other: &Self) -> bool {
        if self.ncols != other.ncols || self.nrows != other.nrows {
            return false
        }

        let len = self.values.len();
        let other_len = other.values.len();

        if len != other_len {
            return false;
        }

        for i in 0..len {
            if (self.values[i] - other.values[i]).abs() > math::SMALL_ERROR {
                return false;
            }
        }

        true
    }
}

impl Into<Vec<f64>> for DenseMatrix {
    fn into(self) -> Vec<f64> {
        self.values
    }
}

impl Matrix for DenseMatrix {

    fn get(&self, row: usize, col: usize) -> f64 {
        self.values[col*self.nrows + row]
    }

    fn zeros(nrows: usize, ncols: usize) -> DenseMatrix {
        DenseMatrix::fill(nrows, ncols, 0f64)
    }

    fn ones(nrows: usize, ncols: usize) -> DenseMatrix {
        DenseMatrix::fill(nrows, ncols, 1f64)
    }

    fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    fn v_stack(&self, other: &Self) -> Self {
        if self.ncols != other.ncols {
            panic!("Number of columns in both matrices should be equal");
        }
        let mut result = DenseMatrix::zeros(self.nrows + other.nrows, self.ncols);
        for c in 0..self.ncols {
            for r in 0..self.nrows+other.nrows {                 
                if r <  self.nrows {              
                    result.set(r, c, self.get(r, c));
                } else {
                    result.set(r, c, other.get(r - self.nrows, c));
                }
            }
        }    
        result    
    }

    fn h_stack(&self, other: &Self) -> Self{
        if self.nrows != other.nrows {
            panic!("Number of rows in both matrices should be equal");
        }
        let mut result = DenseMatrix::zeros(self.nrows, self.ncols + other.ncols);
        for r in 0..self.nrows {
            for c in 0..self.ncols+other.ncols {                             
                if c <  self.ncols {              
                    result.set(r, c, self.get(r, c));
                } else {
                    result.set(r, c, other.get(r, c - self.ncols));
                }
            }
        }
        result
    }

    fn dot(&self, other: &Self) -> Self {
        if self.ncols != other.nrows {
            panic!("Number of rows of A should equal number of columns of B");
        }
        let inner_d = self.ncols;
        let mut result = DenseMatrix::zeros(self.nrows, other.ncols);

        for r in 0..self.nrows {
            for c in 0..other.ncols {
                let mut s = 0f64;
                for i in 0..inner_d {
                    s += self.get(r, i) * other.get(i, c);
                }
                result.set(r, c, s);
            }
        }

        result
    }

    fn slice(&self, rows: Range<usize>, cols: Range<usize>) -> DenseMatrix {

        let ncols = cols.len();
        let nrows = rows.len();

        let mut m = DenseMatrix::from_vec(nrows, ncols, vec![0f64; nrows * ncols]);

        for r in rows.start..rows.end {
            for c in cols.start..cols.end {
                m.set(r-rows.start, c-cols.start, self.get(r, c));
            }
        }

        m
    }

    fn qr_solve_mut(&mut self, mut b: DenseMatrix) -> DenseMatrix {
        let m = self.nrows;
        let n = self.ncols;
        let nrhs = b.ncols;

        let mut r_diagonal: Vec<f64> = vec![0f64; n];

        for k in 0..n {
            let mut nrm = 0f64;
            for i in k..m {
                nrm = nrm.hypot(self.get(i, k));
            }

            if nrm > math::SMALL_ERROR {

                if self.get(k, k) < 0f64 {
                    nrm = -nrm;
                }
                for i in k..m {
                    self.div_element_mut(i, k, nrm);
                }
                self.add_element_mut(k, k, 1f64);

                for j in k+1..n {
                    let mut s = 0f64;
                    for i in k..m {
                        s += self.get(i, k) * self.get(i, j);
                    }
                    s = -s / self.get(k, k);
                    for i in k..m {
                        self.add_element_mut(i, j, s * self.get(i, k));
                    }
                }
            }            
            r_diagonal[k] = -nrm;
        }        
        
        for j in 0..r_diagonal.len() {
            if r_diagonal[j].abs() < math::SMALL_ERROR  {
                panic!("Matrix is rank deficient.");                
            }
        }

        for k in 0..n {
            for j in 0..nrhs {
                let mut s = 0f64;
                for i in k..m {
                    s += self.get(i, k) * b.get(i, j);
                }
                s = -s / self.get(k, k);
                for i in k..m {
                    b.add_element_mut(i, j, s * self.get(i, k));
                }
            }
        }         

        for k in (0..n).rev() {
            for j in 0..nrhs {
                b.set(k, j, b.get(k, j) / r_diagonal[k]);
            }
            
            for i in 0..k {
                for j in 0..nrhs {
                    b.sub_element_mut(i, j, b.get(k, j) * self.get(i, k));
                }
            }
        }

        b

    }

    fn approximate_eq(&self, other: &Self, error: f64) -> bool {
        if self.ncols != other.ncols || self.nrows != other.nrows {
            return false
        }

        for c in 0..self.ncols {
            for r in 0..self.nrows {
                if (self.get(r, c) - other.get(r, c)).abs() > error {
                    return false
                }
            }
        }

        true
    }

    fn fill(nrows: usize, ncols: usize, value: f64) -> Self {
        DenseMatrix::from_vec(nrows, ncols, vec![value; ncols * nrows])
    }

    fn add_mut(&mut self, other: &Self) {
        if self.ncols != other.ncols || self.nrows != other.nrows {
            panic!("A and B should have the same shape");
        }        
        for c in 0..self.ncols {
            for r in 0..self.nrows {
                self.add_element_mut(r, c, other.get(r, c));
            }
        }
    }

}

#[cfg(test)]
mod tests {    
    use super::*; 

    #[test]
    fn qr_solve_mut() { 

            let mut a = DenseMatrix::from_2d_array(&[&[0.9, 0.4, 0.7], &[0.4, 0.5, 0.3], &[0.7, 0.3, 0.8]]);
            let b = DenseMatrix::from_2d_array(&[&[0.5, 0.2],&[0.5, 0.8], &[0.5, 0.3]]);
            let expected_w = DenseMatrix::from_array(3, 2, &[-0.20270270270270263, 0.8783783783783784, 0.4729729729729729, -1.2837837837837829, 2.2297297297297303, 0.6621621621621613]);
            let w = a.qr_solve_mut(b);   
            assert_eq!(w, expected_w);
    }

    #[test]
    fn v_stack() { 

            let a = DenseMatrix::from_2d_array(
                &[
                    &[1., 2., 3.],
                    &[4., 5., 6.],
                    &[7., 8., 9.]]);
            let b = DenseMatrix::from_2d_array(
                &[
                    &[1., 2., 3.],
                    &[4., 5., 6.]]);
            let expected = DenseMatrix::from_2d_array(
                &[
                    &[1., 2., 3.], 
                    &[4., 5., 6.], 
                    &[7., 8., 9.], 
                    &[1., 2., 3.], 
                    &[4., 5., 6.]]);
            let result = a.v_stack(&b);               
            assert_eq!(result, expected);
    }

    #[test]
    fn h_stack() { 

            let a = DenseMatrix::from_2d_array(
                &[
                    &[1., 2., 3.],
                    &[4., 5., 6.],
                    &[7., 8., 9.]]);
            let b = DenseMatrix::from_2d_array(
                &[
                    &[1., 2.],
                    &[3., 4.],
                    &[5., 6.]]);
            let expected = DenseMatrix::from_2d_array(
                &[
                    &[1., 2., 3., 1., 2.], 
                    &[4., 5., 6., 3., 4.], 
                    &[7., 8., 9., 5., 6.]]);
            let result = a.h_stack(&b);               
            assert_eq!(result, expected);
    }

    #[test]
    fn dot() { 

            let a = DenseMatrix::from_2d_array(
                &[
                    &[1., 2., 3.],
                    &[4., 5., 6.]]);
            let b = DenseMatrix::from_2d_array(
                &[
                    &[1., 2.],
                    &[3., 4.],
                    &[5., 6.]]);
            let expected = DenseMatrix::from_2d_array(
                &[
                    &[22., 28.], 
                    &[49., 64.]]);
            let result = a.dot(&b);               
            assert_eq!(result, expected);
    }

    #[test]
    fn slice() { 

            let m = DenseMatrix::from_2d_array(
                &[
                    &[1., 2., 3., 1., 2.], 
                    &[4., 5., 6., 3., 4.], 
                    &[7., 8., 9., 5., 6.]]);
            let expected = DenseMatrix::from_2d_array(
                &[
                    &[2., 3.], 
                    &[5., 6.]]);
            let result = m.slice(0..2, 1..3);
            assert_eq!(result, expected);
    }
    

    #[test]
    fn approximate_eq() {             
            let m = DenseMatrix::from_2d_array(
                &[
                    &[2., 3.], 
                    &[5., 6.]]);
            let m_eq = DenseMatrix::from_2d_array(
                &[
                    &[2.5, 3.0], 
                    &[5., 5.5]]);
            let m_neq = DenseMatrix::from_2d_array(
                &[
                    &[3.0, 3.0], 
                    &[5., 6.5]]);            
            assert!(m.approximate_eq(&m_eq, 0.5));
            assert!(!m.approximate_eq(&m_neq, 0.5));
    }

}

