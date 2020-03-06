use crate::linalg::{Matrix};

#[derive(Debug, Clone)]
pub struct EVD<M: Matrix> {        
    pub d: Vec<f64>,
    pub e: Vec<f64>,
    pub V: M
}

impl<M: Matrix> EVD<M> {
    pub fn new(V: M, d: Vec<f64>, e: Vec<f64>) -> EVD<M> {
        EVD {
            d: d,
            e: e,
            V: V
        }
    }
}

#[cfg(test)]
mod tests {    
    use super::*; 
    use crate::linalg::naive::dense_matrix::DenseMatrix;

    #[test]
    fn decompose_symmetric() { 

        let A = DenseMatrix::from_array(&[
            &[0.9000, 0.4000, 0.7000],
            &[0.4000, 0.5000, 0.3000],
            &[0.7000, 0.3000, 0.8000]]);

        let eigen_values = vec![1.7498382, 0.3165784, 0.1335834];

        let eigen_vectors = DenseMatrix::from_array(&[
            &[0.6881997, -0.07121225, 0.7220180],
            &[0.3700456, 0.89044952, -0.2648886],
            &[0.6240573, -0.44947578, -0.6391588]
        ]);

        let evd = A.evd(true);        

        assert!(eigen_vectors.abs().approximate_eq(&evd.V.abs(), 1e-4));        
        for i in 0..eigen_values.len() {
            assert!((eigen_values[i] - evd.d[i]).abs() < 1e-4);
        }
        for i in 0..eigen_values.len() {
            assert!((0f64 - evd.e[i]).abs() < std::f64::EPSILON);
        }

    }

    #[test]
    fn decompose_asymmetric() { 

        let A = DenseMatrix::from_array(&[
            &[0.9000, 0.4000, 0.7000],
            &[0.4000, 0.5000, 0.3000],
            &[0.8000, 0.3000, 0.8000]]);

        let eigen_values = vec![1.79171122, 0.31908143, 0.08920735];

        let eigen_vectors = DenseMatrix::from_array(&[
            &[0.7178958,  0.05322098,  0.6812010],
            &[0.3837711, -0.84702111, -0.1494582],
            &[0.6952105,  0.43984484, -0.7036135]
        ]);

        let evd = A.evd(false);   
        
        assert!(eigen_vectors.abs().approximate_eq(&evd.V.abs(), 1e-4));        
        for i in 0..eigen_values.len() {
            assert!((eigen_values[i] - evd.d[i]).abs() < 1e-4);
        }
        for i in 0..eigen_values.len() {
            assert!((0f64 - evd.e[i]).abs() < std::f64::EPSILON);
        }

    }

    #[test]
    fn decompose_complex() { 

        let A = DenseMatrix::from_array(&[
            &[3.0, -2.0, 1.0, 1.0],
            &[4.0, -1.0, 1.0, 1.0],
            &[1.0, 1.0, 3.0, -2.0],
            &[1.0, 1.0, 4.0, -1.0]]);

        let eigen_values_d = vec![0.0, 2.0, 2.0, 0.0];
        let eigen_values_e = vec![2.2361, 0.9999, -0.9999, -2.2361];

        let eigen_vectors = DenseMatrix::from_array(&[
            &[-0.9159, -0.1378, 0.3816, -0.0806],
            &[-0.6707, 0.1059, 0.901, 0.6289],
            &[0.9159, -0.1378, 0.3816, 0.0806],
            &[0.6707, 0.1059, 0.901, -0.6289]
        ]);

        let evd = A.evd(false);   
        
        assert!(eigen_vectors.abs().approximate_eq(&evd.V.abs(), 1e-4));        
        for i in 0..eigen_values_d.len() {
            assert!((eigen_values_d[i] - evd.d[i]).abs() < 1e-4);
        }
        for i in 0..eigen_values_e.len() {
            assert!((eigen_values_e[i] - evd.e[i]).abs() < 1e-4);
        }

    }
    
}