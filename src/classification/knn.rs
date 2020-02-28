use crate::linalg::{Matrix, row_iter};
use crate::algorithm::neighbour::{KNNAlgorithm, KNNAlgorithmName};
use crate::algorithm::neighbour::linear_search::LinearKNNSearch;
use crate::algorithm::neighbour::cover_tree::CoverTree;


type F = dyn Fn(&Vec<f64>, &Vec<f64>) -> f64;

pub struct KNNClassifier<'a> {     
    classes: Vec<f64>,
    y: Vec<usize>,    
    knn_algorithm: Box<dyn KNNAlgorithm<Vec<f64>> + 'a>,
    k: usize,        
}

impl<'a> KNNClassifier<'a> {

    pub fn fit<M: Matrix>(x: &M, y: &M::RowVector, k: usize, distance: &'a F, algorithm: KNNAlgorithmName) -> KNNClassifier<'a> {

        let y_m = M::from_row_vector(y.clone());

        let (_, y_n) = y_m.shape();
        let (x_n, _) = x.shape();

        let data = row_iter(x).collect();

        let mut yi: Vec<usize> = vec![0; y_n];
        let classes = y_m.unique();                

        for i in 0..y_n {
            let yc = y_m.get(0, i);                        
            yi[i] = classes.iter().position(|c| yc == *c).unwrap();            
        }

        assert!(x_n == y_n, format!("Size of x should equal size of y; |x|=[{}], |y|=[{}]", x_n, y_n));

        assert!(k > 1, format!("k should be > 1, k=[{}]", k));                  

        let knn_algorithm: Box<dyn KNNAlgorithm<Vec<f64>> + 'a> = match algorithm {
            KNNAlgorithmName::CoverTree => Box::new(CoverTree::<Vec<f64>>::new(data, distance)),
            KNNAlgorithmName::LinearSearch => Box::new(LinearKNNSearch::<Vec<f64>>::new(data, distance))
        };

        KNNClassifier{classes:classes, y: yi, k: k, knn_algorithm: knn_algorithm}

    }

    pub fn predict<M: Matrix>(&self, x: &M) -> M::RowVector {    
        let mut result = M::zeros(1, x.shape().0);           
        
        row_iter(x).enumerate().for_each(|(i, x)| result.set(0, i, self.classes[self.predict_for_row(x)]));

        result.to_row_vector()
    }

    fn predict_for_row(&self, x: Vec<f64>) -> usize {        

        let idxs = self.knn_algorithm.find(&x, self.k);        
        let mut c = vec![0; self.classes.len()];
        let mut max_c = 0;
        let mut max_i = 0;
        for i in idxs {
            c[self.y[i]] += 1;  
            if c[self.y[i]] > max_c {
                max_c = c[self.y[i]];
                max_i = self.y[i];
            }          
        }   

        max_i

    }
    
}

#[cfg(test)]
mod tests {    
    use super::*;    
    use crate::math::distance::euclidian; 
    use crate::linalg::naive::dense_matrix::DenseMatrix;   

    #[test]
    fn knn_fit_predict() {                
        let x = DenseMatrix::from_array(&[
            &[1., 2.],
            &[3., 4.],
            &[5., 6.], 
            &[7., 8.], 
            &[9., 10.]]); 
        let y = vec![2., 2., 2., 3., 3.];        
        let knn = KNNClassifier::fit(&x, &y, 3, &euclidian::distance, KNNAlgorithmName::LinearSearch);        
        let r = knn.predict(&x);
        assert_eq!(5, Vec::len(&r));
        assert_eq!(y.to_vec(), r);
    }
}