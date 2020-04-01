extern crate rand;

use std::default::Default;
use std::fmt::Debug;

use rand::Rng;

use crate::math::num::FloatExt;
use crate::linalg::Matrix;
use crate::tree::decision_tree_classifier::{DecisionTreeClassifier, DecisionTreeClassifierParameters, SplitCriterion, which_max};

#[derive(Debug, Clone)]
pub struct RandomForestClassifierParameters {  
    pub criterion: SplitCriterion,   
    pub max_depth: Option<u16>,
    pub min_samples_leaf: usize,    
    pub min_samples_split: usize,          
    pub n_trees: u16,    
    pub mtry: Option<usize>
}

#[derive(Debug)]
pub struct RandomForestClassifier<T: FloatExt> {    
    parameters: RandomForestClassifierParameters,
    trees: Vec<DecisionTreeClassifier<T>>,
    classes: Vec<T>
}

impl Default for RandomForestClassifierParameters {
    fn default() -> Self { 
        RandomForestClassifierParameters {
            criterion: SplitCriterion::Gini,
            max_depth: None,
            min_samples_leaf: 1,
            min_samples_split: 2,
            n_trees: 100,
            mtry: Option::None
        }
     }
}

impl<T: FloatExt> RandomForestClassifier<T> {

    pub fn fit<M: Matrix<T>>(x: &M, y: &M::RowVector, parameters: RandomForestClassifierParameters) -> RandomForestClassifier<T> {        
        let (_, num_attributes) = x.shape();
        let y_m = M::from_row_vector(y.clone());
        let (_, y_ncols) = y_m.shape();
        let mut yi: Vec<usize> = vec![0; y_ncols];
        let classes = y_m.unique();                

        for i in 0..y_ncols {
            let yc = y_m.get(0, i);                        
            yi[i] = classes.iter().position(|c| yc == *c).unwrap();            
        }
              
        let mtry = parameters.mtry.unwrap_or((T::from(num_attributes).unwrap()).sqrt().floor().to_usize().unwrap());        

        let classes = y_m.unique();        
        let k = classes.len(); 
        let mut trees: Vec<DecisionTreeClassifier<T>> = Vec::new();

        for _ in 0..parameters.n_trees {
            let samples = RandomForestClassifier::<T>::sample_with_replacement(&yi, k);
            let params = DecisionTreeClassifierParameters{
                criterion: parameters.criterion.clone(),
                max_depth: parameters.max_depth,
                min_samples_leaf: parameters.min_samples_leaf,   
                min_samples_split: parameters.min_samples_split          
            };
            let tree = DecisionTreeClassifier::fit_weak_learner(x, y, samples, mtry, params);
            trees.push(tree);
        }

        RandomForestClassifier {
            parameters: parameters,
            trees: trees,
            classes
        }
    }

    pub fn predict<M: Matrix<T>>(&self, x: &M) -> M::RowVector {
        let mut result = M::zeros(1, x.shape().0);   
        
        let (n, _) = x.shape();

        for i in 0..n {
            result.set(0, i, self.classes[self.predict_for_row(x, i)]);
        }

        result.to_row_vector()
    }  

    fn predict_for_row<M: Matrix<T>>(&self, x: &M, row: usize) -> usize {
        let mut result = vec![0; self.classes.len()];
        
        for tree in self.trees.iter() {
            result[tree.predict_for_row(x, row)] += 1;
        }        

        return which_max(&result)
        
    }  
    
    fn sample_with_replacement(y: &Vec<usize>, num_classes: usize) -> Vec<usize>{
        let mut rng = rand::thread_rng();
        let class_weight = vec![1.; num_classes];
        let nrows = y.len();
        let mut samples = vec![0; nrows];
        for l in 0..num_classes {
            let mut nj = 0;
            let mut cj: Vec<usize> = Vec::new();
            for i in 0..nrows {
                if y[i] == l {
                    cj.push(i);
                    nj += 1;
                }
            }
            
            let size = ((nj as f64) / class_weight[l]) as usize;            
            for _ in 0..size {
                let xi: usize = rng.gen_range(0, nj);
                samples[cj[xi]] += 1;
            }
        }
        samples
    }

}

#[cfg(test)]
mod tests {
    use super::*; 
    use crate::linalg::naive::dense_matrix::DenseMatrix;

    #[test]
    fn fit_predict_iris() {             

        let x = DenseMatrix::from_array(&[
            &[5.1, 3.5, 1.4, 0.2],
            &[4.9, 3.0, 1.4, 0.2],
            &[4.7, 3.2, 1.3, 0.2],
            &[4.6, 3.1, 1.5, 0.2],
            &[5.0, 3.6, 1.4, 0.2],
            &[5.4, 3.9, 1.7, 0.4],
            &[4.6, 3.4, 1.4, 0.3],
            &[5.0, 3.4, 1.5, 0.2],
            &[4.4, 2.9, 1.4, 0.2],
            &[4.9, 3.1, 1.5, 0.1],
            &[7.0, 3.2, 4.7, 1.4],
            &[6.4, 3.2, 4.5, 1.5],
            &[6.9, 3.1, 4.9, 1.5],
            &[5.5, 2.3, 4.0, 1.3],
            &[6.5, 2.8, 4.6, 1.5],
            &[5.7, 2.8, 4.5, 1.3],
            &[6.3, 3.3, 4.7, 1.6],
            &[4.9, 2.4, 3.3, 1.0],
            &[6.6, 2.9, 4.6, 1.3],
            &[5.2, 2.7, 3.9, 1.4]]);
        let y = vec![0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.];

        let classifier = RandomForestClassifier::fit(&x, &y, RandomForestClassifierParameters{
            criterion: SplitCriterion::Gini,
            max_depth: None,
            min_samples_leaf: 1,
            min_samples_split: 2,
            n_trees: 1000,
            mtry: Option::None
        });

        assert_eq!(y, classifier.predict(&x));                    
            
    }

}