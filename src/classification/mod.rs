use crate::common::Nominal;

pub mod knn;
pub mod logistic_regression;

pub trait Classifier<X, Y>
where 
    Y: Nominal
{    

    fn predict(&self, x: &X) -> Y;

    fn predict_vec(&self, x: &Vec<X>) -> Vec<Y>{
        let mut result = Vec::new();        
        for xv in x.iter() {
            result.push(self.predict(xv));
        }          
        result
    }

}