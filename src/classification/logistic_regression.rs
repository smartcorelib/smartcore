use std::marker::PhantomData;
use crate::linalg::{Matrix, Vector};
use crate::optimization::FunctionOrder;
use crate::optimization::first_order::FirstOrderOptimizer;
use crate::optimization::line_search::Backtracking;
use crate::optimization::first_order::lbfgs::LBFGS;

#[derive(Debug)]
pub struct LogisticRegression<M: Matrix, V: Vector> {    
    weights: M,     
    classes: Vec<f64>,
    num_attributes: usize,
    num_classes: usize,
    v_phantom: PhantomData<V>
}

struct MultiClassObjectiveFunction<'a, M: Matrix> {
    x: &'a M,
    y: Vec<usize>,
    k: usize
}

impl<'a, M: Matrix> MultiClassObjectiveFunction<'a, M> {    

    fn f<X: Vector>(&self, w: &X) -> f64 {                
        let mut f = 0.;
        let mut prob = X::zeros(self.k);
        let (n, p) = self.x.shape();
        for i in 0..n {
            for j in 0..self.k {
                prob.set(j, MultiClassObjectiveFunction::dot(w, self.x, j * (p + 1), i));
            }
            prob.softmax_mut();
            f -= prob.get(self.y[i]).ln();
        }        
        
        f 
    }

    fn df<X: Vector>(&self, g: &mut X, w: &X) {
        
        g.copy_from(&X::zeros(g.shape().1));

        let mut f = 0.;
        let mut prob = X::zeros(self.k);
        let (n, p) = self.x.shape();        
        
        for i in 0..n {
            for j in 0..self.k {
                prob.set(j, MultiClassObjectiveFunction::dot(w, self.x, j * (p + 1), i));
            }

            prob.softmax_mut();
            f -= prob.get(self.y[i]).ln();

            for j in 0..self.k {
                let yi =(if self.y[i] == j { 1.0 } else { 0.0 }) - prob.get(j);
    
                for l in 0..p {
                    let pos = j * (p + 1);
                    g.set(pos + l, g.get(pos + l) - yi * self.x.get(i, l));
                }
                g.set(j * (p + 1) + p, g.get(j * (p + 1) + p) - yi);                
            }
        }        
        
    }
    
    fn dot<X: Vector>(v: &X, m: &M, v_pos: usize, w_row: usize) -> f64 {
        let mut sum = 0f64;  
        let p =  m.shape().1;    
        for i in 0..p {
            sum += m.get(w_row, i) * v.get(i + v_pos);
        }

        sum + v.get(p + v_pos)
    }

}

impl<M: Matrix, V: Vector> LogisticRegression<M, V> {

    pub fn fit(x: &M, y: &V) -> LogisticRegression<M, V>{

        let (x_nrows, num_attributes) = x.shape();
        let (_, y_nrows) = y.shape();

        if x_nrows != y_nrows {
            panic!("Number of rows of X doesn't match number of rows of Y");
        }
        
        let mut classes = y.unique();        

        let k = classes.len();                      

        let x0 = V::zeros((num_attributes + 1) * k);

        let mut yi: Vec<usize> = vec![0; y_nrows];

        for i in 0..y_nrows {
            let yc = y.get(i); 
            let j = classes.iter().position(|c| yc == *c).unwrap();            
            yi[i] = classes.iter().position(|c| yc == *c).unwrap();
        }

        if k < 2 {

            panic!("Incorrect number of classes: {}", k);

        } else if k == 2 {

            LogisticRegression {                 
                weights: x.clone(),               
                classes: classes,
                num_attributes: num_attributes,
                num_classes: k,
                v_phantom: PhantomData
            }

        } else {

            let objective = MultiClassObjectiveFunction{
                x: x,
                y: yi,
                k: k
            };

            let f = |w: &V| -> f64 {                
                objective.f(w)
            };

            let df = |g: &mut V, w: &V| {
                objective.df(g, w)
            };

            let mut ls: Backtracking = Default::default();
            ls.order = FunctionOrder::THIRD;
            let optimizer: LBFGS = Default::default();  
            
            let result = optimizer.optimize(&f, &df, &x0, &ls);                        

            let weights = M::from_vector(&result.x, k, num_attributes + 1);            

            LogisticRegression {
                weights: weights,                
                classes: classes,
                num_attributes: num_attributes,
                num_classes: k,
                v_phantom: PhantomData
            }            
        }        
        
        
    }

    pub fn predict(&self, x: &M) -> V {
        let (nrows, _) = x.shape();
        let x_and_bias = x.h_stack(&M::ones(nrows, 1));        
        let mut y_hat = x_and_bias.dot(&self.weights.transpose());        
        y_hat.softmax_mut();
        let class_idxs = y_hat.argmax();
        V::from_vec(&class_idxs.iter().map(|class_idx| self.classes[*class_idx]).collect())
    }

    pub fn coefficients(&self) -> M {
        self.weights.slice(0..self.num_classes, 0..self.num_attributes)
    }

    pub fn intercept(&self) -> M {
        self.weights.slice(0..self.num_classes, self.num_attributes..self.num_attributes+1)
    }

}

#[cfg(test)]
mod tests {    
    use super::*; 
    use crate::linalg::naive::dense_matrix::DenseMatrix;
    use crate::linalg::naive::dense_vector::DenseVector;

    #[test]
    fn multiclass_objective_f() { 

        let x = DenseMatrix::from_2d_array(&[
            &[1., -5.],
            &[ 2.,  5.],
            &[ 3., -2.],
            &[ 1.,  2.],
            &[ 2.,  0.],
            &[ 6., -5.],
            &[ 7.,  5.],
            &[ 6., -2.],
            &[ 7.,  2.],
            &[ 6.,  0.],
            &[ 8., -5.],
            &[ 9.,  5.],
            &[10., -2.],
            &[ 8.,  2.],
            &[ 9.,  0.]]);

        let y = vec![0, 0, 1, 1, 2, 1, 1, 0, 0, 2, 1, 1, 0, 0, 1];

        let objective = MultiClassObjectiveFunction{
            x: &x,
            y: y,
            k: 3
        };

        let mut g = DenseVector::zeros(9);                

        objective.df(&mut g, &DenseVector::from_array(&[1., 2., 3., 4., 5., 6., 7., 8., 9.]));
        objective.df(&mut g, &DenseVector::from_array(&[1., 2., 3., 4., 5., 6., 7., 8., 9.]));
        
        assert!((g.get(0) + 33.000068218163484).abs() < std::f64::EPSILON); 

        let f = objective.f(&DenseVector::from_array(&[1., 2., 3., 4., 5., 6., 7., 8., 9.]));

        assert!((f -  408.0052230582765).abs() < std::f64::EPSILON);        
    }

    #[test]
    fn lr_fit_predict() {             

        let x = DenseMatrix::from_2d_array(&[
            &[1., -5.],
            &[ 2.,  5.],
            &[ 3., -2.],
            &[ 1.,  2.],
            &[ 2.,  0.],
            &[ 6., -5.],
            &[ 7.,  5.],
            &[ 6., -2.],
            &[ 7.,  2.],
            &[ 6.,  0.],
            &[ 8., -5.],
            &[ 9.,  5.],
            &[10., -2.],
            &[ 8.,  2.],
            &[ 9.,  0.]]);
        let y = DenseVector::from_array(&[0., 0., 1., 1., 2., 1., 1., 0., 0., 2., 1., 1., 0., 0., 1.]);

        let lr = LogisticRegression::fit(&x, &y);

        assert_eq!(lr.coefficients().shape(), (3, 2));
        assert_eq!(lr.intercept().shape(), (3, 1));
        
        assert!((lr.coefficients().get(0, 0) - 0.0435).abs() < 1e-4);
        assert!((lr.intercept().get(0, 0) - 0.1250).abs() < 1e-4);        

        let y_hat = lr.predict(&x);                

        assert_eq!(y_hat, DenseVector::from_array(&[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]));
        

    }
}