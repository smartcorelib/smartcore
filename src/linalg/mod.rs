use std::ops::Range;
use std::fmt::Debug;

pub mod naive;

pub trait Matrix: Into<Vec<f64>> + Clone{ 

    fn get(&self, row: usize, col: usize) -> f64; 

    fn qr_solve_mut(&mut self, b: Self) -> Self;

    fn svd_solve_mut(&mut self, b: Self) -> Self;

    fn zeros(nrows: usize, ncols: usize) -> Self;

    fn ones(nrows: usize, ncols: usize) -> Self;

    fn fill(nrows: usize, ncols: usize, value: f64) -> Self;

    fn shape(&self) -> (usize, usize);

    fn v_stack(&self, other: &Self) -> Self;

    fn h_stack(&self, other: &Self) -> Self;

    fn dot(&self, other: &Self) -> Self;

    fn slice(&self, rows: Range<usize>, cols: Range<usize>) -> Self;

    fn approximate_eq(&self, other: &Self, error: f64) -> bool;

    fn add_mut(&mut self, other: &Self);

    fn add_scalar_mut(&mut self, scalar: f64);

    fn sub_scalar_mut(&mut self, scalar: f64);

    fn mul_scalar_mut(&mut self, scalar: f64);

    fn div_scalar_mut(&mut self, scalar: f64);

    fn transpose(&self) -> Self;

    fn generate_positive_definite(nrows: usize, ncols: usize) -> Self;

    fn rand(nrows: usize, ncols: usize) -> Self;

    fn norm2(&self) -> f64;

    fn negative_mut(&mut self);

}

pub trait Vector: Into<Vec<f64>> + Clone + Debug {

    fn get(&self, i: usize) -> f64; 

    fn set(&mut self, i: usize, value: f64); 

    fn zeros(size: usize) -> Self;

    fn ones(size: usize) -> Self;

    fn fill(size: usize, value: f64) -> Self;

    fn shape(&self) -> (usize, usize);

    fn norm2(&self) -> f64;

    fn negative_mut(&mut self) -> &Self;

    fn negative(&self) -> Self;

    fn add_mut(&mut self, other: &Self) -> &Self;

    fn add_scalar_mut(&mut self, scalar: f64) -> &Self;

    fn sub_scalar_mut(&mut self, scalar: f64) -> &Self;

    fn mul_scalar_mut(&mut self, scalar: f64) -> &Self;

    fn div_scalar_mut(&mut self, scalar: f64) -> &Self;

    fn dot(&self, other: &Self) -> f64;

} 