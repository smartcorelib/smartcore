use std::ops::Range;

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

}