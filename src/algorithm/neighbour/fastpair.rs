use std::collections::HashMap;

use crate::linalg::Matrix;
use crate::math::num::RealNumber;
use crate::math::distance::euclidian::Euclidian;
use crate::algorithm::neighbour::dissimilarities::PairwiseDissimilarity;

///
/// FastPair: Data-structure for the dynamic closest-pair problem.
/// 
/// Reference:
///  Eppstein, David: Fast hierarchical clustering and other applications of
///  dynamic closest pairs. Journal of Experimental Algorithmics 5 (2000) 1.
/// 
/// Python implementation:
/// <https://github.com/carsonfarmer/fastpair/blob/b8b4d3000ab6f795a878936667eee1b557bf353d/fastpair/base.py>
#[derive(Debug)]
pub struct FastPair<'a, T: RealNumber, M: Matrix<T>> {
    samples: &'a M,
    neighbours: Box<HashMap<&'a usize, PairwiseDissimilarity<T>>>,
    nodes: Box<Vec<PairwiseDissimilarity<T>>>
}

impl<'a, T: RealNumber, M: Matrix<T>> FastPair<'a, T, M> {
    fn new(&self, m: &'a M) {
        // Go through and find all neighbors, placing then in a conga line
        self.samples = m;
        let mut neighbours = Box::new(HashMap::with_capacity(self.samples.shape().0));
        let mut nodes      = Box::new(Vec::with_capacity(self.samples.shape().0));

        let last_idx = self.samples.shape().0 - 1;
        // initilize nodes list
        for k in 0..last_idx {
            (*nodes).push(
                PairwiseDissimilarity {
                    node: k,
                    neighbour: None,
                    distance: None  // None means MAX_DISTANCE
                }
            )
        }

        // loop through nodes and neighbours
        for i in 0..last_idx {
            neighbours.insert(
                &i,
                PairwiseDissimilarity {
                    node: i,
                    neighbour: None,
                    distance: None 
                }
            );

            // start looking for the neighbour in the second element
            let nbr = i + 1;
            let nbd: Option<T> = None;
            for j in (i+1)..last_idx {
                let row_i = &(self.samples.get_row_as_vec(i));
                let row_j = &(self.samples.get_row_as_vec(j));

                let d = Euclidian::squared_distance(row_i, row_j);
                if nbd.is_none() || d < nbd.unwrap() {
                    nbr = j;
                    nbd = Some(d);
                }
            }
            // Add that edge, move nbr to points[i+1]
            neighbours[&i].distance = nbd;
            neighbours[&i].neighbour = Some((*nodes)[nbr].node);
            (*nodes)[nbr] = (*nodes)[i+1];
            (*nodes)[i+1] = PairwiseDissimilarity {
                node: neighbours[&i].neighbour.unwrap(),
                neighbour: None,
                distance: None 
            };
            
        }
        // No more neighbors, terminate conga line.
        // Last person on the line has no neigbors
        neighbours[&last_idx].neighbour = Some(last_idx);
        neighbours[&last_idx].distance = None;

        self.neighbours = Box::new(*neighbours);
        self.nodes = Box::new(*nodes);
    }

    // Find and update nearest neighbor of a given point.
    fn find_neighbour(&self, node: usize) -> &PairwiseDissimilarity<T> {
        // Find first point unequal to `node` itself
        let first_nbr = 0;
        if node == (*self.nodes)[first_nbr].node {
            first_nbr = 1;
        }
        self.neighbours[&node].neighbour = Some((*self.nodes)[first_nbr].node);
        self.neighbours[&node].distance = Some(Euclidian::squared_distance(
            &(self.samples.get_row_as_vec(node)),
            &(self.samples.get_row_as_vec(self.neighbours[&node].neighbour.unwrap()))
        ));
        // Now test whether each other point is closer
        for &q in self.nodes[first_nbr+1..(self.samples.shape().0 - 1)].iter().clone() {
            if node != q.node {
                let d = Euclidian::squared_distance(
                    &(self.samples.get_row_as_vec(node)),
                    &(self.samples.get_row_as_vec(q.node))
                );
                if d < self.neighbours[&node].distance.unwrap() {
                    self.neighbours[&node].distance = Some(d);
                    self.neighbours[&node].neighbour = Some(q.node);
                }
            }
        }

        return &(*self.neighbours)[&node]
    }

}


#[cfg(test)]
mod tests {

    use super::*;
    use crate::linalg::naive::dense_matrix::*;

    #[test]
    fn fastpair_init() {
        let x: DenseMatrix<f64> = DenseMatrix::rand(10, 4);
        let fastpair = FastPair(x);

        print!("{:?}", fastpair);       
    }
}
