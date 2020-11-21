#![allow(non_snake_case)]
///
/// FastPair: Data-structure for the dynamic closest-pair problem.
///
/// Reference:
///  Eppstein, David: Fast hierarchical clustering and other applications of
///  dynamic closest pairs. Journal of Experimental Algorithmics 5 (2000) 1.
///
use std::collections::HashMap;
use std::iter;

use crate::algorithm::neighbour::dissimilarities::PairwiseDissimilarity;
use crate::error::{Failed, FailedError};
use crate::linalg::naive::dense_matrix::*;
use crate::linalg::Matrix;
use crate::math::distance::euclidian::Euclidian;
use crate::math::num::RealNumber;

///
/// FastPair factory function
///
pub fn FastPair<'a, T: RealNumber, M: Matrix<T>>(m: &'a M) -> Result<_FastPair<T, M>, Failed> {
    if m.shape().0 < 3 {
        return Err(Failed::because(
            FailedError::FindFailed,
            "min number of rows is 3",
        ));
    }

    let distances = Box::new(HashMap::with_capacity(m.shape().0));
    let neighbours = Box::new(Vec::with_capacity(m.shape().0 + 1));

    let mut init = _FastPair {
        samples: m,
        distances: distances,
        neighbours: neighbours,
        // to be computed in new(..)
        connectivity: None,
    };
    init.new();
    Ok(init)
}

///
/// FastPair
///
/// Ported from Python implementation:
/// <https://github.com/carsonfarmer/fastpair/blob/b8b4d3000ab6f795a878936667eee1b557bf353d/fastpair/base.py>
/// MIT License (MIT) Copyright (c) 2016 Carson Farmer
///
/// affinity used is Euclidean so to allow linkage with single, ward, complete and average
///
#[derive(Debug, Clone)]
pub struct _FastPair<'a, T: RealNumber, M: Matrix<T>> {
    /// initial matrix
    samples: &'a M,
    /// closest pair hashmap (connectivity matrix for closest pairs)
    pub distances: Box<HashMap<usize, PairwiseDissimilarity<T>>>,
    /// conga line used to keep track of the closest pair
    pub neighbours: Box<Vec<usize>>,
    /// sparse matrix of closest pairs
    /// values are set for closest pairs distances, other pairs are zeroed
    pub connectivity: Option<Box<M>>,
}

impl<'a, T: RealNumber, M: Matrix<T>> _FastPair<'a, T, M> {
    ///
    /// Initialise `FastPair` by passing a `Matrix`.
    /// Build a FastPairs data-structure from a set of (new) points.
    ///
    fn new(&mut self) {
        // basic measures
        let len = self.samples.shape().0;
        let max_index = self.samples.shape().0 - 1;

        // Store all closest neighbors
        let _distances = Box::new(HashMap::with_capacity(len));
        let _neighbours = Box::new(Vec::with_capacity(len));

        let mut distances = *_distances;
        let mut neighbours = *_neighbours;

        // fill neighbours with -1 values
        neighbours.extend(iter::repeat(0).take(len));

        // loop through indeces and neighbours
        for index_row_i in 0..len {
            // init closest neighbour pairwise data
            distances.insert(
                index_row_i,
                PairwiseDissimilarity {
                    node: index_row_i,
                    neighbour: None,
                    distance: Some(T::max_value()),
                },
            );

            // start looking for the neighbour in the second element
            let mut index_closest = index_row_i + 1; // closest neighbour index
            let mut nbd: Option<T> = Some(T::max_value()); // init neighbour distance
            for index_row_j in (index_row_i + 1)..len {
                distances.insert(
                    index_row_j,
                    PairwiseDissimilarity {
                        node: index_row_j,
                        neighbour: None,
                        distance: Some(T::max_value()),
                    },
                );

                let d = Euclidian::squared_distance(
                    &(self.samples.get_row_as_vec(index_row_i)),
                    &(self.samples.get_row_as_vec(index_row_j)),
                );
                if d < nbd.unwrap() {
                    // set this j-value to be the closest neighbour
                    index_closest = index_row_j;
                    nbd = Some(d);
                }
            }
            // Add that edge, move nbr to points[i+1] in conga line
            distances.entry(index_row_i).and_modify(|e| {
                e.distance = nbd;
                e.neighbour = Some(index_closest);
            });

            // update conga line
            if index_closest != len {
                neighbours[index_closest] = neighbours[index_row_i + 1];
                neighbours[index_row_i + 1] = index_closest;
            }
        }
        // No more neighbors, terminate conga line.
        // Last person on the line has no neigbors
        distances.get_mut(&max_index).unwrap().neighbour = Some(max_index);
        distances.get_mut(&(len - 1)).unwrap().distance = Some(T::max_value());

        // compute sparse matrix (connectivity matrix)
        let mut sparse_matrix = M::zeros(len, len);
        for (_, p) in distances.iter() {
            sparse_matrix.set(p.node, p.neighbour.unwrap(), p.distance.unwrap());
        }

        // TODO: as we now store the connectivity matrix in `self.connectivity`,
        //       it may be possible to avoid storing closest pairs in `self.distances`
        self.distances = Box::new(distances);
        self.neighbours = Box::new(neighbours);
        self.connectivity = Some(Box::new(sparse_matrix));
    }

    ///
    /// Find closest pair by scanning list of nearest neighbors.
    ///
    pub fn closest_pair(&self) -> PairwiseDissimilarity<T> {
        let mut a = self.neighbours[0]; // Start with first point
        let mut d = self.distances[&a].distance;
        for p in self.neighbours.iter() {
            if self.distances[&p].distance < d {
                a = *p; // Update `a` and distance `d`
                d = self.distances[&p].distance;
            }
        }
        let b = self.distances[&a].neighbour;
        PairwiseDissimilarity {
            node: a,
            neighbour: b,
            distance: d,
        }
    }

    //
    // Compute distances from input to all other points in data-structure.
    // input is the row index of the sample matrix
    //
    fn distances_from(&self, index_row: usize) -> Vec<PairwiseDissimilarity<T>> {
        let mut distances = Vec::<PairwiseDissimilarity<T>>::with_capacity(self.samples.shape().0);
        for other in self.neighbours.iter() {
            if index_row != *other {
                distances.push(PairwiseDissimilarity {
                    node: index_row,
                    neighbour: Some(*other),
                    distance: Some(Euclidian::squared_distance(
                        &(self.samples.get_row_as_vec(index_row)),
                        &(self.samples.get_row_as_vec(*other)),
                    )),
                })
            }
        }
        distances
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::linalg::naive::dense_matrix::*;

    #[test]
    fn fastpair_init() {
        let x: DenseMatrix<f64> = DenseMatrix::rand(10, 4);
        let fastpair = FastPair(&x);
        assert!(fastpair.is_ok());

        let result = fastpair.unwrap();
        let distances = *result.distances;
        let neighbours = *result.neighbours;
        let sparse_matrix = *(result.connectivity.unwrap());
        assert_eq!(10, neighbours.len());
        assert_eq!(10, distances.len());
        assert_eq!(10, sparse_matrix.shape().0);
        assert_eq!(10, sparse_matrix.shape().1);
    }

    #[test]
    fn fastpair_new() {
        // compute
        let x = DenseMatrix::from_2d_array(&[
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
        ]);
        let fastpair = FastPair(&x);
        assert!(fastpair.is_ok());

        // unwrap results
        let result = fastpair.unwrap();
        let neighbours = *result.neighbours;
        // let distances = *result.distances;
        let sparse_matrix = *(result.connectivity.unwrap());

        // sequence of indeces computed
        assert_eq!(
            neighbours,
            &[0, 4, 9, 3, 8, 7, 7, 7, 9, 9, 13, 12, 14, 14, 14]
        );

        // list of minimal pairwise dissimilarities
        let dissimilarities = vec!(
            (1, PairwiseDissimilarity { node: 1, neighbour: Some(9), distance: Some(0.030000000000000037) }),
            (10, PairwiseDissimilarity { node: 10, neighbour: Some(12), distance: Some(0.07000000000000003) }),
            (11, PairwiseDissimilarity { node: 11, neighbour: Some(14), distance: Some(0.18000000000000013) }),
            (12, PairwiseDissimilarity { node: 12, neighbour: Some(14), distance: Some(0.34000000000000086) }),
            (13, PairwiseDissimilarity { node: 13, neighbour: Some(14), distance: Some(1.6499999999999997) }),
            (14, PairwiseDissimilarity { node: 14, neighbour: Some(14), distance: Some(179769313486231570000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.0) }),
            (6, PairwiseDissimilarity { node: 6, neighbour: Some(7), distance: Some(0.18000000000000027) }),
            (0, PairwiseDissimilarity { node: 0, neighbour: Some(4), distance: Some(0.01999999999999995) }),
            (8, PairwiseDissimilarity { node: 8, neighbour: Some(9), distance: Some(0.3100000000000001) }),
            (2, PairwiseDissimilarity { node: 2, neighbour: Some(3), distance: Some(0.0600000000000001) }),
            (3, PairwiseDissimilarity { node: 3, neighbour: Some(8), distance: Some(0.08999999999999982) }),
            (7, PairwiseDissimilarity { node: 7, neighbour: Some(9), distance: Some(0.10999999999999982) }),
            (9, PairwiseDissimilarity { node: 9, neighbour: Some(13), distance: Some(8.69) }),
            (4, PairwiseDissimilarity { node: 4, neighbour: Some(7), distance: Some(0.050000000000000086) }),
            (5, PairwiseDissimilarity { node: 5, neighbour: Some(7), distance: Some(0.4900000000000002) })
        );

        let expected: HashMap<_, _> = dissimilarities.into_iter().collect();

        for i in 0..(x.shape().0 - 1) {
            let input_node = result.samples.get_row_as_vec(i);
            let input_neighbour: usize = expected.get(&i).unwrap().neighbour.unwrap();
            let distance = Euclidian::squared_distance(
                &input_node,
                &result.samples.get_row_as_vec(input_neighbour),
            );

            assert_eq!(i, expected.get(&i).unwrap().node);
            assert_eq!(
                input_neighbour,
                expected.get(&i).unwrap().neighbour.unwrap()
            );
            assert_eq!(distance, expected.get(&i).unwrap().distance.unwrap());
            assert_eq!(
                sparse_matrix.get(i, input_neighbour),
                expected.get(&i).unwrap().distance.unwrap()
            );
        }
    }

    #[test]
    fn fastpair_closest_pair() {
        let x = DenseMatrix::from_2d_array(&[
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
        ]);
        // compute
        let fastpair = FastPair(&x);
        assert!(fastpair.is_ok());

        let dissimilarity = fastpair.unwrap().closest_pair();
        let closest = PairwiseDissimilarity {
            node: 0,
            neighbour: Some(4),
            distance: Some(0.01999999999999995),
        };

        assert_eq!(closest, dissimilarity);
    }

    #[test]
    fn fastpair_distances() {
        let x = DenseMatrix::from_2d_array(&[
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
        ]);
        // compute
        let fastpair = FastPair(&x);
        assert!(fastpair.is_ok());

        let dissimilarities = fastpair.unwrap().distances_from(0);

        let mut min_dissimilarity = PairwiseDissimilarity {
            node: 0,
            neighbour: None,
            distance: Some(179769313486231570000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.0),
        };
        for p in dissimilarities.iter() {
            if p.distance.unwrap() < min_dissimilarity.distance.unwrap() {
                min_dissimilarity = p.clone()
            }
        }

        let closest = PairwiseDissimilarity {
            node: 0,
            neighbour: Some(4),
            distance: Some(0.01999999999999995),
        };

        assert_eq!(closest, min_dissimilarity);
    }
}
