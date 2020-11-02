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
    };
    init.new();
    Ok(init)
}

///
/// FastPair
///
/// Python implementation:
/// <https://github.com/carsonfarmer/fastpair/blob/b8b4d3000ab6f795a878936667eee1b557bf353d/fastpair/base.py>
///
#[derive(Debug, Clone)]
pub struct _FastPair<'a, T: RealNumber, M: Matrix<T>> {
    /// initial matrix
    samples: &'a M,
    /// closest pair hashmap:
    distances: Box<HashMap<usize, PairwiseDissimilarity<T>>>,
    /// conga line used to keep track of the closest pair
    neighbours: Box<Vec<usize>>,
}

impl<'a, T: RealNumber, M: Matrix<T>> _FastPair<'a, T, M> {
    ///
    /// Initialise `FastPair` by passing a `Matrix`
    ///
    fn new(&mut self) {
        // basic measures
        let len = self.samples.shape().0;
        let max_index = ((self.samples.shape().0) - 1);

        // Store all closest neighbors
        let _distances = Box::new(HashMap::with_capacity(len));
        let _neighbours = Box::new(Vec::with_capacity(len));

        let mut distances = *_distances;
        let mut neighbours = *_neighbours;

        // fill neighbours with -1 values
        neighbours.extend(iter::repeat(0).take(len));

        println!("Neighbours {:?}", neighbours);

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

            println!(
                "Init Neighbour {:?} {:?}",
                index_row_i,
                distances.get(&index_row_i).unwrap()
            );

            // start looking for the neighbour in the second element
            let mut index_closest = index_row_i + 1; // closest neighbour index
            let mut nbd: Option<T> = Some(T::max_value()); // init neighbour distance
            for index_row_j in (index_row_i + 1)..len {
                println!("here J {:?}", index_row_j);
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
                println!("Euclidian distance: {:?}", d);
                if d < nbd.unwrap() {
                    println!("Assign new distance");
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
            println!(
                "Dissimilarities {:?} -> {:?}",
                index_row_i,
                distances.get(&index_row_i)
            );

            // update conga line
            println!("Current conga {:?}", neighbours);
            if index_closest != len {
                // self.points[nbr] = self.points[i + 1]
                // self.points[i + 1] = self.neighbors[self.points[i]].neigh
                println!("LENGTH: {:?}", neighbours.len());
                neighbours[index_closest] = neighbours[index_row_i + 1];
                neighbours[index_row_i + 1] = index_closest;
            }
            println!("New conga {:?}", neighbours);
        }
        // No more neighbors, terminate conga line.
        // Last person on the line has no neigbors
        println!("Assign last node {:?}", (len - 1));
        distances.get_mut(&max_index).unwrap().neighbour = Some(max_index);

        distances.get_mut(&(len - 1)).unwrap().distance = Some(T::max_value());
        println!("AFTER LOOP");

        self.distances = Box::new(distances);
        self.neighbours = Box::new(neighbours);
    }

    // // Find and update nearest neighbor of a given point.
    // fn find_neighbour(&mut self, index_row: usize) -> Option<PairwiseDissimilarity<T>> {
    //     // Find first point unequal to `node` itself
    //     let mut first_nbr = 0;
    //     if index_row == (*self.nodes)[first_nbr] {
    //         first_nbr = 1;
    //     }
    //     self.neighbours.get_mut(&node).unwrap().neighbour = Some((*self.nodes)[first_nbr]);
    //     self.neighbours.get_mut(&node).unwrap().distance = Some(Euclidian::squared_distance(
    //         &(self.samples.get_row_as_vec(node)),
    //         &(self
    //             .samples
    //             .get_row_as_vec(self.neighbours[&node].neighbour.unwrap())),
    //     ));
    //     // Now test whether each other point is closer
    //     for &q in self.nodes[first_nbr + 1..(self.samples.shape().0 - 1)]
    //         .iter()
    //         .clone()
    //     {
    //         if node != q {
    //             let d = Euclidian::squared_distance(
    //                 &(self.samples.get_row_as_vec(node)),
    //                 &(self.samples.get_row_as_vec(q)),
    //             );
    //             if d < self.neighbours.get_mut(&node).unwrap().distance.unwrap() {
    //                 self.neighbours.get_mut(&node).unwrap().distance = Some(d);
    //                 self.neighbours.get_mut(&node).unwrap().neighbour = Some(q);
    //             }
    //         }
    //     }
    //     let result = self.neighbours.get(&node);
    //     let value = *(result.unwrap());
    //     Some(value.clone())
    // }

    // // Compute distances from input to all other points in data-structure.
    // // input is the row index of the sample matrix
    // fn distances_from(&self, row: usize) -> Vec<PairwiseDissimilarity<'a, T>> {
    //     let mut distances = Vec::<PairwiseDissimilarity<T>>::with_capacity(self.samples.shape().0);
    //     for other in self.nodes.iter() {
    //         if row != *other {
    //             distances.push(PairwiseDissimilarity {
    //                 node: row,
    //                 neighbour: Some(*other),
    //                 distance: Some(Euclidian::squared_distance(
    //                     &(self.samples.get_row_as_vec(row)),
    //                     &(self.samples.get_row_as_vec(*other)),
    //                 )),
    //             })
    //         }
    //     }
    //     distances
    // }
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
        assert_eq!(10, neighbours.len());
        assert_eq!(10, distances.len());

        println!("DISTANCES {:?}", distances)
    }

    #[test]
    fn fastpair_check() {
        // input
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

        // unwrap results
        let result = fastpair.unwrap();
        let neighbours = *result.neighbours;
        let distances = *result.distances;

        println!("RESULTS neighbours: {:?}", &neighbours);
        println!("RESULTS distances: {:?}", &distances);

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
        }
    }
}
