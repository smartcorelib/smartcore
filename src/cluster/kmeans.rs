extern crate rand;

use rand::Rng;

use crate::linalg::Matrix;
use crate::algorithm::neighbour::bbd_tree::BBDTree;

#[derive(Debug)]
pub struct KMeans {    
    k: usize,
    y: Vec<usize>,
    size: Vec<usize>,
    distortion: f64,
    centroids: Vec<Vec<f64>>
}

#[derive(Debug, Clone)]
pub struct KMeansParameters {  
    pub max_iter: usize
}

impl Default for KMeansParameters {
    fn default() -> Self { 
        KMeansParameters {
            max_iter: 100
        }
     }
}

impl KMeans{
    pub fn new<M: Matrix>(data: &M, k: usize, parameters: KMeansParameters) -> KMeans {

        let bbd = BBDTree::new(data);

        if k < 2 {
            panic!("Invalid number of clusters: {}", k);
        }

        if parameters.max_iter <= 0 {
            panic!("Invalid maximum number of iterations: {}", parameters.max_iter);
        }

        let (n, d) = data.shape();
                
        let mut distortion = std::f64::MAX;
        let mut y = KMeans::kmeans_plus_plus(data, k);
        let mut size = vec![0; k];
        let mut centroids = vec![vec![0f64; d]; k];

        for i in 0..n {
            size[y[i]] += 1;
        }

        for i in 0..n {
            for j in 0..d {
                centroids[y[i]][j] += data.get(i, j);
            }
        }

        for i in 0..k {
            for j in 0..d {
                centroids[i][j] /= size[i] as f64;
            }
        }        

        let mut sums = vec![vec![0f64; d]; k];
        for _ in 1..= parameters.max_iter {
            let dist = bbd.clustering(&centroids, &mut sums, &mut size, &mut y);
            for i in 0..k {
                if size[i] > 0 {
                    for j in 0..d {
                        centroids[i][j] = sums[i][j] as f64 / size[i] as f64;
                    }
                }
            }

            if distortion <= dist {
                break;
            } else {
                distortion = dist;
            }
            
        }        

        KMeans{
            k: k,
            y: y,
            size: size,
            distortion: distortion,
            centroids: centroids
        }
    }

    pub fn predict<M: Matrix>(&self, x: &M) -> M::RowVector {
        let (n, _) = x.shape();        
        let mut result = M::zeros(1, n); 

        for i in 0..n {

            let mut min_dist = std::f64::MAX;
            let mut best_cluster = 0;

            for j in 0..self.k {
                let dist = KMeans::squared_distance(&x.get_row_as_vec(i), &self.centroids[j]);                
                if dist < min_dist {
                    min_dist = dist;
                    best_cluster = j;
                }
            }            
            result.set(0, i, best_cluster as f64);
        }

        result.to_row_vector()
    }

    fn kmeans_plus_plus<M: Matrix>(data: &M, k: usize) -> Vec<usize>{
        let mut rng = rand::thread_rng();        
        let (n, _) = data.shape();
        let mut y = vec![0; n];
        let mut centroid = data.get_row_as_vec(rng.gen_range(0, n));

        let mut d = vec![std::f64::MAX; n];
        
        // pick the next center
        for j in 1..k {
            // Loop over the samples and compare them to the most recent center.  Store
            // the distance from each sample to its closest center in scores.
            for i in 0..n {
                // compute the distance between this sample and the current center
                let dist = KMeans::squared_distance(&data.get_row_as_vec(i), &centroid);
                
                if dist < d[i] {
                    d[i] = dist;
                    y[i] = j - 1;
                }
            }

            let sum: f64 = d.iter().sum();
            let cutoff = rng.gen::<f64>() * sum;
            let mut cost = 0f64;
            let index = 0;
            for index in 0..n {
                cost += d[index];
                if cost >= cutoff {
                    break;
                }
            }

            centroid = data.get_row_as_vec(index);
        }

        for i in 0..n {
            // compute the distance between this sample and the current center
            let dist = KMeans::squared_distance(&data.get_row_as_vec(i), &centroid);            
            
            if dist < d[i] {
                d[i] = dist;
                y[i] = k - 1;
            }
        }

        y
    }

    fn squared_distance(x: &Vec<f64>,y: &Vec<f64>) -> f64 {
        if x.len() != y.len() {
            panic!("Input vector sizes are different.");
        }

        let mut sum = 0f64;
        for i in 0..x.len() {
            sum += (x[i] - y[i]).powf(2.);
        }

        return sum;
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

        let kmeans = KMeans::new(&x, 2, Default::default());

        let y = kmeans.predict(&x);

        for i in 0..y.len() {
            assert_eq!(y[i] as usize, kmeans.y[i]);
        }        
        
    }
    
}