use std::collections::HashMap;

use crate::math::num::RealNumber;
use crate::math::vector::RealNumberVector;

pub fn contingency_matrix<T: RealNumber>(
    labels_true: &Vec<T>,
    labels_pred: &Vec<T>,
) -> Vec<Vec<usize>> {
    let (classes, class_idx) = labels_true.unique();
    let (clusters, cluster_idx) = labels_pred.unique();

    let mut contingency_matrix = Vec::with_capacity(classes.len());

    for _ in 0..classes.len() {
        contingency_matrix.push(vec![0; clusters.len()]);
    }

    for i in 0..class_idx.len() {
        contingency_matrix[class_idx[i]][cluster_idx[i]] += 1;
    }

    contingency_matrix
}

pub fn entropy<T: RealNumber>(data: &Vec<T>) -> Option<T> {
    let mut bincounts = HashMap::with_capacity(data.len());

    for e in data.iter() {
        let k = e.to_i64().unwrap();
        bincounts.insert(k, bincounts.get(&k).unwrap_or(&0) + 1);
    }

    let mut entropy = T::zero();
    let sum = T::from_usize(bincounts.values().sum()).unwrap();

    for &c in bincounts.values() {
        if c > 0 {
            let pi = T::from_usize(c).unwrap();
            entropy = entropy - (pi / sum) * (pi.ln() - sum.ln());
        }
    }

    Some(entropy)
}

pub fn mutual_info_score<T: RealNumber>(contingency: &Vec<Vec<usize>>) -> T {
    let mut contingency_sum = 0;
    let mut pi = vec![0; contingency.len()];
    let mut pj = vec![0; contingency[0].len()];
    let (mut nzx, mut nzy, mut nz_val) = (Vec::new(), Vec::new(), Vec::new());

    for r in 0..contingency.len() {
        for c in 0..contingency[0].len() {
            contingency_sum += contingency[r][c];
            pi[r] += contingency[r][c];
            pj[c] += contingency[r][c];
            if contingency[r][c] > 0 {
                nzx.push(r);
                nzy.push(c);
                nz_val.push(contingency[r][c]);
            }
        }
    }

    let contingency_sum = T::from_usize(contingency_sum).unwrap();
    let contingency_sum_ln = contingency_sum.ln();
    let pi_sum_l = T::from_usize(pi.iter().sum()).unwrap().ln();
    let pj_sum_l = T::from_usize(pj.iter().sum()).unwrap().ln();

    let log_contingency_nm: Vec<T> = nz_val
        .iter()
        .map(|v| T::from_usize(*v).unwrap().ln())
        .collect();
    let contingency_nm: Vec<T> = nz_val
        .iter()
        .map(|v| T::from_usize(*v).unwrap() / contingency_sum)
        .collect();
    let outer: Vec<usize> = nzx
        .iter()
        .zip(nzy.iter())
        .map(|(&x, &y)| pi[x] * pj[y])
        .collect();
    let log_outer: Vec<T> = outer
        .iter()
        .map(|&o| -T::from_usize(o).unwrap().ln() + pi_sum_l + pj_sum_l)
        .collect();

    let mut result = T::zero();

    for i in 0..log_outer.len() {
        result = result
            + ((contingency_nm[i] * (log_contingency_nm[i] - contingency_sum_ln))
                + contingency_nm[i] * log_outer[i])
    }

    result.max(T::zero())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contingency_matrix_test() {
        let v1 = vec![0.0, 0.0, 1.0, 1.0, 2.0, 0.0, 4.0];
        let v2 = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];

        println!("{:?}", contingency_matrix(&v1, &v2));
    }

    #[test]
    fn entropy_test() {
        let v1 = vec![0.0, 0.0, 1.0, 1.0, 2.0, 0.0, 4.0];

        println!("{:?}", entropy(&v1));
    }

    #[test]
    fn mutual_info_score_test() {
        let v1 = vec![0.0, 0.0, 1.0, 1.0, 2.0, 0.0, 4.0];
        let v2 = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let s: f32 = mutual_info_score(&contingency_matrix(&v1, &v2));

        println!("{}", s);
    }
}
