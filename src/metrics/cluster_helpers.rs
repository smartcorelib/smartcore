#![allow(clippy::ptr_arg)]
use std::collections::HashMap;

use crate::linalg::basic::arrays::Array1;
use crate::numbers::basenum::Number;

pub fn contingency_matrix<T: Number + Ord, V: Array1<T>>(
    labels_true: &V,
    labels_pred: &V,
) -> Vec<Vec<usize>> {
    let (classes, class_idx) = labels_true.unique_with_indices();
    let (clusters, cluster_idx) = labels_pred.unique_with_indices();

    let mut contingency_matrix = Vec::with_capacity(classes.len());

    for _ in 0..classes.len() {
        contingency_matrix.push(vec![0; clusters.len()]);
    }

    for i in 0..class_idx.len() {
        contingency_matrix[class_idx[i]][cluster_idx[i]] += 1;
    }

    contingency_matrix
}

pub fn entropy<T: Number, V: Array1<T>>(data: &V) -> Option<f64> {
    let mut bincounts = HashMap::with_capacity(data.shape());

    for e in data.iterator(0) {
        let k = e.to_i64().unwrap();
        bincounts.insert(k, bincounts.get(&k).unwrap_or(&0) + 1);
    }

    let mut entropy = 0f64;
    let sum: i64 = bincounts.values().sum();

    for &c in bincounts.values() {
        if c > 0 {
            let pi = c as f64;
            entropy -= (pi / sum as f64) * (pi.ln() - (sum as f64).ln());
        }
    }

    Some(entropy)
}

pub fn mutual_info_score(contingency: &[Vec<usize>]) -> f64 {
    let mut contingency_sum = 0;
    let mut pi = vec![0; contingency.len()];
    let mut pj = vec![0; contingency[0].len()];
    let (mut nzx, mut nzy, mut nz_val) = (Vec::new(), Vec::new(), Vec::new());

    for r in 0..contingency.len() {
        for (c, pj_c) in pj.iter_mut().enumerate().take(contingency[0].len()) {
            contingency_sum += contingency[r][c];
            pi[r] += contingency[r][c];
            *pj_c += contingency[r][c];
            if contingency[r][c] > 0 {
                nzx.push(r);
                nzy.push(c);
                nz_val.push(contingency[r][c]);
            }
        }
    }

    let contingency_sum = contingency_sum as f64;
    let contingency_sum_ln = contingency_sum.ln();
    let pi_sum: usize = pi.iter().sum();
    let pj_sum: usize = pj.iter().sum();
    let pi_sum_l = (pi_sum as f64).ln();
    let pj_sum_l = (pj_sum as f64).ln();

    let log_contingency_nm: Vec<f64> = nz_val.iter().map(|v| (*v as f64).ln()).collect();
    let contingency_nm: Vec<f64> = nz_val
        .iter()
        .map(|v| (*v as f64) / contingency_sum)
        .collect();
    let outer: Vec<usize> = nzx
        .iter()
        .zip(nzy.iter())
        .map(|(&x, &y)| pi[x] * pj[y])
        .collect();
    let log_outer: Vec<f64> = outer
        .iter()
        .map(|&o| -(o as f64).ln() + pi_sum_l + pj_sum_l)
        .collect();

    let mut result = 0f64;

    for i in 0..log_outer.len() {
        result += (contingency_nm[i] * (log_contingency_nm[i] - contingency_sum_ln))
            + contingency_nm[i] * log_outer[i]
    }

    result.max(0f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contingency_matrix_test() {
        let v1 = vec![0, 0, 1, 1, 2, 0, 4];
        let v2 = vec![1, 0, 0, 0, 0, 1, 0];

        assert_eq!(
            vec!(vec!(1, 2), vec!(2, 0), vec!(1, 0), vec!(1, 0)),
            contingency_matrix(&v1, &v2)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn entropy_test() {
        let v1 = vec![0, 0, 1, 1, 2, 0, 4];

        assert!((1.2770 - entropy(&v1).unwrap()).abs() < 1e-4);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mutual_info_score_test() {
        let v1 = vec![0, 0, 1, 1, 2, 0, 4];
        let v2 = vec![1, 0, 0, 0, 0, 1, 0];
        let s = mutual_info_score(&contingency_matrix(&v1, &v2));

        assert!((0.3254 - s).abs() < 1e-4);
    }
}
