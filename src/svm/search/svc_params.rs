// /// SVC grid search parameters
// #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
// #[derive(Debug, Clone)]
// pub struct SVCSearchParameters<
//     TX: Number + RealNumber,
//     TY: Number + Ord,
//     X: Array2<TX>,
//     Y: Array1<TY>,
//     K: Kernel,
// > {
//     #[cfg_attr(feature = "serde", serde(default))]
//     /// Number of epochs.
//     pub epoch: Vec<usize>,
//     #[cfg_attr(feature = "serde", serde(default))]
//     /// Regularization parameter.
//     pub c: Vec<TX>,
//     #[cfg_attr(feature = "serde", serde(default))]
//     /// Tolerance for stopping epoch.
//     pub tol: Vec<TX>,
//     #[cfg_attr(feature = "serde", serde(default))]
//     /// The kernel function.
//     pub kernel: Vec<K>,
//     #[cfg_attr(feature = "serde", serde(default))]
//     /// Unused parameter.
//     m: PhantomData<(X, Y, TY)>,
//     #[cfg_attr(feature = "serde", serde(default))]
//     /// Controls the pseudo random number generation for shuffling the data for probability estimates
//     seed: Vec<Option<u64>>,
// }

// /// SVC grid search iterator
// pub struct SVCSearchParametersIterator<
//     TX: Number + RealNumber,
//     TY: Number + Ord,
//     X: Array2<TX>,
//     Y: Array1<TY>,
//     K: Kernel,
// > {
//     svc_search_parameters: SVCSearchParameters<TX, TY, X, Y, K>,
//     current_epoch: usize,
//     current_c: usize,
//     current_tol: usize,
//     current_kernel: usize,
//     current_seed: usize,
// }

// impl<TX: Number + RealNumber, TY: Number + Ord, X: Array2<TX>, Y: Array1<TY>, K: Kernel>
//     IntoIterator for SVCSearchParameters<TX, TY, X, Y, K>
// {
//     type Item = SVCParameters<'a, TX, TY, X, Y>;
//     type IntoIter = SVCSearchParametersIterator<TX, TY, X, Y, K>;

//     fn into_iter(self) -> Self::IntoIter {
//         SVCSearchParametersIterator {
//             svc_search_parameters: self,
//             current_epoch: 0,
//             current_c: 0,
//             current_tol: 0,
//             current_kernel: 0,
//             current_seed: 0,
//         }
//     }
// }

// impl<TX: Number + RealNumber, TY: Number + Ord, X: Array2<TX>, Y: Array1<TY>, K: Kernel>
//     Iterator for SVCSearchParametersIterator<TX, TY, X, Y, K>
// {
//     type Item = SVCParameters<TX, TY, X, Y>;

//     fn next(&mut self) -> Option<Self::Item> {
//         if self.current_epoch == self.svc_search_parameters.epoch.len()
//             && self.current_c == self.svc_search_parameters.c.len()
//             && self.current_tol == self.svc_search_parameters.tol.len()
//             && self.current_kernel == self.svc_search_parameters.kernel.len()
//             && self.current_seed == self.svc_search_parameters.seed.len()
//         {
//             return None;
//         }

//         let next = SVCParameters {
//             epoch: self.svc_search_parameters.epoch[self.current_epoch],
//             c: self.svc_search_parameters.c[self.current_c],
//             tol: self.svc_search_parameters.tol[self.current_tol],
//             kernel: self.svc_search_parameters.kernel[self.current_kernel].clone(),
//             m: PhantomData,
//             seed: self.svc_search_parameters.seed[self.current_seed],
//         };

//         if self.current_epoch + 1 < self.svc_search_parameters.epoch.len() {
//             self.current_epoch += 1;
//         } else if self.current_c + 1 < self.svc_search_parameters.c.len() {
//             self.current_epoch = 0;
//             self.current_c += 1;
//         } else if self.current_tol + 1 < self.svc_search_parameters.tol.len() {
//             self.current_epoch = 0;
//             self.current_c = 0;
//             self.current_tol += 1;
//         } else if self.current_kernel + 1 < self.svc_search_parameters.kernel.len() {
//             self.current_epoch = 0;
//             self.current_c = 0;
//             self.current_tol = 0;
//             self.current_kernel += 1;
//         } else if self.current_seed + 1 < self.svc_search_parameters.seed.len() {
//             self.current_epoch = 0;
//             self.current_c = 0;
//             self.current_tol = 0;
//             self.current_kernel = 0;
//             self.current_seed += 1;
//         } else {
//             self.current_epoch += 1;
//             self.current_c += 1;
//             self.current_tol += 1;
//             self.current_kernel += 1;
//             self.current_seed += 1;
//         }

//         Some(next)
//     }
// }

// impl<TX: Number + RealNumber, TY: Number + Ord, X: Array2<TX>, Y: Array1<TY>, K: Kernel> Default
//     for SVCSearchParameters<TX, TY, X, Y, K>
// {
//     fn default() -> Self {
//         let default_params: SVCParameters<TX, TY, X, Y> = SVCParameters::default();

//         SVCSearchParameters {
//             epoch: vec![default_params.epoch],
//             c: vec![default_params.c],
//             tol: vec![default_params.tol],
//             kernel: vec![default_params.kernel],
//             m: PhantomData,
//             seed: vec![default_params.seed],
//         }
//     }
// }


// #[cfg(test)]
// mod tests {
//     use num::ToPrimitive;

//     use super::*;
//     use crate::linalg::basic::matrix::DenseMatrix;
//     use crate::metrics::accuracy;
//     #[cfg(feature = "serde")]
//     use crate::svm::*;

//     #[test]
//     fn search_parameters() {
//         let parameters: SVCSearchParameters<f64, DenseMatrix<f64>, LinearKernel> =
//             SVCSearchParameters {
//                 epoch: vec![10, 100],
//                 kernel: vec![LinearKernel {}],
//                 ..Default::default()
//             };
//         let mut iter = parameters.into_iter();
//         let next = iter.next().unwrap();
//         assert_eq!(next.epoch, 10);
//         assert_eq!(next.kernel, LinearKernel {});
//         let next = iter.next().unwrap();
//         assert_eq!(next.epoch, 100);
//         assert_eq!(next.kernel, LinearKernel {});
//         assert!(iter.next().is_none());
//     }

//     #[test]
//     fn search_parameters() {
//         let parameters: SVCSearchParameters<f64, DenseMatrix<f64>, LinearKernel> =
//             SVCSearchParameters {
//                 epoch: vec![10, 100],
//                 kernel: vec![LinearKernel {}],
//                 ..Default::default()
//             };
//         let mut iter = parameters.into_iter();
//         let next = iter.next().unwrap();
//         assert_eq!(next.epoch, 10);
//         assert_eq!(next.kernel, LinearKernel {});
//         let next = iter.next().unwrap();
//         assert_eq!(next.epoch, 100);
//         assert_eq!(next.kernel, LinearKernel {});
//         assert!(iter.next().is_none());
//     }
// }
