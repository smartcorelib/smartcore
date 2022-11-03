// /// SVR grid search parameters
// #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
// #[derive(Debug, Clone)]
// pub struct SVRSearchParameters<T: Number + RealNumber, M: Matrix<T>, K: Kernel<T, M::RowVector>> {
//     /// Epsilon in the epsilon-SVR model.
//     pub eps: Vec<T>,
//     /// Regularization parameter.
//     pub c: Vec<T>,
//     /// Tolerance for stopping eps.
//     pub tol: Vec<T>,
//     /// The kernel function.
//     pub kernel: Vec<K>,
//     /// Unused parameter.
//     m: PhantomData<M>,
// }

// /// SVR grid search iterator
// pub struct SVRSearchParametersIterator<T: Number + RealNumber, M: Matrix<T>, K: Kernel<T, M::RowVector>> {
//     svr_search_parameters: SVRSearchParameters<T, M, K>,
//     current_eps: usize,
//     current_c: usize,
//     current_tol: usize,
//     current_kernel: usize,
// }

// impl<T: Number + RealNumber, M: Matrix<T>, K: Kernel<T, M::RowVector>> IntoIterator
//     for SVRSearchParameters<T, M, K>
// {
//     type Item = SVRParameters<T, M, K>;
//     type IntoIter = SVRSearchParametersIterator<T, M, K>;

//     fn into_iter(self) -> Self::IntoIter {
//         SVRSearchParametersIterator {
//             svr_search_parameters: self,
//             current_eps: 0,
//             current_c: 0,
//             current_tol: 0,
//             current_kernel: 0,
//         }
//     }
// }

// impl<T: Number + RealNumber, M: Matrix<T>, K: Kernel<T, M::RowVector>> Iterator
//     for SVRSearchParametersIterator<T, M, K>
// {
//     type Item = SVRParameters<T, M, K>;

//     fn next(&mut self) -> Option<Self::Item> {
//         if self.current_eps == self.svr_search_parameters.eps.len()
//             && self.current_c == self.svr_search_parameters.c.len()
//             && self.current_tol == self.svr_search_parameters.tol.len()
//             && self.current_kernel == self.svr_search_parameters.kernel.len()
//         {
//             return None;
//         }

//         let next = SVRParameters::<T, M, K> {
//             eps: self.svr_search_parameters.eps[self.current_eps],
//             c: self.svr_search_parameters.c[self.current_c],
//             tol: self.svr_search_parameters.tol[self.current_tol],
//             kernel: self.svr_search_parameters.kernel[self.current_kernel].clone(),
//             m: PhantomData,
//         };

//         if self.current_eps + 1 < self.svr_search_parameters.eps.len() {
//             self.current_eps += 1;
//         } else if self.current_c + 1 < self.svr_search_parameters.c.len() {
//             self.current_eps = 0;
//             self.current_c += 1;
//         } else if self.current_tol + 1 < self.svr_search_parameters.tol.len() {
//             self.current_eps = 0;
//             self.current_c = 0;
//             self.current_tol += 1;
//         } else if self.current_kernel + 1 < self.svr_search_parameters.kernel.len() {
//             self.current_eps = 0;
//             self.current_c = 0;
//             self.current_tol = 0;
//             self.current_kernel += 1;
//         } else {
//             self.current_eps += 1;
//             self.current_c += 1;
//             self.current_tol += 1;
//             self.current_kernel += 1;
//         }

//         Some(next)
//     }
// }

// impl<T: Number + RealNumber, M: Matrix<T>> Default for SVRSearchParameters<T, M, LinearKernel> {
//     fn default() -> Self {
//         let default_params: SVRParameters<T, M, LinearKernel> = SVRParameters::default();

//         SVRSearchParameters {
//             eps: vec![default_params.eps],
//             c: vec![default_params.c],
//             tol: vec![default_params.tol],
//             kernel: vec![default_params.kernel],
//             m: PhantomData,
//         }
//     }
// }

// #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
// #[derive(Debug)]
// #[cfg_attr(
//     feature = "serde",
//     serde(bound(
//         serialize = "M::RowVector: Serialize, K: Serialize, T: Serialize",
//         deserialize = "M::RowVector: Deserialize<'de>, K: Deserialize<'de>, T: Deserialize<'de>",
//     ))
// )]