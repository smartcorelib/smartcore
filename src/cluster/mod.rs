//! # Clustering
//!
//! Clustering is the type of unsupervised learning where you divide the population or data points into a number of groups such that data points in the same groups
//! are more similar to other data points in the same group than those in other groups. In simple words, the aim is to segregate groups with similar traits and assign them into clusters.

pub mod dbscan;
/// An iterative clustering algorithm that aims to find local maxima in each iteration.
pub mod kmeans;
