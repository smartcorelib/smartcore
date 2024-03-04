/// # Hierarchical clustering
/// 
/// Implement hierarchical clustering methods:
/// * Agglomerative clustering (current)
/// * Bisecting K-Means (future)
/// * Fastcluster (future)
/// 

/* 
class AgglomerativeClustering():
    """
    Parameters
    ----------
    n_clusters : int or None, default=2
        The number of clusters to find. It must be ``None`` if
        ``distance_threshold`` is not ``None``.
    affinity : str or callable, default='euclidean'
        If linkage is "ward", only "euclidean" is accepted.
    linkage : {'ward',}, default='ward'
        Which linkage criterion to use. The linkage criterion determines which
        distance to use between sets of observation. The algorithm will merge
        the pairs of cluster that minimize this criterion.
        - 'ward' minimizes the variance of the clusters being merged.
    compute_distances : bool, default=False
        Computes distances between clusters even if `distance_threshold` is not
        used. This can be used to make dendrogram visualization, but introduces
        a computational and memory overhead.
    """

    def fit(X):
        # compute tree
        # <https://github.com/scikit-learn/scikit-learn/blob/02ebf9e68fe1fc7687d9e1047b9e465ae0fd945e/sklearn/cluster/_agglomerative.py#L172>
        parents, childern = ward_tree(X, ....)
        # compute clusters
        # <https://github.com/scikit-learn/scikit-learn/blob/70c495250fea7fa3c8c1a4631e6ddcddc9f22451/sklearn/cluster/_hierarchical_fast.pyx#L98>
        labels = _hierarchical.hc_get_heads(parents)
        # assign cluster numbers
        self.labels_ = np.searchsorted(np.unique(labels), labels)

*/

// implement ward tree
// use scipy.cluster.hierarchy.ward
// <https://github.com/scipy/scipy/blob/main/scipy/cluster/hierarchy.py#L738>
// use linkage
// <https://github.com/scipy/scipy/blob/main/scipy/cluster/hierarchy.py#L837>
// use nn_chain
// <https://github.com/scipy/scipy/blob/main/scipy/cluster/_hierarchy.pyx#L906>

// implement hc_get_heads


mod tests {
    // >>> from sklearn.cluster import AgglomerativeClustering
    // >>> import numpy as np
    // >>> X = np.array([[1, 2], [1, 4], [1, 0],
    // ...               [4, 2], [4, 4], [4, 0]])
    // >>> clustering = AgglomerativeClustering().fit(X)
    // >>> clustering
    // AgglomerativeClustering()
    // >>> clustering.labels_
    // array([1, 1, 1, 0, 0, 0])
}