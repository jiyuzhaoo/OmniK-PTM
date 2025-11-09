import numpy as np
from scipy.interpolate import splrep, BSpline
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from dtaidistance import dtw

class EfficientAdaptiveFunctionalSampler:
    def __init__(self, data, n_samples_to_keep, max_k=10, smooth_factor=1.0, subset_size=1500, random_state=None):
        if n_samples_to_keep >= data.shape[0]:
            raise ValueError("n_samples_to_keep must be smaller than total sample size.")
        self.data = data
        self.n = data.shape[0]
        self.dim = data.shape[1]
        self.n_samples_to_keep = n_samples_to_keep
        self.max_k = min(max_k, self.n - 1)
        self.smooth_factor = smooth_factor
        self.subset_size = min(subset_size, self.n)
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.normalized_data = None
        self.functions = []
        self.distance_matrix = None

    def _find_medoids_from_labels(self, distance_matrix, labels):
        medoid_indices = []
        unique_labels = sorted(list(np.unique(labels)))
        for label in unique_labels:
            indices_in_cluster = np.where(labels == label)[0]
            if len(indices_in_cluster) == 0:
                continue
            cluster_dist_matrix = distance_matrix[np.ix_(indices_in_cluster, indices_in_cluster)]
            sum_of_distances = cluster_dist_matrix.sum(axis=1)
            local_medoid_idx = np.argmin(sum_of_distances)
            global_medoid_idx = indices_in_cluster[local_medoid_idx]
            medoid_indices.append(global_medoid_idx)
        return np.array(medoid_indices)

    def _normalize_data(self):
        scaler = MinMaxScaler()
        self.normalized_data = scaler.fit_transform(self.data)

    def _fit_b_spline(self, x, y):
        tck = splrep(x, y, s=self.smooth_factor, k=3)
        return BSpline(tck[0], tck[1], tck[2])

    def _fit_all_functions(self):
        x = np.linspace(0, 1, self.dim)
        for i in range(self.n):
            y = self.normalized_data[i]
            spline_func = self._fit_b_spline(x, y)
            self.functions.append(spline_func)

    def _calculate_distance_matrix_for_subset(self, indices):
        x_eval = np.linspace(0, 1, self.dim * 2)
        func_evals = np.array([self.functions[i](x_eval) for i in indices])
        return dtw.distance_matrix_fast(func_evals.astype(np.double))

    def _find_optimal_k(self, distance_matrix):
        current_max_k = min(self.max_k, distance_matrix.shape[0] - 1)
        best_score = -1
        best_k = 2
        if current_max_k < 2:
            print("[WARNING] Not enough samples in subset to perform clustering. Using k=1.")
            return 1
        for k in range(2, current_max_k + 1):
            agg_cluster = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='average')
            labels = agg_cluster.fit_predict(distance_matrix)
            if len(np.unique(labels)) < 2:
                continue
            score = silhouette_score(distance_matrix, labels, metric='precomputed')
            if score > best_score:
                best_score = score
                best_k = k
        return best_k

    def _stratified_sample(self, labels, medoid_indices):
        k = len(medoid_indices)
        selected_indices = []
        cluster_labels, cluster_counts = np.unique(labels, return_counts=True)
        proportions = cluster_counts / self.n
        ideal_samples = proportions * self.n_samples_to_keep
        n_to_sample_per_cluster = ideal_samples.astype(int)
        if self.n_samples_to_keep >= k:
            n_to_sample_per_cluster = np.maximum(n_to_sample_per_cluster, 1)
        remainder = self.n_samples_to_keep - np.sum(n_to_sample_per_cluster)
        if remainder > 0:
            fractional_parts = ideal_samples - n_to_sample_per_cluster
            top_indices = np.argsort(fractional_parts)[-remainder:]
            n_to_sample_per_cluster[top_indices] += 1
        n_to_sample_per_cluster = n_to_sample_per_cluster[:len(cluster_labels)]
        final_sum = np.sum(n_to_sample_per_cluster)
        if final_sum != self.n_samples_to_keep:
            diff = self.n_samples_to_keep - final_sum
            adjust_indices = np.argsort(n_to_sample_per_cluster)
            if diff > 0:
                n_to_sample_per_cluster[adjust_indices[-1]] += diff
            else:
                n_to_sample_per_cluster[adjust_indices[-1]] -= abs(diff)
        x_eval = np.linspace(0, 1, self.dim * 2)
        medoid_evals = np.array([self.functions[i](x_eval) for i in medoid_indices])
        all_evals = np.array([f(x_eval) for f in self.functions])
        distances_to_medoid = np.full(self.n, np.inf)
        for i in range(self.n):
            cluster_idx = labels[i]
            dist = dtw.distance(all_evals[i], medoid_evals[cluster_idx])
            distances_to_medoid[i] = dist
        for i, label in enumerate(cluster_labels):
            indices_in_cluster = np.where(labels == label)[0]
            distances_in_cluster = distances_to_medoid[indices_in_cluster]
            n_to_sample = n_to_sample_per_cluster[i]
            if n_to_sample == 0:
                continue
            sorted_local_indices = np.argsort(distances_in_cluster)
            selected_local_indices = sorted_local_indices[:n_to_sample]
            selected_global_indices = indices_in_cluster[selected_local_indices]
            selected_indices.extend(selected_global_indices)
        return np.array(selected_indices)

    def run(self):
        self._normalize_data()
        self._fit_all_functions()
        all_indices = np.arange(self.n)
        self.rng.shuffle(all_indices)
        subset_indices = all_indices[:self.subset_size]
        subset_dist_matrix = self._calculate_distance_matrix_for_subset(subset_indices)
        optimal_k = self._find_optimal_k(subset_dist_matrix)
        if optimal_k <= 1:
            print("[WARNING] Optimal k <= 1. Performing random undersampling.")
            self.rng.shuffle(all_indices)
            return all_indices[:self.n_samples_to_keep]
        agg_cluster = AgglomerativeClustering(n_clusters=optimal_k, metric='precomputed', linkage='average')
        subset_labels = agg_cluster.fit_predict(subset_dist_matrix)
        medoid_local_indices = self._find_medoids_from_labels(subset_dist_matrix, subset_labels)
        medoid_global_indices = subset_indices[medoid_local_indices]
        x_eval = np.linspace(0, 1, self.dim * 2)
        medoid_evals = np.array([self.functions[i](x_eval) for i in medoid_global_indices])
        all_evals = np.array([f(x_eval) for f in self.functions])
        dist_to_medoids = np.array(dtw.distance_matrix(all_evals.astype(np.double), medoid_evals.astype(np.double)))
        if dist_to_medoids.ndim == 2 and dist_to_medoids.shape[1] > 1:
            all_labels = np.argmin(dist_to_medoids, axis=1)
        else:
            print("[WARNING] Only one cluster detected. Assigning all samples to cluster 0.")
            all_labels = np.zeros(self.n, dtype=int)
        selected_indices = self._stratified_sample(all_labels, medoid_global_indices)
        return selected_indices

def EfficientAdaptiveClusterSampler(sampling_strategy, X, y, max_k=10, subset_size=1500, smooth_factor=1.0):
    if len(sampling_strategy) > 1:
        raise NotImplementedError("Only single minority class undersampling is supported.")
    minority_label, n_samples_to_keep = list(sampling_strategy.items())[0]
    minority_indices = np.where(y == minority_label)[0]
    majority_indices = np.where(y != minority_label)[0]
    X_minority = X[minority_indices]
    sampler = EfficientAdaptiveFunctionalSampler(
        data=X_minority,
        n_samples_to_keep=n_samples_to_keep,
        max_k=max_k,
        smooth_factor=smooth_factor,
        subset_size=subset_size
    )
    selected_local_indices = sampler.run()
    selected_original_indices = minority_indices[selected_local_indices]
    final_indices = np.concatenate([selected_original_indices, majority_indices])
    x_resampled = X[final_indices]
    y_resampled = y[final_indices]
    return x_resampled, y_resampled, final_indices
