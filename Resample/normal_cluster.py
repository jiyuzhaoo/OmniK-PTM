import numpy as np


class NomalClustering:
    def __init__(self, data, k):
        self.data = data
        self.k = k
        self.n = data.shape[0]
        self.dim = data.shape[1]
        self.cluster_labels = None
        self.distances_to_centroid = None
        self.cov_matrix = np.cov(self.data, rowvar=False)
        self.regularization_term = 1e-5 * np.eye(self.cov_matrix.shape[0])

    def calculate(self, data1, data2):
        return np.linalg.norm(data1 - data2)

    def mahalanobis_distance(self, data1, data2):
        inv_cov_matrix = np.linalg.inv(self.cov_matrix)
        diff = data1 - data2
        dis = np.sqrt(diff.T.dot(inv_cov_matrix).dot(diff))

        return dis

    def kmeans_pp(self):

        distances = np.full(self.n, np.inf)
        centers = [np.random.randint(0, self.n)]

        for _ in range(1, self.k):
            for i in range(self.n):
                distances[i] = min(distances[i], self.mahalanobis_distance(self.data[i], self.data[centers[-1]]))
            centers.append(np.argmax(distances))
        print(centers)

        labels = np.zeros(self.n)
        distances_to_centroid = np.zeros(self.n)
        for i in range(self.n):
            min_distance = 1000
            for j in range(self.k):
                dist = self.mahalanobis_distance(self.data[i], self.data[centers[j]])
                if dist < min_distance:
                    min_distance = dist
                    labels[i] = j
            distances_to_centroid[i] = min_distance

        self.cluster_labels = labels
        self.distances_to_centroid = distances_to_centroid

    def run_clustering(self):

        self.kmeans_pp()
        results = np.vstack((np.arange(self.n), self.cluster_labels, self.distances_to_centroid)).T
        return results



def normal_cluster(sampling_strategy, X, y, k):
    # 返回索引
    ori_indices = []

    x_resampled, y_resampled = [], []
    indices = np.where(y == 1)[0]
    X1 = X[indices]
    fc = NomalClustering(X1, k)
    result = fc.run_clustering()

    indices = result[:, 2].argsort()
    sorted_data = X1[indices]
    result1 = result[indices]

    for i in range(sampling_strategy[1]):
        ori_indices.append(result1[i][0])
        x_resampled.append(sorted_data[i])
        y_resampled.append(1)

    indices = np.where(y != 1)[0]
    for i in indices:
        ori_indices.append(i)
        x_resampled.append(X[i])
        y_resampled.append(y[i])

    ori_indices = [int(x) for x in ori_indices]
    x_resampled = np.array(x_resampled, dtype=float)
    y_resampled = np.array(y_resampled, dtype=float)
    return x_resampled, y_resampled, ori_indices
