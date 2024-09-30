import numpy as np

class KMeans:
    def __init__(self, data, n_clusters=3, init_method='random'):
        self.data = np.array(data)
        self.n_clusters = n_clusters
        self.init_method = init_method
        self.centroids = None
        self.assignments = None

    def initialize_centroids(self, manual_centroids=None):
        if manual_centroids is not None:
            self.centroids = np.array(manual_centroids)
        elif self.init_method == 'random':
            indices = np.random.choice(self.data.shape[0], self.n_clusters, replace=False)
            self.centroids = self.data[indices]
        elif self.init_method == 'farthest':
            self.centroids = self.farthest_first_initialization()
        elif self.init_method == 'kmeans++':
            self.centroids = self.kmeans_plus_plus_initialization()

    def farthest_first_initialization(self):
        centroids = [self.data[np.random.choice(self.data.shape[0])]]
        for _ in range(1, self.n_clusters):
            distances = np.min([np.linalg.norm(self.data - c, axis=1) for c in centroids], axis=0)
            new_centroid = self.data[np.argmax(distances)]
            centroids.append(new_centroid)
        return np.array(centroids)

    def kmeans_plus_plus_initialization(self):
        centroids = [self.data[np.random.choice(self.data.shape[0])]]
        for _ in range(1, self.n_clusters):
            distances = np.min([np.linalg.norm(self.data - c, axis=1)**2 for c in centroids], axis=0)
            probabilities = distances / np.sum(distances)
            cumulative_probabilities = np.cumsum(probabilities)
            r = np.random.rand()
            for idx, prob in enumerate(cumulative_probabilities):
                if r < prob:
                    centroids.append(self.data[idx])
                    break
        return np.array(centroids)

    def step(self):
        # Assign points to the nearest centroid
        distances = np.array([np.linalg.norm(self.data - c, axis=1) for c in self.centroids])
        self.assignments = np.argmin(distances, axis=0)

        # Recompute centroids
        new_centroids = np.array([
            self.data[self.assignments == i].mean(axis=0) if np.any(self.assignments == i) else self.centroids[i]
            for i in range(self.n_clusters)
        ])

        converged = np.allclose(self.centroids, new_centroids)
        self.centroids = new_centroids

        return converged

    def converge(self):
        while True:
            converged = self.step()
            if converged:
                break