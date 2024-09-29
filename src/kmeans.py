import numpy as np

class KMeans:
    def __init__(self, data, n_clusters=3, init_method='random'):
        self.data = data
        self.n_clusters = n_clusters
        self.init_method = init_method
        self.centroids = None
        self.assignments = None
    
    def initialize_centroids(self):
        if self.init_method == 'random':
            self.centroids = self.data[np.random.choice(self.data.shape[0], self.n_clusters, replace=False)]
        elif self.init_method == 'farthest':
            # Implement farthest-first initialization
            self.centroids = self.farthest_first_initialization()
        elif self.init_method == 'kmeans++':
            # Implement KMeans++ initialization
            self.centroids = self.kmeans_plus_plus_initialization()
        # Manual initialization will be handled by the front end

    def farthest_first_initialization(self):
        # Select one point randomly
        centroids = [self.data[np.random.choice(self.data.shape[0])]]
        for _ in range(1, self.n_clusters):
            distances = np.min([np.linalg.norm(self.data - c, axis=1) for c in centroids], axis=0)
            new_centroid = self.data[np.argmax(distances)]
            centroids.append(new_centroid)
        return np.array(centroids)

    def kmeans_plus_plus_initialization(self):
        centroids = [self.data[np.random.choice(self.data.shape[0])]]
        for _ in range(1, self.n_clusters):
            distances = np.min([np.linalg.norm(self.data - c, axis=1) for c in centroids], axis=0)
            probabilities = distances / np.sum(distances)
            new_centroid = self.data[np.random.choice(self.data.shape[0], p=probabilities)]
            centroids.append(new_centroid)
        return np.array(centroids)

    def step(self):
        # Assign points to the nearest centroid
        distances = np.array([np.linalg.norm(self.data - c, axis=1) for c in self.centroids])
        self.assignments = np.argmin(distances, axis=0)
        
        # Recompute centroids
        new_centroids = np.array([self.data[self.assignments == i].mean(axis=0) for i in range(self.n_clusters)])
        self.centroids = new_centroids

    def converge(self):
        while True:
            old_centroids = np.copy(self.centroids)
            self.step()
            if np.all(old_centroids == self.centroids):
                break