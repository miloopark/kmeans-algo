from flask import Flask, render_template, jsonify, request
from src.kmeans import KMeans
import numpy as np

app = Flask(__name__)

# Global variables to store dataset and kmeans instance
dataset = None
kmeans = None
current_method = None
current_num_clusters = None

# Generate random dataset
@app.route('/generate_dataset', methods=['POST'])
def generate_dataset():
    global dataset, kmeans, current_method, current_num_clusters
    dataset = np.random.randn(300, 2) * 5  # Generate 300 random 2D points
    kmeans = None  # Reset KMeans instance when dataset changes
    current_method = None
    current_num_clusters = None
    return jsonify({'data': dataset.tolist()}), 200

# Perform one step of KMeans algorithm
@app.route('/step', methods=['POST'])
def step():
    global kmeans, current_method, current_num_clusters
    method = request.json.get('method')
    num_clusters = int(request.json.get('num_clusters', 3))
    centroids_input = request.json.get('centroids', None)

    if dataset is None:
        return jsonify({'error': 'No dataset generated'}), 400

    # Reinitialize kmeans if method or num_clusters have changed
    if kmeans is None or method != current_method or num_clusters != current_num_clusters:
        current_method = method
        current_num_clusters = num_clusters
        if method == 'manual':
            if centroids_input is None or len(centroids_input) != num_clusters:
                return jsonify({'error': 'Incorrect number of centroids provided.'}), 400
            kmeans = KMeans(dataset, n_clusters=num_clusters, init_method=method)
            kmeans.initialize_centroids(manual_centroids=centroids_input)
        else:
            kmeans = KMeans(dataset, n_clusters=num_clusters, init_method=method)
            kmeans.initialize_centroids()

    converged = kmeans.step()
    return jsonify({
        'centroids': kmeans.centroids.tolist(),
        'assignments': kmeans.assignments.tolist(),
        'converged': converged
    }), 200

# Run KMeans to convergence
@app.route('/converge', methods=['POST'])
def converge():
    global kmeans, current_method, current_num_clusters
    method = request.json.get('method')
    num_clusters = int(request.json.get('num_clusters', 3))
    centroids_input = request.json.get('centroids', None)

    if dataset is None:
        return jsonify({'error': 'No dataset generated'}), 400

    # Reinitialize kmeans if method or num_clusters have changed
    if kmeans is None or method != current_method or num_clusters != current_num_clusters:
        current_method = method
        current_num_clusters = num_clusters
        if method == 'manual':
            if centroids_input is None or len(centroids_input) != num_clusters:
                return jsonify({'error': 'Incorrect number of centroids provided.'}), 400
            kmeans = KMeans(dataset, n_clusters=num_clusters, init_method=method)
            kmeans.initialize_centroids(manual_centroids=centroids_input)
        else:
            kmeans = KMeans(dataset, n_clusters=num_clusters, init_method=method)
            kmeans.initialize_centroids()

    kmeans.converge()
    return jsonify({
        'centroids': kmeans.centroids.tolist(),
        'assignments': kmeans.assignments.tolist(),
        'converged': True
    }), 200

# Reset KMeans algorithm
@app.route('/reset', methods=['POST'])
def reset():
    global kmeans, current_method, current_num_clusters
    kmeans = None
    current_method = None
    current_num_clusters = None
    return jsonify({'message': 'KMeans reset successfully'}), 200

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)