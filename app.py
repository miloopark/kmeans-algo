from flask import Flask, render_template, jsonify, request
from src.kmeans import KMeans
import numpy as np

app = Flask(__name__)

# Global variables to store dataset and kmeans instance
dataset = None
kmeans = None

# Generate random dataset
@app.route('/generate_dataset', methods=['POST'])
def generate_dataset():
    global dataset
    dataset = np.random.rand(100, 2)  # Generate 100 random 2D points
    return jsonify({'data': dataset.tolist()}), 200

# Initialize KMeans with chosen method
@app.route('/initialize', methods=['POST'])
def initialize():
    global kmeans
    method = request.json.get('method')
    if dataset is None:
        return jsonify({'error': 'No dataset generated'}), 400
    
    kmeans = KMeans(dataset, n_clusters=3, init_method=method)
    kmeans.initialize_centroids()
    return jsonify({'centroids': kmeans.centroids.tolist()}), 200

# Perform one step of KMeans algorithm
@app.route('/step', methods=['POST'])
def step():
    if kmeans is None:
        return jsonify({'error': 'KMeans not initialized'}), 400
    kmeans.step()
    return jsonify({
        'centroids': kmeans.centroids.tolist(),
        'assignments': kmeans.assignments.tolist()
    }), 200

# Run KMeans to convergence
@app.route('/converge', methods=['POST'])
def converge():
    if kmeans is None:
        return jsonify({'error': 'KMeans not initialized'}), 400
    kmeans.converge()
    return jsonify({
        'centroids': kmeans.centroids.tolist(),
        'assignments': kmeans.assignments.tolist()
    }), 200

# Reset KMeans algorithm
@app.route('/reset', methods=['POST'])
def reset():
    global kmeans
    kmeans = None
    return jsonify({'message': 'KMeans reset successfully'}), 200

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)