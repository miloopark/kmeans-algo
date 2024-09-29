# KMeans Clustering Algorithm Visualization

## Overview

This project is an interactive web application that demonstrates the **KMeans Clustering Algorithm** using various initialization methods. The goal of this project is to visualize how different initialization methods impact the clustering outcome in real time, as well as allow users to step through the clustering process or converge the clusters directly.

### Features

- **Initialization Methods**:
  - `Random`: Centroids are randomly selected from the dataset.
  - `Farthest First`: Initial centroids are chosen to be as far apart as possible.
  - `KMeans++`: A smart initialization that accelerates convergence by spreading out centroids.
  - `Manual`: Allows users to manually select the initial centroids.

- **Visualization**:
  - 2D plot to display data points and centroids.
  - Step-by-step visualization of the clustering process.
  - Interactive manual centroid selection (planned).

- **User Interface**:
  - Drop-down menu to choose initialization methods.
  - Buttons to generate a new dataset, step through the algorithm, converge the clusters, and reset the algorithm.

## Running the Project

### Prerequisites

- Python 3.x
- Pip (Python package installer)
- Git

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/YOUR_GITHUB_USERNAME/kmeans-algo.git
   cd kmeans-algo```

2. **Install dependencies**:

   ```make install```

3. **Run application**:

   ```make run```
