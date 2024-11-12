# Imports
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# Generate some data with anomalies
n_samples = 1000
n_features = 2
n_clusters = 5
n_anomalies = 50
# Generate normal data points
X, _ = make_blobs(n_samples=n_samples - n_anomalies, n_features=n_features, centers=n_clusters,random_state=46)

# Generate anomaly data points
anomaly_center = np.array([[10, 10]]) # Anomaly center
anomalies = anomaly_center + np.random.randn(n_anomalies, n_features) * 2.0
X = np.vstack((X, anomalies))


n_clusters_to_detect = n_clusters # Number of clusters (including anomalies)
kmeans = KMeans(n_clusters=n_clusters_to_detect)
kmeans.fit(X)
# Predict cluster labels
cluster_labels = kmeans.predict(X)

# Find the cluster centers
cluster_centers = kmeans.cluster_centers_
# Calculate the distance from each point to its assigned cluster center
distances = [np.linalg.norm(x - cluster_centers[cluster]) for x, cluster in zip(X, cluster_labels)]

# Define a threshold for anomaly detection (e.g., based on the distance percentile)
percentile_threshold = 95
threshold_distance = np.percentile(distances, percentile_threshold)

# Identify anomalies
anomalies = [X[i] for i, distance in enumerate(distances) if distance > threshold_distance]
anomalies = np.asarray(anomalies, dtype=np.float32)

# Printing the clusters
colors = cm.nipy_spectral(cluster_labels.astype(float) / 3)
plt.scatter(X[:, 0], X[:, 1], marker='.', s=50, lw=0, alpha=0.7,c=colors, edgecolor='k')
plt.scatter(anomalies[:, 0], anomalies[:, 1], color='purple', marker='.', s=50, label='Anomalies')
plt.show()

