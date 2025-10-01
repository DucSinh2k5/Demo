import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=42)


plt.scatter(X[:,0], X[:,1], s=50)
plt.title("Data points")
plt.show()


kmeans = KMeans(n_clusters=4, random_state=42)
y_kmeans = kmeans.fit_predict(X)


plt.scatter(X[:,0], X[:,1], c=y_kmeans, cmap='viridis', s=50)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='red', marker='X', s=200, label='Centroids')
plt.title("KMeans Clustering")
plt.legend()
plt.show()
