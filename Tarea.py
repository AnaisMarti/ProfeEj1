import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs

# Generar datos de ejemplo
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Método del Codo
def elbow_method(X):
    sse = []
    k_values = range(1, 11)
    for k in k_values:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, sse, marker='o')
    plt.title('Método del Codo')
    plt.xlabel('Número de Clústeres (k)')
    plt.ylabel('Suma de Errores Cuadráticos (SSE)')
    plt.show()

# Método de Silueta
def silhouette_method(X):
    silhouette_scores = []
    k_values = range(2, 11)
    for k in k_values:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        score = silhouette_score(X, kmeans.labels_)
        silhouette_scores.append(score)
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, silhouette_scores, marker='o')
    plt.title('Método de Silueta')
    plt.xlabel('Número de Clústeres (k)')
    plt.ylabel('Coeficiente de Silueta')
    plt.show()

# Ejecutar ambos métodos
elbow_method(X)
silhouette_method(X)
