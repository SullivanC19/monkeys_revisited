import os
import multiprocessing as mp
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import monkeys

def env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default

if __name__ == "__main__":
    print("Hello from your sklearn container! 🐳")

    # Show thread settings
    print("Detected CPU cores:", mp.cpu_count())

    # Tiny demo
    X, y = make_blobs(n_samples=10_000, centers=5, n_features=8, random_state=42)
    df = pd.DataFrame(X)
    print("DataFrame shape:", df.shape)

    kmeans = KMeans(n_clusters=5, n_init="auto", random_state=42)
    kmeans.fit(df.values)
    print("KMeans inertia:", kmeans.inertia_)
    print("Cluster centers (first row):", kmeans.cluster_centers_[0])

    print("Done ✅")

