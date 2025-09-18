import os
import multiprocessing as mp
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

def env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default

if __name__ == "__main__":
    print("Hello from your sklearn container! 🐳")

    # Show thread settings
    print("Detected CPU cores:", mp.cpu_count())
    print("Thread caps:",
          {k: os.environ.get(k) for k in
           ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]})

    # Tiny demo
    X, y = make_blobs(n_samples=10_000, centers=5, n_features=8, random_state=42)
    df = pd.DataFrame(X)
    print("DataFrame shape:", df.shape)

    kmeans = KMeans(n_clusters=5, n_init="auto", random_state=42)
    kmeans.fit(df.values)
    print("KMeans inertia:", kmeans.inertia_)
    print("Cluster centers (first row):", kmeans.cluster_centers_[0])

    # Optional: probe for GPU presence (informational only; sklearn won’t use it)
    try:
        import subprocess
        smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if smi.returncode == 0:
            print("nvidia-smi found. GPU visible to container (not used by sklearn).")
        else:
            print("nvidia-smi not found or no GPU available (totally fine).")
    except Exception:
        print("nvidia-smi probe skipped.")

    print("Done ✅")

