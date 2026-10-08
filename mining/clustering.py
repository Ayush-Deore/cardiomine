"""
Mining — Clustering
K-Means patient segmentation.
Uses the Elbow Method to suggest optimal k, then trains with k=3 by default.
"""

import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CLUSTER_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "cluster_model.pkl")


def elbow_analysis(X_scaled: np.ndarray, k_range=range(2, 9)) -> dict:
    """
    Compute WCSS (inertia) and silhouette scores for a range of k values.
    Useful for the Elbow Method visualization.
    """
    results = {"k": [], "inertia": [], "silhouette": []}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        results["k"].append(k)
        results["inertia"].append(km.inertia_)
        sil = silhouette_score(X_scaled, labels) if k > 1 else 0
        results["silhouette"].append(round(sil, 4))
        logger.info(f"  k={k}: inertia={km.inertia_:.1f}, silhouette={sil:.4f}")
    return results


def train_kmeans(X_scaled: np.ndarray, n_clusters: int = 3, random_state: int = 42) -> KMeans:
    logger.info(f"Training K-Means with k={n_clusters}...")
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    km.fit(X_scaled)
    logger.info(f"K-Means complete. Inertia: {km.inertia_:.2f}")
    return km


def get_cluster_profiles(df: pd.DataFrame, labels: np.ndarray, feature_names: list) -> pd.DataFrame:
    """
    Build a cluster profile summary table.
    We DO NOT label clusters as high/medium/low risk; we report the data
    and let the analyst interpret.

    Returns
    -------
    pd.DataFrame
        One row per cluster with mean feature values + disease prevalence.
    """
    df_c = df.copy()
    df_c["cluster"] = labels

    profile_parts = []

    for cluster_id in sorted(df_c["cluster"].unique()):
        subset = df_c[df_c["cluster"] == cluster_id]
        row = {"cluster": int(cluster_id), "count": len(subset)}

        for feat in feature_names:
            if feat in subset.columns:
                row[f"avg_{feat}"] = round(subset[feat].mean(), 2)

        if "target" in df_c.columns:
            row["disease_prevalence_pct"] = round(subset["target"].mean() * 100, 1)

        profile_parts.append(row)

    profile_df = pd.DataFrame(profile_parts)
    logger.info(f"\nCluster profiles:\n{profile_df.to_string()}")
    return profile_df


def save_cluster_model(km: KMeans, scaler, feature_names: list, profiles: pd.DataFrame, elbow: dict):
    os.makedirs(os.path.dirname(CLUSTER_MODEL_PATH), exist_ok=True)
    package = {
        "kmeans": km,
        "scaler": scaler,
        "feature_names": feature_names,
        "profiles": profiles.to_dict("records"),
        "elbow": elbow,
    }
    joblib.dump(package, CLUSTER_MODEL_PATH)
    logger.info(f"Cluster model saved to: {CLUSTER_MODEL_PATH}")


def load_cluster_model(path: str = CLUSTER_MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cluster model not found at {path}. Run clustering.py first.")
    return joblib.load(path)


# ─────────────────────────────────────────────
# Run as standalone script
# ─────────────────────────────────────────────

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from mining.preprocess import load_clean_data, prepare_clustering_data

    df = load_clean_data()
    X_scaled, scaler, feature_names = prepare_clustering_data(df)

    # Elbow analysis
    logger.info("Running elbow analysis...")
    elbow = elbow_analysis(X_scaled)

    # Train with k=3
    km = train_kmeans(X_scaled, n_clusters=3)
    labels = km.labels_

    # Profile
    profiles = get_cluster_profiles(df, labels, feature_names)
    print(profiles)

    # Save
    save_cluster_model(km, scaler, feature_names, profiles, elbow)
