"""
Clustering models module
"""
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class ClusteringModel:
    """Wrapper for K-Means clustering"""

    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.silhouette_score = None

    def train(self, X: np.ndarray, feature_names: list = None) -> dict:
        """Train K-Means model"""
        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]

        # Train K-Means
        self.model = KMeans(
            n_clusters=self.n_clusters,
            init='k-means++',
            n_init=15,
            max_iter=500,
            random_state=self.random_state,
            verbose=0
        )
        self.model.fit(X_scaled)

        # Calculate metrics
        self.silhouette_score = silhouette_score(X_scaled, self.model.labels_, sample_size=min(10000, len(X)))

        return {
            'n_clusters': self.n_clusters,
            'silhouette_score': self.silhouette_score,
            'inertia': self.model.inertia_,
            'n_samples': X.shape[0]
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster for new data"""
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained yet. Call train() first.")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_with_distance(self, X: np.ndarray) -> tuple:
        """Predict cluster and distance to centroid"""
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained yet. Call train() first.")

        X_scaled = self.scaler.transform(X)
        labels = self.model.predict(X_scaled)
        distances = self.model.transform(X_scaled)  # Distance to each centroid

        return labels, distances

    def save(self, model_path: Path, scaler_path: Path) -> bool:
        """Save model and scaler to pickle files"""
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False

    @classmethod
    def load(cls, model_path: Path, scaler_path: Path):
        """Load model and scaler from pickle files"""
        instance = cls()
        try:
            with open(model_path, 'rb') as f:
                instance.model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                instance.scaler = pickle.load(f)
            instance.n_clusters = instance.model.n_clusters
            return instance
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def get_centroids_original_scale(self) -> np.ndarray:
        """Get centroids in original scale (before StandardScaler)"""
        if self.model is None or self.scaler is None:
            return None
        return self.scaler.inverse_transform(self.model.cluster_centers_)

    def get_cluster_profile(self, df: pd.DataFrame, cluster_labels: np.ndarray) -> pd.DataFrame:
        """Get profile (mean values) for each cluster"""
        df_temp = df.copy()
        df_temp['CLUSTER'] = cluster_labels

        profile = df_temp.groupby('CLUSTER')[self.feature_names].agg(['mean', 'median', 'std']).round(4)
        cluster_counts = df_temp['CLUSTER'].value_counts().sort_index()

        profile['n_samples'] = cluster_counts.values
        return profile
