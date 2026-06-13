"""
Prediction API - Used by frontend for predictions
"""
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from .models import ClusteringModel, ClassificationModel
from . import config


class PredictionAPI:
    """API for making predictions using trained models"""

    def __init__(self):
        self.clustering_model = None
        self.classification_model = None
        self.loaded = False

    def load_models(self) -> bool:
        """Load all trained models from disk"""
        try:
            # Load clustering
            self.clustering_model = ClusteringModel.load(
                config.CLUSTERING_MODEL_PATH,
                config.SCALER_PATH
            )
            if self.clustering_model is None:
                raise Exception("Failed to load clustering model")

            # Load classification
            self.classification_model = ClassificationModel.load(
                config.CLASSIFIER_PATH,
                config.SCALER_PATH,
                config.LABEL_ENCODER_PATH
            )
            if self.classification_model is None:
                raise Exception("Failed to load classification model")

            self.loaded = True
            print("✓ All models loaded successfully")
            return True

        except Exception as e:
            print(f"✗ Error loading models: {e}")
            self.loaded = False
            return False

    def predict_cluster(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict cluster assignment for given features

        Args:
            features: Input features [IKS_2024, IKE_2024, IKL_2024]

        Returns:
            Tuple of (cluster_labels, distances_to_centroids)
        """
        if not self.loaded or self.clustering_model is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        return self.clustering_model.predict_with_distance(features)

    def predict_status(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict STATUS_IDM for given features

        Args:
            features: Input features [IKS_2024, IKE_2024, IKL_2024]

        Returns:
            Tuple of (predicted_labels, prediction_probabilities)
        """
        if not self.loaded or self.classification_model is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        predictions = self.classification_model.predict(features)
        probabilities = self.classification_model.predict_proba(features)

        return predictions, probabilities

    def predict_single_village(self, iks: float, ike: float, ikl: float) -> Dict:
        """
        Make predictions for a single village

        Args:
            iks: IKS value (0-1)
            ike: IKE value (0-1)
            ikl: IKL value (0-1)

        Returns:
            Dictionary with predictions and confidence scores
        """
        if not self.loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        # Prepare input
        features = np.array([[iks, ike, ikl]])

        # Get predictions
        cluster_pred, cluster_dist = self.clustering_model.predict_with_distance(features)
        status_pred, status_prob = self.classification_model.predict_status(features)

        # Get label classes
        status_classes = self.classification_model.label_encoder.classes_

        # Create result
        result = {
            'iks': float(iks),
            'ike': float(ike),
            'ikl': float(ikl),
            'idm': float((iks + ike + ikl) / 3),
            'clustering': {
                'cluster': int(cluster_pred[0]) + 1,  # 1-indexed
                'confidence_scores': cluster_dist[0].tolist(),
                'closest_centroid': float(np.min(cluster_dist[0]))
            },
            'classification': {
                'predicted_status': status_pred[0],
                'confidence': float(np.max(status_prob[0])),
                'probabilities': {
                    status_classes[i]: float(status_prob[0][i])
                    for i in range(len(status_classes))
                }
            }
        }

        return result

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions for multiple villages

        Args:
            df: DataFrame with columns [IKS_2024, IKE_2024, IKL_2024]

        Returns:
            DataFrame with original data + predictions
        """
        if not self.loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        result_df = df.copy()

        # Prepare features
        features = df[config.FEATURE_COLS].values

        # Get predictions
        cluster_pred, _ = self.clustering_model.predict_with_distance(features)
        status_pred, status_prob = self.classification_model.predict_status(features)

        # Add to dataframe
        result_df['predicted_cluster'] = cluster_pred + 1  # 1-indexed
        result_df['predicted_status'] = status_pred
        result_df['status_confidence'] = np.max(status_prob, axis=1)

        return result_df

    def get_cluster_info(self) -> Dict:
        """Get information about clusters"""
        if not self.loaded or self.clustering_model.model is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        n_clusters = self.clustering_model.model.n_clusters
        centroids = self.clustering_model.get_centroids_original_scale()

        info = {
            'n_clusters': n_clusters,
            'feature_names': config.FEATURE_COLS,
            'centroids': centroids.tolist() if centroids is not None else None,
            'silhouette_score': self.clustering_model.silhouette_score
        }

        return info

    def get_feature_importance(self) -> Dict:
        """Get feature importance from classification model"""
        if not self.loaded or self.classification_model is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        importance_df = self.classification_model.get_feature_importance()
        return importance_df.set_index('feature')['importance'].to_dict() if importance_df is not None else None
