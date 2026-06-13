"""
Prediction API - Load trained models and make predictions
Usage: Import this in Flask/FastAPI to serve predictions to frontend
"""

import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Tuple

from pipeline import Config


class PredictionAPI:
    """Unified API for making predictions using trained models"""

    def __init__(self):
        self.clustering_model = None
        self.clustering_scaler = None
        self.classification_model = None
        self.classification_scaler = None
        self.label_encoder = None
        self.loaded = False

    def load_models(self) -> bool:
        """Load all trained models from .pkl files"""
        print("\nLoading trained models...")

        try:
            # Clustering
            with open(Config.CLUSTERING_MODEL_PATH, 'rb') as f:
                self.clustering_model = pickle.load(f)
            print(f"✓ Loaded: {Config.CLUSTERING_MODEL_PATH}")

            with open(Config.SCALER_CLUSTERING_PATH, 'rb') as f:
                self.clustering_scaler = pickle.load(f)
            print(f"✓ Loaded: {Config.SCALER_CLUSTERING_PATH}")

            # Classification
            with open(Config.CLASSIFICATION_MODEL_PATH, 'rb') as f:
                self.classification_model = pickle.load(f)
            print(f"✓ Loaded: {Config.CLASSIFICATION_MODEL_PATH}")

            with open(Config.SCALER_CLASSIFICATION_PATH, 'rb') as f:
                self.classification_scaler = pickle.load(f)
            print(f"✓ Loaded: {Config.SCALER_CLASSIFICATION_PATH}")

            with open(Config.LABEL_ENCODER_PATH, 'rb') as f:
                self.label_encoder = pickle.load(f)
            print(f"✓ Loaded: {Config.LABEL_ENCODER_PATH}")

            self.loaded = True
            print("\n✅ All models loaded successfully!")
            return True

        except FileNotFoundError as e:
            print(f"\n❌ Error: Model file not found - {e}")
            print("   Run pipeline.py first to generate model files")
            return False

    def predict_single(self, iks: float, ike: float, ikl: float) -> Dict:
        """
        Make predictions for single village

        Args:
            iks: IKS value (0-1)
            ike: IKE value (0-1)
            ikl: IKL value (0-1)

        Returns:
            Dict with clustering and classification predictions
        """
        if not self.loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        features = np.array([[iks, ike, ikl]])

        # Clustering prediction
        X_scaled_cluster = self.clustering_scaler.transform(features)
        cluster_pred = self.clustering_model.predict(X_scaled_cluster)[0]
        cluster_distances = self.clustering_model.transform(X_scaled_cluster)[0]

        # Classification prediction
        X_scaled_class = self.classification_scaler.transform(features)
        status_pred_encoded = self.classification_model.predict(X_scaled_class)[0]
        status_pred = self.label_encoder.inverse_transform([status_pred_encoded])[0]
        status_proba = self.classification_model.predict_proba(X_scaled_class)[0]

        # Construct result
        result = {
            'input': {
                'IKS_2024': float(iks),
                'IKE_2024': float(ike),
                'IKL_2024': float(ikl),
                'IDM_average': float((iks + ike + ikl) / 3)
            },
            'clustering': {
                'cluster': int(cluster_pred + 1),  # 1-indexed
                'confidence_scores': cluster_distances.tolist(),
                'closest_distance': float(np.min(cluster_distances))
            },
            'classification': {
                'predicted_status': str(status_pred),
                'confidence': float(np.max(status_proba)),
                'probabilities': {
                    label: float(prob)
                    for label, prob in zip(self.label_encoder.classes_, status_proba)
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
        features = df[Config.FEATURE_COLS].values

        # Clustering
        X_scaled_cluster = self.clustering_scaler.transform(features)
        cluster_pred = self.clustering_model.predict(X_scaled_cluster)
        result_df['predicted_cluster'] = cluster_pred + 1  # 1-indexed

        # Classification
        X_scaled_class = self.classification_scaler.transform(features)
        status_pred_encoded = self.classification_model.predict(X_scaled_class)
        status_proba = self.classification_model.predict_proba(X_scaled_class)
        status_pred = self.label_encoder.inverse_transform(status_pred_encoded)

        result_df['predicted_status'] = status_pred
        result_df['status_confidence'] = np.max(status_proba, axis=1)

        return result_df


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    # Initialize API
    api = PredictionAPI()
    api.load_models()

    # Single prediction
    print("\n" + "="*70)
    print("EXAMPLE: Single Village Prediction")
    print("="*70)
    result = api.predict_single(iks=0.75, ike=0.65, ikl=0.70)
    print(f"\nInput: IKS=0.75, IKE=0.65, IKL=0.70")
    print(f"Result:\n{result}")

    # Batch prediction
    print("\n" + "="*70)
    print("EXAMPLE: Batch Prediction")
    print("="*70)
    df_sample = pd.DataFrame({
        'IKS_2024': [0.75, 0.85, 0.65],
        'IKE_2024': [0.65, 0.75, 0.55],
        'IKL_2024': [0.70, 0.80, 0.60]
    })
    results = api.predict_batch(df_sample)
    print(f"\nBatch Results:\n{results}")
