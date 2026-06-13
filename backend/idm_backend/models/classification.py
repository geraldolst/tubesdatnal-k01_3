"""
Classification models module
"""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix)

class ClassificationModel:
    """Wrapper for classification models (STATUS_IDM prediction)"""

    def __init__(self, model_type: str = 'RandomForest', random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        self.performance_metrics = {}

    def _initialize_model(self):
        """Initialize the base model"""
        if self.model_type == 'RandomForest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            )
        else:
            raise ValueError(f"Model type {self.model_type} not supported")

    def train(self, X: pd.DataFrame, y: pd.Series, feature_names: list = None,
              test_size: float = 0.2, cv_folds: int = 5) -> dict:
        """Train classification model"""
        self._initialize_model()
        self.feature_names = feature_names or X.columns.tolist()

        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=test_size, random_state=self.random_state, stratify=y_encoded
        )

        # Train model
        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)

        self.performance_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'n_test_samples': len(y_test)
        }

        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv_folds,
                                    scoring='f1_weighted', n_jobs=-1)
        self.performance_metrics['cv_f1_mean'] = cv_scores.mean()
        self.performance_metrics['cv_f1_std'] = cv_scores.std()

        return self.performance_metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for new data"""
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained yet. Call train() first.")

        X_scaled = self.scaler.transform(X)
        y_pred_encoded = self.model.predict(X_scaled)
        return self.label_encoder.inverse_transform(y_pred_encoded)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities for new data"""
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained yet. Call train() first.")

        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """Get feature importance"""
        if self.model is None:
            return None

        importances = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        return feature_importance.head(top_n)

    def save(self, model_path: Path, scaler_path: Path, encoder_path: Path) -> bool:
        """Save model, scaler, and label encoder"""
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            with open(encoder_path, 'wb') as f:
                pickle.dump(self.label_encoder, f)
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False

    @classmethod
    def load(cls, model_path: Path, scaler_path: Path, encoder_path: Path):
        """Load model, scaler, and label encoder"""
        instance = cls()
        try:
            with open(model_path, 'rb') as f:
                instance.model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                instance.scaler = pickle.load(f)
            with open(encoder_path, 'rb') as f:
                instance.label_encoder = pickle.load(f)
            return instance
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def get_performance_summary(self) -> str:
        """Get human-readable performance summary"""
        if not self.performance_metrics:
            return "No performance metrics available"

        summary = f"""
Classification Model Performance Summary:
- Accuracy:        {self.performance_metrics.get('accuracy', 0):.4f}
- Precision:       {self.performance_metrics.get('precision', 0):.4f}
- Recall:          {self.performance_metrics.get('recall', 0):.4f}
- F1 Score:        {self.performance_metrics.get('f1_score', 0):.4f}
- CV F1 Mean ± Std: {self.performance_metrics.get('cv_f1_mean', 0):.4f} ± {self.performance_metrics.get('cv_f1_std', 0):.4f}
- Test Samples:    {self.performance_metrics.get('n_test_samples', 0)}
        """
        return summary
