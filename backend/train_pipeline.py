"""
Training pipeline script - Trains and exports models to .pkl files
Run this from the notebook or standalone
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add backend to path
BACKEND_PATH = Path(__file__).parent
sys.path.insert(0, str(BACKEND_PATH))

from idm_backend import config, IDMCleaner, FeatureEngineer, ClusteringModel, ClassificationModel

def train_and_export_models(raw_data_path: str) -> dict:
    """
    Complete training pipeline

    Args:
        raw_data_path: Path to raw IDM data CSV

    Returns:
        Dictionary with training results and model paths
    """
    print("=" * 70)
    print("IDM MODEL TRAINING PIPELINE")
    print("=" * 70)

    # 1. LOAD DATA
    print("\n[1/5] Loading data...")
    try:
        df_raw = pd.read_csv(raw_data_path)
        print(f"✓ Loaded {len(df_raw):,} rows x {df_raw.shape[1]} columns")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return {}

    # 2. PREPROCESSING
    print("\n[2/5] Cleaning and preprocessing data...")
    cleaner = IDMCleaner()

    # Clean
    df = cleaner.drop_unused_columns(df_raw)
    df = cleaner.remove_duplicates(df)
    df = cleaner.handle_missing_values(df, config.INDEX_COLS)
    df = cleaner.fix_data_types(df)
    df = cleaner.handle_status_encoding(df, status_map={v.upper(): k for k, v in enumerate(config.STATUS_MAP.values(), 1)})

    print(f"✓ Data cleaned: {len(df):,} rows")
    print(f"  Cleaning report: {cleaner.get_cleaning_summary()}")

    # Save clean data
    df.to_csv(config.CLEAN_DATA_PATH, index=False)
    print(f"✓ Clean data saved to {config.CLEAN_DATA_PATH}")

    # 3. FEATURE ENGINEERING
    print("\n[3/5] Engineering features...")
    df_engineered = FeatureEngineer.apply_all_features(df)
    print("✓ Features created:")
    print(f"  - dimensi_terendah")
    print(f"  - gap_iks_ike, gap_iks_ikl, gap_ike_ikl")
    print(f"  - idm_category")

    # 4. TRAIN CLUSTERING MODEL
    print("\n[4/5] Training clustering model (K-Means)...")
    X_cluster = df_engineered[config.FEATURE_COLS].values

    clustering_model = ClusteringModel(n_clusters=config.KMEANS_K,
                                       random_state=config.MODEL_RANDOM_STATE)
    cluster_metrics = clustering_model.train(X_cluster, feature_names=config.FEATURE_COLS)

    print(f"✓ K-Means trained (k={config.KMEANS_K})")
    print(f"  Silhouette Score: {cluster_metrics['silhouette_score']:.4f}")
    print(f"  Inertia: {cluster_metrics['inertia']:.2f}")

    # Save clustering model
    clustering_model.save(config.CLUSTERING_MODEL_PATH, config.SCALER_PATH)
    print(f"✓ Clustering model saved to {config.CLUSTERING_MODEL_PATH}")
    print(f"✓ Scaler saved to {config.SCALER_PATH}")

    # 5. TRAIN CLASSIFICATION MODEL
    print("\n[5/5] Training classification model (STATUS prediction)...")
    X_classify = df_engineered[config.FEATURE_COLS].copy()
    y_classify = df_engineered[config.TARGET_COL]

    classification_model = ClassificationModel(model_type='RandomForest',
                                               random_state=config.MODEL_RANDOM_STATE)
    class_metrics = classification_model.train(X_classify, y_classify,
                                                feature_names=config.FEATURE_COLS,
                                                test_size=config.TEST_SIZE,
                                                cv_folds=config.CV_FOLDS)

    print("✓ Classification model trained")
    print(classification_model.get_performance_summary())

    # Get feature importance
    feature_importance = classification_model.get_feature_importance()
    print("\nTop features:")
    print(feature_importance.to_string())

    # Save classification model
    classification_model.save(config.CLASSIFIER_PATH, config.SCALER_PATH, config.LABEL_ENCODER_PATH)
    print(f"✓ Classification model saved to {config.CLASSIFIER_PATH}")
    print(f"✓ Label encoder saved to {config.LABEL_ENCODER_PATH}")

    # SUMMARY
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nModel files created:")
    print(f"  1. {config.CLUSTERING_MODEL_PATH}")
    print(f"  2. {config.SCALER_PATH}")
    print(f"  3. {config.CLASSIFIER_PATH}")
    print(f"  4. {config.LABEL_ENCODER_PATH}")
    print(f"\nClean data:")
    print(f"  {config.CLEAN_DATA_PATH}")

    return {
        'clustering_metrics': cluster_metrics,
        'classification_metrics': class_metrics,
        'feature_importance': feature_importance.to_dict('records'),
        'model_paths': {
            'clustering': str(config.CLUSTERING_MODEL_PATH),
            'classifier': str(config.CLASSIFIER_PATH),
            'scaler': str(config.SCALER_PATH),
            'label_encoder': str(config.LABEL_ENCODER_PATH),
            'clean_data': str(config.CLEAN_DATA_PATH)
        }
    }


if __name__ == '__main__':
    # Example usage
    raw_data_path = config.RAW_DATA_PATH

    if not Path(raw_data_path).exists():
        print(f"Error: Raw data file not found at {raw_data_path}")
        sys.exit(1)

    results = train_and_export_models(str(raw_data_path))
