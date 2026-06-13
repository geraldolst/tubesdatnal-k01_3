"""
Simplified notebook that uses backend modules
This can be run in Google Colab or locally
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add backend to path
BACKEND_PATH = Path('backend')
if BACKEND_PATH.exists():
    sys.path.insert(0, str(BACKEND_PATH))

# Imports from backend
try:
    from idm_backend import config, IDMCleaner, FeatureEngineer, ClusteringModel, ClassificationModel
    from idm_backend.api import PredictionAPI
    BACKEND_AVAILABLE = True
except ImportError:
    print("Warning: Backend modules not available. Install via: pip install -r backend/requirements.txt")
    BACKEND_AVAILABLE = False

# Visualization config
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*70)
print("IDM 2024 - ANALYSIS NOTEBOOK (Backend-Integrated)")
print("="*70)

# ============================================================================
# SECTION 1: LOAD & CLEAN DATA
# ============================================================================

print("\n[1] Loading data...")

# Load raw data (from local file or Google Drive)
DATA_PATH = config.RAW_DATA_PATH  # Change this to your data path

try:
    df_raw = pd.read_csv(DATA_PATH)
    print(f"✓ Loaded: {len(df_raw):,} rows × {df_raw.shape[1]} columns")
except FileNotFoundError:
    print(f"✗ Data file not found: {DATA_PATH}")
    print("  Please set DATA_PATH to your CSV file")
    df_raw = None

if df_raw is not None:
    print("\n[2] Cleaning data...")
    cleaner = IDMCleaner()

    df = cleaner.drop_unused_columns(df_raw)
    df = cleaner.remove_duplicates(df)
    df = cleaner.handle_missing_values(df, config.INDEX_COLS)
    df = cleaner.fix_data_types(df)
    df = cleaner.handle_status_encoding(df)

    print(f"✓ Data cleaned: {len(df):,} rows")
    print(f"  - Columns removed: {cleaner.cols_dropped}")
    print(f"  - Rows removed: {cleaner.rows_removed}")

    # ============================================================================
    # SECTION 2: FEATURE ENGINEERING
    # ============================================================================

    print("\n[3] Engineering features...")
    df_engineered = FeatureEngineer.apply_all_features(df)

    print("✓ Features created:")
    print("  - dimensi_terendah: Bottleneck dimension identification")
    print("  - gap_iks_ike, gap_iks_ikl: Dimension differences")
    print("  - intensitas_rekomendasi: Recommendation intensity")
    print("  - idm_category: IDM value categorization")

    # ============================================================================
    # SECTION 3: CLUSTERING ANALYSIS
    # ============================================================================

    if BACKEND_AVAILABLE:
        print("\n[4] Training clustering model...")

        X_cluster = df_engineered[config.FEATURE_COLS].values
        clustering_model = ClusteringModel(
            n_clusters=config.KMEANS_K,
            random_state=config.MODEL_RANDOM_STATE
        )

        metrics = clustering_model.train(X_cluster, feature_names=config.FEATURE_COLS)

        print(f"✓ K-Means trained (k={config.KMEANS_K})")
        print(f"  Silhouette Score: {metrics['silhouette_score']:.4f}")
        print(f"  Inertia: {metrics['inertia']:.2f}")

        # Get predictions
        df_engineered['cluster'] = clustering_model.predict(X_cluster)

        # Visualize clusters
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 2D scatter
        for cluster in sorted(df_engineered['cluster'].unique()):
            mask = df_engineered['cluster'] == cluster
            axes[0].scatter(
                df_engineered.loc[mask, 'IKL_2024'],
                df_engineered.loc[mask, 'IKE_2024'],
                label=f'Cluster {cluster+1}',
                alpha=0.6,
                s=20
            )

        axes[0].set_xlabel('IKL 2024')
        axes[0].set_ylabel('IKE 2024')
        axes[0].set_title(f'K-Means Clusters (k={config.KMEANS_K})')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Cluster sizes
        cluster_counts = df_engineered['cluster'].value_counts().sort_index()
        axes[1].bar([f'C{i+1}' for i in cluster_counts.index], cluster_counts.values)
        axes[1].set_title('Cluster Sizes')
        axes[1].set_ylabel('Number of Villages')
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.show()

        print(f"\nCluster Distribution:")
        for i, count in cluster_counts.items():
            pct = count / len(df_engineered) * 100
            print(f"  Cluster {i+1}: {count:>7,} villages ({pct:>5.1f}%)")

    # ============================================================================
    # SECTION 4: CLASSIFICATION ANALYSIS
    # ============================================================================

    if BACKEND_AVAILABLE:
        print("\n[5] Training classification model...")

        X_classify = df_engineered[config.FEATURE_COLS]
        y_classify = df_engineered[config.TARGET_COL]

        classification_model = ClassificationModel(
            model_type='RandomForest',
            random_state=config.MODEL_RANDOM_STATE
        )

        class_metrics = classification_model.train(
            X_classify, y_classify,
            feature_names=config.FEATURE_COLS,
            test_size=config.TEST_SIZE,
            cv_folds=config.CV_FOLDS
        )

        print("✓ Classification model trained")
        print(classification_model.get_performance_summary())

        # Feature importance
        importance = classification_model.get_feature_importance()

        fig, ax = plt.subplots(figsize=(8, 4))
        importance.set_index('feature')['importance'].plot(kind='barh', ax=ax)
        ax.set_title('Feature Importance for STATUS Prediction')
        ax.set_xlabel('Importance')
        plt.tight_layout()
        plt.show()

    # ============================================================================
    # SECTION 5: SAVE RESULTS
    # ============================================================================

    print("\n[6] Saving results...")

    if BACKEND_AVAILABLE:
        # Save models
        clustering_model.save(config.CLUSTERING_MODEL_PATH, config.SCALER_PATH)
        classification_model.save(
            config.CLASSIFIER_PATH,
            config.SCALER_PATH,
            config.LABEL_ENCODER_PATH
        )
        print(f"✓ Models saved to {config.MODELS_DIR}/")

    # Save data
    df_engineered.to_csv(config.PROCESSED_DATA_PATH, index=False)
    print(f"✓ Processed data saved: {config.PROCESSED_DATA_PATH}")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

    if BACKEND_AVAILABLE:
        print("\nNext steps:")
        print("1. Use PredictionAPI to make predictions")
        print("2. Integrate with frontend/Flask API")
        print("3. Deploy models to production")

print("\nNote: Import models for predictions:")
print("  from idm_backend.api import PredictionAPI")
print("  api = PredictionAPI()")
print("  api.load_models()")
print("  result = api.predict_single_village(0.75, 0.65, 0.70)")
