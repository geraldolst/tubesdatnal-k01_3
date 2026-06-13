"""
IDM 2024 Backend - Complete ML Pipeline
Comprehensive, documented, and production-ready

Structure:
- Clustering Pipeline: IKS, IKE, IKL → K-Means model
- Classification Pipeline: IKS, IKE, IKL → STATUS prediction model
"""

import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List

# ML & preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (silhouette_score, accuracy_score, precision_score,
                             recall_score, f1_score, classification_report, confusion_matrix)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Centralized configuration for the entire pipeline"""

    # Directory structure
    BASE_DIR = Path(__file__).parent
    MODELS_DIR = BASE_DIR / 'models'
    DATA_DIR = BASE_DIR / 'data'
    REPORTS_DIR = BASE_DIR / 'reports'

    # Create directories if not exist
    for dir_path in [MODELS_DIR, DATA_DIR, REPORTS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Data paths
    RAW_DATA_PATH = DATA_DIR / 'idm_2024_raw.csv'
    CLEAN_DATA_PATH = DATA_DIR / 'idm_2024_clean.csv'

    # Feature columns
    FEATURE_COLS = ['IKS_2024', 'IKE_2024', 'IKL_2024']
    TARGET_COL = 'STATUS_IDM_2024'
    INDEX_COLS = ['IKS_2024', 'IKE_2024', 'IKL_2024', 'NILAI_IDM_2024']

    # Clustering config
    KMEANS_K = 3
    KMEANS_RANDOM_STATE = 42
    CLUSTERING_MODEL_PATH = MODELS_DIR / 'kmeans_model.pkl'
    SCALER_CLUSTERING_PATH = MODELS_DIR / 'scaler_clustering.pkl'

    # Classification config
    RF_RANDOM_STATE = 42
    CLASSIFICATION_MODEL_PATH = MODELS_DIR / 'classifier_model.pkl'
    SCALER_CLASSIFICATION_PATH = MODELS_DIR / 'scaler_classification.pkl'
    LABEL_ENCODER_PATH = MODELS_DIR / 'label_encoder.pkl'

    # Training config
    TEST_SIZE = 0.2
    CV_FOLDS = 5

    # Status mapping
    STATUS_MAP_UPPER = {
        'SANGAT TERTINGGAL': 1,
        'TERTINGGAL': 2,
        'BERKEMBANG': 3,
        'MAJU': 4,
        'MANDIRI': 5
    }


# ============================================================================
# PHASE 1: DATA LOADING & INITIAL EXPLORATION
# ============================================================================

class DataLoader:
    """Load and explore the raw dataset"""

    @staticmethod
    def load_data(path: Path) -> pd.DataFrame:
        """Load raw IDM data from CSV"""
        print("\n" + "="*70)
        print("PHASE 1: DATA LOADING & EXPLORATION")
        print("="*70)
        print(f"\n[1.1] Loading data from {path}...")

        df = pd.read_csv(path)
        print(f"✓ Loaded: {len(df):,} rows × {df.shape[1]} columns")
        print(f"\nDataframe info:")
        print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"  Data types:\n{df.dtypes}")

        return df

    @staticmethod
    def explore_missing_values(df: pd.DataFrame) -> None:
        """Explore and report missing values"""
        print(f"\n[1.2] Missing Value Analysis:")
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)

        missing_data = pd.DataFrame({
            'Missing_Count': missing,
            'Percentage': missing_pct
        }).query('Missing_Count > 0').sort_values('Missing_Count', ascending=False)

        if missing_data.empty:
            print("✓ No missing values found")
        else:
            print(f"\nColumns with missing values:")
            print(missing_data)

        return missing_data


# ============================================================================
# PHASE 2: DATA CLEANING & PREPARATION
# ============================================================================

class DataCleaner:
    """
    Clean and prepare data for ML pipelines

    Justification for cleaning steps:
    1. Drop unused columns: Reduce noise and memory footprint
    2. Remove duplicates: Ensure data integrity
    3. Handle missing values: Required for ML models
    4. Fix data types: Ensure correct computation
    5. Encode status: Prepare for classification
    """

    def __init__(self):
        self.cleaning_report = {}

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute complete cleaning pipeline"""
        print("\n" + "="*70)
        print("PHASE 2: DATA CLEANING & PREPARATION")
        print("="*70)

        df_clean = df.copy()

        # Step 1: Drop unused columns
        print("\n[2.1] Dropping unused columns...")
        cols_to_drop = ['Keterangan', 'Unnamed: 14']
        cols_exist = [c for c in cols_to_drop if c in df_clean.columns]
        df_clean = df_clean.drop(columns=cols_exist, errors='ignore')
        print(f"✓ Dropped columns: {cols_exist}")
        self.cleaning_report['cols_dropped'] = cols_exist

        # Step 2: Remove duplicates
        print("\n[2.2] Removing duplicate rows...")
        n_before = len(df_clean)
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)
        n_removed = n_before - len(df_clean)
        print(f"✓ Removed {n_removed} duplicate rows")
        self.cleaning_report['duplicates_removed'] = n_removed

        # Step 3: Handle missing values
        print("\n[2.3] Handling missing values...")

        # JUSTIFICATION: Remove rows with all index values NaN
        # These are non-active villages (no population/governance) and can't be analyzed
        mask_all_nan = df_clean[Config.INDEX_COLS].isnull().all(axis=1)
        n_all_nan = mask_all_nan.sum()
        df_clean = df_clean[~mask_all_nan].reset_index(drop=True)
        print(f"  - Removed {n_all_nan} rows with all indices NaN (non-active villages)")

        # JUSTIFICATION: Fill NAMA_DESA with placeholder
        # Name doesn't affect numerical analysis; placeholder maintains referential integrity
        if 'NAMA_DESA' in df_clean.columns:
            n_desa_nan = df_clean['NAMA_DESA'].isnull().sum()
            df_clean['NAMA_DESA'] = df_clean['NAMA_DESA'].fillna('Tidak_Diketahui')
            print(f"  - Filled {n_desa_nan} missing NAMA_DESA")

        # JUSTIFICATION: Fill index columns with median
        # Median is robust to skewed distributions (IDM data often skewed)
        # Doesn't introduce artificial extremes like mean would
        print(f"\n  Filling numeric columns with median (robust to outliers):")
        for col in Config.INDEX_COLS:
            if col in df_clean.columns:
                n_nan = df_clean[col].isnull().sum()
                if n_nan > 0:
                    median_val = df_clean[col].median()
                    df_clean[col] = df_clean[col].fillna(median_val)
                    print(f"    {col}: {n_nan} → median={median_val:.4f}")

        # Step 4: Fix data types
        print("\n[2.4] Fixing data types...")
        # JUSTIFICATION: Convert indices to float for mathematical operations
        for col in Config.INDEX_COLS:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

        # JUSTIFICATION: Keep kode columns as string to preserve leading zeros
        # Example: KODE_DESA 1101012001 must not be treated as integer
        for col in ['KODE_PROV', 'KODE_KAB', 'KODE_KEC', 'KODE_DESA']:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.strip()

        print("✓ Data types corrected")

        # Step 5: Ordinal encode STATUS
        print("\n[2.5] Encoding STATUS column...")
        # JUSTIFICATION: Ordinal encoding (not one-hot) because status is hierarchical
        # Sangat Tertinggal (1) < Tertinggal (2) < ... < Mandiri (5)
        # This hierarchy is meaningful for ML models
        if Config.TARGET_COL in df_clean.columns:
            df_clean[f'{Config.TARGET_COL}_ORD'] = (
                df_clean[Config.TARGET_COL].str.upper().map(Config.STATUS_MAP_UPPER)
            )
            print(f"✓ STATUS encoded: Sangat Tertinggal=1 ... Mandiri=5")

        print(f"\n✓ Data cleaning complete: {len(df_clean):,} rows × {df_clean.shape[1]} columns")
        self.cleaning_report['final_shape'] = df_clean.shape

        return df_clean

    def get_report(self) -> Dict:
        """Return cleaning report"""
        return self.cleaning_report


# ============================================================================
# PHASE 2.5: RECOMMENDATION CONSOLIDATION (S2)
# ============================================================================

class RecommendationLoader:
    """
    Load and consolidate 32 recommendation Excel files from rekomendasi/ folder

    S2: Konsolidasi 32 File Rekomendasi
    """

    def __init__(self):
        self.rekomendasi_dir = Config.DATA_DIR / 'rekomendasi'
        self.consolidation_report = {}

    def load_all_files(self) -> pd.DataFrame:
        """Load all 32 Excel files from rekomendasi folder"""
        print("\n" + "="*70)
        print("PHASE 2.5: RECOMMENDATION CONSOLIDATION (S2)")
        print("="*70)

        if not self.rekomendasi_dir.exists():
            print(f"\n[!] Rekomendasi folder not found: {self.rekomendasi_dir}")
            print("    Skipping S2-S4 (recommendation integration)")
            self.consolidation_report['status'] = 'skipped'
            return pd.DataFrame()

        # Find all Excel files
        excel_files = sorted(list(self.rekomendasi_dir.glob('REKOMENDASI_IDM_PROVINSI_*.xlsx')))
        print(f"\n[2.5.1] Found {len(excel_files)} recommendation files")

        if len(excel_files) == 0:
            print("    No files found, skipping S2-S4")
            self.consolidation_report['status'] = 'no_files'
            return pd.DataFrame()

        dfs_list = []
        errors = []

        for fpath in excel_files:
            try:
                # Read Excel with header at row 3 (0-indexed: header=2)
                # Skip rows 0-1 (title + blank)
                df_temp = pd.read_excel(fpath, sheet_name=0, header=2)

                # Rename columns to remove spaces and standardize
                df_temp.columns = [c.strip().replace(' ', '_').upper() if isinstance(c, str) else c
                                  for c in df_temp.columns]

                dfs_list.append(df_temp)
                print(f"  ✓ {fpath.name}: {len(df_temp):,} rows")
            except Exception as e:
                errors.append((fpath.name, str(e)))
                print(f"  ✗ {fpath.name}: {str(e)[:50]}")

        if dfs_list:
            df_consolidated = pd.concat(dfs_list, ignore_index=True)
            print(f"\n[2.5.2] Consolidated {len(dfs_list)} files → {len(df_consolidated):,} rows")
            self.consolidation_report['status'] = 'success'
            self.consolidation_report['files_loaded'] = len(dfs_list)
            self.consolidation_report['total_rows'] = len(df_consolidated)
            self.consolidation_report['errors'] = errors
            return df_consolidated
        else:
            print("\n[!] No files successfully loaded")
            self.consolidation_report['status'] = 'failed'
            return pd.DataFrame()

    def get_report(self) -> Dict:
        """Return consolidation report"""
        return self.consolidation_report


# ============================================================================
# PHASE 2.6: RECOMMENDATION AGGREGATION (S3)
# ============================================================================

class RecommendationAggregator:
    """
    Aggregate recommendations per village (S3)
    Strategy: COUNT recommendations per KODE_DESA
    """

    def __init__(self):
        self.aggregation_report = {}

    def aggregate(self, df_rekom: pd.DataFrame) -> pd.DataFrame:
        """Aggregate recommendations by KODE_DESA"""
        print("\n" + "="*70)
        print("PHASE 2.6: RECOMMENDATION AGGREGATION (S3)")
        print("="*70)

        if df_rekom.empty:
            print("\n[!] No recommendation data to aggregate")
            return pd.DataFrame()

        print("\n[2.6.1] Identifying KODE_DESA column...")

        # Find KODE_DESA column (may have different naming)
        kode_desa_cols = [c for c in df_rekom.columns
                         if 'KODE' in c and 'DESA' in c]

        if not kode_desa_cols:
            print(f"    [!] KODE_DESA not found. Available columns:")
            print(f"    {list(df_rekom.columns[:15])}")
            return pd.DataFrame()

        kode_desa_col = kode_desa_cols[0]
        print(f"  ✓ Using column: {kode_desa_col}")

        # Convert KODE_DESA to string 10-digit format
        print("\n[2.6.2] Normalizing KODE_DESA to 10-digit string...")
        df_rekom[kode_desa_col] = (
            df_rekom[kode_desa_col]
            .astype(str)
            .str.strip()
            .str.zfill(10)
        )

        # Aggregate: count recommendations per village
        print("\n[2.6.3] Counting recommendations per village...")
        df_agg = df_rekom.groupby(kode_desa_col).size().reset_index(name='jumlah_rekomendasi')

        # Also aggregate KETERANGAN (concatenate with semicolon)
        if 'KETERANGAN' in df_rekom.columns:
            df_keterangan = df_rekom.groupby(kode_desa_col)['KETERANGAN'].apply(
                lambda x: ' | '.join(str(v) for v in x.dropna().unique() if str(v).strip() != 'nan')
            ).reset_index()
            df_keterangan.columns = [kode_desa_col, 'rekomendasi_keterangan']
            df_agg = df_agg.merge(df_keterangan, on=kode_desa_col, how='left')

        print(f"  ✓ Aggregated to {len(df_agg):,} villages with recommendations")

        # Rename KODE_DESA column to standard name for join
        df_agg.rename(columns={kode_desa_col: 'KODE_DESA_STR'}, inplace=True)

        self.aggregation_report['status'] = 'success'
        self.aggregation_report['unique_villages'] = len(df_agg)
        self.aggregation_report['total_recommendations'] = df_rekom.shape[0]

        return df_agg

    def get_report(self) -> Dict:
        """Return aggregation report"""
        return self.aggregation_report


# ============================================================================
# PHASE 2.7: DATASET INTEGRATION (S4)
# ============================================================================

class RecommendationIntegrator:
    """
    Integrate recommendation data with main dataset (S4)
    LEFT JOIN: main dataset (basis) ← recommendations
    """

    def __init__(self):
        self.integration_report = {}

    def integrate(self, df_clean: pd.DataFrame, df_agg_rekom: pd.DataFrame) -> pd.DataFrame:
        """LEFT JOIN recommendations with cleaned IDM data"""
        print("\n" + "="*70)
        print("PHASE 2.7: DATASET INTEGRATION (S4)")
        print("="*70)

        if df_agg_rekom.empty:
            print("\n[!] No aggregated recommendations. Using main dataset only.")
            df_clean['has_rekomendasi'] = False
            self.integration_report['status'] = 'no_rekomendasi'
            self.integration_report['join_type'] = 'none'
            return df_clean

        print("\n[2.7.1] Normalizing KODE_DESA for join...")
        df_result = df_clean.copy()
        df_result['KODE_DESA_STR'] = (
            df_result['KODE_DESA']
            .astype(str)
            .str.strip()
            .str.zfill(10)
        )

        print(f"  Main dataset: {len(df_result):,} villages")
        print(f"  Recommendations: {len(df_agg_rekom):,} villages")

        print("\n[2.7.2] Performing LEFT JOIN...")
        df_result = df_result.merge(
            df_agg_rekom,
            on='KODE_DESA_STR',
            how='left'
        )

        print(f"\n[2.7.3] Adding has_rekomendasi flag...")
        df_result['has_rekomendasi'] = df_result['jumlah_rekomendasi'].notna()

        n_dengan = df_result['has_rekomendasi'].sum()
        n_tanpa = len(df_result) - n_dengan

        print(f"  ✓ Villages WITH recommendations: {n_dengan:,} ({n_dengan/len(df_result)*100:.1f}%)")
        print(f"  ✓ Villages WITHOUT recommendations: {n_tanpa:,} ({n_tanpa/len(df_result)*100:.1f}%)")

        # Fill NaN in rekomendasi columns
        df_result['jumlah_rekomendasi'].fillna(0, inplace=True)
        df_result['rekomendasi_keterangan'].fillna('', inplace=True)

        self.integration_report['status'] = 'success'
        self.integration_report['total_villages'] = len(df_result)
        self.integration_report['villages_with_rekomendasi'] = n_dengan
        self.integration_report['villages_without_rekomendasi'] = n_tanpa

        return df_result

    def get_report(self) -> Dict:
        """Return integration report"""
        return self.integration_report


# ============================================================================
# PHASE 3: FEATURE ENGINEERING
# ============================================================================

class FeatureEngineer:
    """
    Create engineered features for ML models

    Features created:
    1. dimensi_terendah: Identifies bottleneck dimension (answers "which dimension is limiting?")
    2. gap_iks_ike: Difference between social & economic (shows inequality)
    3. gap_iks_ikl: Difference between social & environmental
    4. gap_ike_ikl: Difference between economic & environmental
    """

    @staticmethod
    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create all engineered features"""
        print("\n" + "="*70)
        print("PHASE 3: FEATURE ENGINEERING")
        print("="*70)

        df_eng = df.copy()

        # Feature 1: Lowest dimension (bottleneck)
        print("\n[3.1] Creating dimensi_terendah (bottleneck identification)...")
        def get_lowest_dimension(row):
            """
            JUSTIFICATION: Identifies which dimension limits overall development
            Used to answer: "What's the main constraint for this village?"
            """
            if pd.isna(row['IKS_2024']) or pd.isna(row['IKE_2024']) or pd.isna(row['IKL_2024']):
                return 'Tidak_Diketahui'

            values = {
                'IKS': row['IKS_2024'],
                'IKE': row['IKE_2024'],
                'IKL': row['IKL_2024']
            }
            min_key = min(values, key=values.get)
            return {
                'IKS': 'Sosial',
                'IKE': 'Ekonomi',
                'IKL': 'Lingkungan'
            }.get(min_key, 'Tidak_Diketahui')

        df_eng['dimensi_terendah'] = df_eng.apply(get_lowest_dimension, axis=1)
        print(f"✓ Created dimensi_terendah")
        print(f"  Distribution: {df_eng['dimensi_terendah'].value_counts().to_dict()}")

        # Features 2-4: Dimension gaps
        print("\n[3.2] Creating dimension gaps (inequality measures)...")
        # JUSTIFICATION: Large gaps indicate inequality between dimensions
        # Helps identify villages needing targeted interventions
        df_eng['gap_iks_ike'] = df_eng['IKS_2024'] - df_eng['IKE_2024']
        df_eng['gap_iks_ikl'] = df_eng['IKS_2024'] - df_eng['IKL_2024']
        df_eng['gap_ike_ikl'] = df_eng['IKE_2024'] - df_eng['IKL_2024']

        print(f"✓ Created gap_iks_ike: [{df_eng['gap_iks_ike'].min():.4f}, {df_eng['gap_iks_ike'].max():.4f}]")
        print(f"✓ Created gap_iks_ikl: [{df_eng['gap_iks_ikl'].min():.4f}, {df_eng['gap_iks_ikl'].max():.4f}]")
        print(f"✓ Created gap_ike_ikl: [{df_eng['gap_ike_ikl'].min():.4f}, {df_eng['gap_ike_ikl'].max():.4f}]")

        # Feature 5: Intensitas Rekomendasi (if available)
        if 'jumlah_rekomendasi' in df_eng.columns:
            mean_rekom = df_eng['jumlah_rekomendasi'].mean()
            df_eng['intensitas_rekomendasi'] = (
                df_eng['jumlah_rekomendasi'] / max(mean_rekom, 1)
            ).fillna(0)
            print(f"\n✓ Created intensitas_rekomendasi: [{df_eng['intensitas_rekomendasi'].min():.4f}, {df_eng['intensitas_rekomendasi'].max():.4f}]")

        print(f"\n✓ Feature engineering complete: +4-5 features created")

        return df_eng


# ============================================================================
# PHASE 4: CLUSTERING PIPELINE
# ============================================================================

class ClusteringPipeline:
    """
    Clustering pipeline: K-Means on [IKS, IKE, IKL]

    Input: Raw features [IKS_2024, IKE_2024, IKL_2024]
    Output: .pkl files (model, scaler)

    Why K-Means?
    - Unsupervised: No labels needed
    - Scalable: Handles 75k+ villages efficiently
    - Interpretable: Cluster centers have meaning
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self.silhouette_score = None

    def train(self, df: pd.DataFrame) -> Dict:
        """Train clustering model"""
        print("\n" + "="*70)
        print("PHASE 4: CLUSTERING PIPELINE")
        print("="*70)

        print("\n[4.1] Preparing data for clustering...")
        X = df[Config.FEATURE_COLS].values
        print(f"✓ Features shape: {X.shape}")
        print(f"  Feature ranges:")
        for i, col in enumerate(Config.FEATURE_COLS):
            print(f"    {col}: [{X[:, i].min():.4f}, {X[:, i].max():.4f}]")

        # JUSTIFICATION: StandardScaler normalizes features
        # Without scaling, IKE (typically lower range) would be overwhelmed by higher-variance features
        # Ensures each dimension contributes equally to distance calculations
        print("\n[4.2] Standardizing features...")
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        print(f"✓ Scaled features: mean≈0, std≈1")
        print(f"  After scaling - mean: {X_scaled.mean(axis=0)}")
        print(f"  After scaling - std: {X_scaled.std(axis=0)}")

        # Train K-Means
        print(f"\n[4.3] Training K-Means (k={Config.KMEANS_K})...")
        self.model = KMeans(
            n_clusters=Config.KMEANS_K,
            init='k-means++',  # Smart initialization for better convergence
            n_init=15,  # Run 15 times, keep best
            max_iter=500,
            random_state=Config.KMEANS_RANDOM_STATE,
            verbose=0
        )
        self.model.fit(X_scaled)
        print(f"✓ Model converged in {self.model.n_iter_} iterations")

        # Evaluate
        print(f"\n[4.4] Evaluating model...")
        self.silhouette_score = silhouette_score(X_scaled, self.model.labels_, sample_size=10000)
        print(f"✓ Silhouette Score: {self.silhouette_score:.4f}")
        print(f"  Inertia (WCSS): {self.model.inertia_:.2f}")

        # Cluster distribution
        print(f"\n[4.5] Cluster distribution:")
        unique, counts = np.unique(self.model.labels_, return_counts=True)
        for cluster, count in zip(unique, counts):
            pct = count / len(self.model.labels_) * 100
            print(f"  Cluster {cluster+1}: {count:>7,} villages ({pct:>5.1f}%)")

        return {
            'n_clusters': Config.KMEANS_K,
            'silhouette_score': self.silhouette_score,
            'inertia': self.model.inertia_,
            'n_iterations': self.model.n_iter_
        }

    def save_models(self) -> None:
        """Save trained models to pickle files"""
        print(f"\n[4.6] Saving models...")
        with open(Config.CLUSTERING_MODEL_PATH, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"✓ Saved: {Config.CLUSTERING_MODEL_PATH}")

        with open(Config.SCALER_CLUSTERING_PATH, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"✓ Saved: {Config.SCALER_CLUSTERING_PATH}")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict clusters for data"""
        X = df[Config.FEATURE_COLS].values
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


# ============================================================================
# PHASE 5: CLASSIFICATION PIPELINE
# ============================================================================

class ClassificationPipeline:
    """
    Classification pipeline: Random Forest for STATUS_IDM prediction

    Input: Raw features [IKS_2024, IKE_2024, IKL_2024]
    Target: STATUS_IDM_2024 (5 classes: Sangat Tertinggal → Mandiri)
    Output: .pkl files (model, scaler, label_encoder)

    Why Random Forest?
    - Handles non-linear relationships
    - Feature importance interpretation
    - Robust to outliers
    - Works well with mixed feature ranges
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.performance_metrics = {}

    def train(self, df: pd.DataFrame) -> Dict:
        """Train classification model"""
        print("\n" + "="*70)
        print("PHASE 5: CLASSIFICATION PIPELINE")
        print("="*70)

        print("\n[5.1] Preparing data for classification...")
        X = df[Config.FEATURE_COLS].copy()
        y_raw = df[Config.TARGET_COL].str.upper()

        # JUSTIFICATION: LabelEncoder converts categorical STATUS to numeric codes
        # Required input format for scikit-learn classifiers
        print(f"✓ Target classes: {sorted(y_raw.unique())}")
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y_raw)
        print(f"  Encoded as: {dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}")

        # Train/test split
        print(f"\n[5.2] Splitting data (test_size={Config.TEST_SIZE})...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded,
            test_size=Config.TEST_SIZE,
            random_state=Config.RF_RANDOM_STATE,
            stratify=y_encoded
        )
        print(f"✓ Train: {len(X_train):,} | Test: {len(X_test):,}")
        print(f"  Class distribution (train):")
        for label, count in zip(*np.unique(y_train, return_counts=True)):
            status = self.label_encoder.inverse_transform([label])[0]
            pct = count / len(y_train) * 100
            print(f"    {status}: {count:>6,} ({pct:>5.1f}%)")

        # JUSTIFICATION: StandardScaler for better Random Forest performance
        # Though RF is scale-invariant, scaled features can improve tree splits
        print(f"\n[5.3] Standardizing features...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        print(f"✓ Scaled training and test data")

        # Train Random Forest
        print(f"\n[5.4] Training Random Forest classifier...")
        self.model = RandomForestClassifier(
            n_estimators=200,  # 200 trees for stability
            max_depth=20,  # Limit depth to prevent overfitting
            min_samples_split=5,  # Require at least 5 samples to split
            min_samples_leaf=2,  # At least 2 samples in leaf
            class_weight='balanced',  # Handle class imbalance
            random_state=Config.RF_RANDOM_STATE,
            n_jobs=-1,
            verbose=0
        )
        self.model.fit(X_train_scaled, y_train)
        print(f"✓ Model trained with 200 trees")

        # Evaluate
        print(f"\n[5.5] Evaluating model...")
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        self.performance_metrics = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1
        }

        print(f"\nTest Set Performance:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1 Score:  {f1:.4f}")

        # Cross-validation
        print(f"\n[5.6] Cross-validation (5-fold)...")
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='f1_weighted',
            n_jobs=-1
        )
        print(f"✓ CV F1 Scores: {cv_scores}")
        print(f"  Mean: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

        # Feature importance
        print(f"\n[5.7] Feature importance:")
        feature_importance = sorted(
            zip(Config.FEATURE_COLS, self.model.feature_importances_),
            key=lambda x: x[1],
            reverse=True
        )
        for feat, imp in feature_importance:
            print(f"  {feat}: {imp:.4f}")

        # Classification report
        print(f"\n[5.8] Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))

        return self.performance_metrics

    def save_models(self) -> None:
        """Save trained models to pickle files"""
        print(f"\n[5.9] Saving models...")
        with open(Config.CLASSIFICATION_MODEL_PATH, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"✓ Saved: {Config.CLASSIFICATION_MODEL_PATH}")

        with open(Config.SCALER_CLASSIFICATION_PATH, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"✓ Saved: {Config.SCALER_CLASSIFICATION_PATH}")

        with open(Config.LABEL_ENCODER_PATH, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print(f"✓ Saved: {Config.LABEL_ENCODER_PATH}")

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict STATUS for data, return labels and probabilities"""
        X = df[Config.FEATURE_COLS].values
        X_scaled = self.scaler.transform(X)
        pred_encoded = self.model.predict(X_scaled)
        pred_labels = self.label_encoder.inverse_transform(pred_encoded)
        pred_proba = self.model.predict_proba(X_scaled)
        return pred_labels, pred_proba


# ============================================================================
# PHASE 6: EXECUTION & FINAL SUMMARY
# ============================================================================

class Pipeline:
    """Main execution pipeline"""

    @staticmethod
    def run_full_pipeline(raw_data_path: Path) -> Dict:
        """Execute complete ML pipeline"""

        # Load (Phase 1)
        loader = DataLoader()
        df_raw = loader.load_data(raw_data_path)
        loader.explore_missing_values(df_raw)

        # Clean (Phase 2)
        cleaner = DataCleaner()
        df_clean = cleaner.clean(df_raw)
        df_clean.to_csv(Config.CLEAN_DATA_PATH, index=False)
        print(f"✓ Clean data saved: {Config.CLEAN_DATA_PATH}")

        # S2: Load Recommendations
        rekom_loader = RecommendationLoader()
        df_rekom_raw = rekom_loader.load_all_files()

        # S3: Aggregate Recommendations
        rekom_agg = RecommendationAggregator()
        df_rekom_agg = rekom_agg.aggregate(df_rekom_raw)

        # S4: Integrate Datasets
        integrator = RecommendationIntegrator()
        df_integrated = integrator.integrate(df_clean, df_rekom_agg)

        # Engineer features (Phase 3)
        df_eng = FeatureEngineer.engineer_features(df_integrated)

        # Clustering
        clustering_pipe = ClusteringPipeline()
        clustering_metrics = clustering_pipe.train(df_eng)
        clustering_pipe.save_models()

        # Classification
        classification_pipe = ClassificationPipeline()
        classification_metrics = classification_pipe.train(df_eng)
        classification_pipe.save_models()

        # Summary
        print("\n" + "="*70)
        print("FINAL SUMMARY - COMPLETE PIPELINE")
        print("="*70)

        print(f"\n[S1] Data Cleaning:")
        print(f"  Status: Complete")
        print(f"  Rows: {len(df_clean):,}")

        print(f"\n[S2-S4] Recommendation Integration:")
        print(f"  S2 Status: {rekom_loader.get_report().get('status', 'N/A')}")
        if rekom_loader.get_report().get('status') == 'success':
            print(f"    Files loaded: {rekom_loader.get_report().get('files_loaded', 0)}")
            print(f"    Total rows: {rekom_loader.get_report().get('total_rows', 0):,}")
        print(f"  S3 Status: {rekom_agg.get_report().get('status', 'N/A')}")
        if rekom_agg.get_report().get('status') == 'success':
            print(f"    Villages with recommendations: {rekom_agg.get_report().get('unique_villages', 0):,}")
        print(f"  S4 Status: {integrator.get_report().get('status', 'N/A')}")
        if integrator.get_report().get('status') == 'success':
            print(f"    Villages with rekomendasi: {integrator.get_report().get('villages_with_rekomendasi', 0):,}")
            print(f"    Villages without rekomendasi: {integrator.get_report().get('villages_without_rekomendasi', 0):,}")

        print(f"\n[S5] Feature Engineering:")
        print(f"  Features created: 4-5 new features")

        print(f"\n✅ Pipeline execution complete!")
        print(f"\nModel files generated:")
        print(f"  1. {Config.CLUSTERING_MODEL_PATH}")
        print(f"  2. {Config.SCALER_CLUSTERING_PATH}")
        print(f"  3. {Config.CLASSIFICATION_MODEL_PATH}")
        print(f"  4. {Config.SCALER_CLASSIFICATION_PATH}")
        print(f"  5. {Config.LABEL_ENCODER_PATH}")
        print(f"\nClustering Performance:")
        print(f"  Silhouette Score: {clustering_metrics['silhouette_score']:.4f}")
        print(f"\nClassification Performance:")
        print(f"  Accuracy: {classification_metrics['accuracy']:.4f}")
        print(f"  F1-Score: {classification_metrics['f1_score']:.4f}")

        return {
            'clustering': clustering_metrics,
            'classification': classification_metrics
        }


# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == '__main__':
    # Run pipeline
    results = Pipeline.run_full_pipeline(Config.RAW_DATA_PATH)
    print("\n✨ All models ready for deployment!")
