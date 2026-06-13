"""
Feature engineering module
"""
import pandas as pd
import numpy as np
from typing import List

class FeatureEngineer:
    """Create additional features for clustering and classification"""

    @staticmethod
    def create_lowest_dimension(df: pd.DataFrame,
                                iks_col: str = 'IKS_2024',
                                ike_col: str = 'IKE_2024',
                                ikl_col: str = 'IKL_2024') -> pd.Series:
        """Identify the lowest dimension (bottleneck) for each village"""
        def get_lowest(row):
            if pd.isna(row[iks_col]) or pd.isna(row[ike_col]) or pd.isna(row[ikl_col]):
                return 'Tidak Diketahui'

            values = {'IKS': row[iks_col], 'IKE': row[ike_col], 'IKL': row[ikl_col]}
            min_key = min(values, key=values.get)

            return {'IKS': 'Sosial', 'IKE': 'Ekonomi', 'IKL': 'Lingkungan'}.get(min_key, 'Tidak Diketahui')

        return df.apply(get_lowest, axis=1)

    @staticmethod
    def create_dimension_gaps(df: pd.DataFrame,
                              iks_col: str = 'IKS_2024',
                              ike_col: str = 'IKE_2024',
                              ikl_col: str = 'IKL_2024') -> pd.DataFrame:
        """Calculate gaps between dimensions"""
        df_new = df.copy()
        df_new['gap_iks_ike'] = df[iks_col] - df[ike_col]
        df_new['gap_iks_ikl'] = df[iks_col] - df[ikl_col]
        df_new['gap_ike_ikl'] = df[ike_col] - df[ikl_col]
        return df_new

    @staticmethod
    def normalize_recommendation_intensity(df: pd.DataFrame,
                                          rekom_col: str = 'jumlah_rekomendasi') -> pd.Series:
        """Normalize recommendation count by mean"""
        if rekom_col not in df.columns:
            return pd.Series(0, index=df.index)

        mean_rekom = df[rekom_col].mean()
        if mean_rekom == 0:
            return pd.Series(0, index=df.index)

        return (df[rekom_col] / mean_rekom).fillna(0)

    @staticmethod
    def create_idm_category(df: pd.DataFrame,
                            idm_col: str = 'NILAI_IDM_2024') -> pd.Series:
        """Categorize IDM values into bins"""
        bins = [0, 0.4, 0.6, 0.8, 1.0]
        labels = ['Sangat_Rendah', 'Rendah', 'Tinggi', 'Sangat_Tinggi']
        return pd.cut(df[idm_col], bins=bins, labels=labels, include_lowest=True)

    @staticmethod
    def apply_all_features(df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps"""
        df_eng = df.copy()

        # Feature 1: Lowest dimension
        df_eng['dimensi_terendah'] = FeatureEngineer.create_lowest_dimension(df)

        # Feature 2-4: Dimension gaps
        df_eng = FeatureEngineer.create_dimension_gaps(df_eng)

        # Feature 5: Recommendation intensity
        if 'jumlah_rekomendasi' in df_eng.columns:
            df_eng['intensitas_rekomendasi'] = FeatureEngineer.normalize_recommendation_intensity(df_eng)

        # Feature 6: IDM category
        if 'NILAI_IDM_2024' in df_eng.columns:
            df_eng['idm_category'] = FeatureEngineer.create_idm_category(df)

        return df_eng
