"""
Data cleaner module for IDM preprocessing
"""
import pandas as pd
import numpy as np
from typing import Tuple, List

class IDMCleaner:
    """Handle data cleaning for IDM dataset"""

    def __init__(self):
        self.cols_dropped = []
        self.rows_removed = 0
        self.cleaning_report = {}

    def drop_unused_columns(self, df: pd.DataFrame,
                            cols_to_drop: List[str] = None) -> pd.DataFrame:
        """Drop columns not relevant for analysis"""
        if cols_to_drop is None:
            cols_to_drop = ['Keterangan', 'Unnamed: 14']

        cols_exist = [c for c in cols_to_drop if c in df.columns]
        df_clean = df.drop(columns=cols_exist, errors='ignore')
        self.cols_dropped = cols_exist
        return df_clean

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows"""
        n_before = len(df)
        df_clean = df.drop_duplicates().reset_index(drop=True)
        self.rows_removed = n_before - len(df_clean)
        return df_clean

    def handle_missing_values(self, df: pd.DataFrame,
                              index_cols: List[str] = None) -> pd.DataFrame:
        """Handle missing values in key columns"""
        if index_cols is None:
            index_cols = ['IKS_2024', 'IKE_2024', 'IKL_2024', 'NILAI_IDM_2024']

        df_clean = df.copy()

        # Remove rows where all index cols are NaN
        mask_all_nan = df_clean[index_cols].isnull().all(axis=1)
        df_clean = df_clean[~mask_all_nan].reset_index(drop=True)

        # Fill NAMA_DESA
        if 'NAMA_DESA' in df_clean.columns:
            n_desa_nan = df_clean['NAMA_DESA'].isnull().sum()
            df_clean['NAMA_DESA'] = df_clean['NAMA_DESA'].fillna('Tidak Diketahui')
            self.cleaning_report['NAMA_DESA'] = n_desa_nan

        # Fill numeric columns with median
        for col in index_cols:
            if col in df_clean.columns:
                n_nan = df_clean[col].isnull().sum()
                if n_nan > 0:
                    median_val = df_clean[col].median()
                    df_clean[col] = df_clean[col].fillna(median_val)
                    self.cleaning_report[col] = {'count': n_nan, 'filled_value': median_val}

        return df_clean

    def fix_data_types(self, df: pd.DataFrame,
                       numeric_cols: List[str] = None,
                       string_cols: List[str] = None) -> pd.DataFrame:
        """Convert columns to correct data types"""
        df_clean = df.copy()

        if numeric_cols is None:
            numeric_cols = ['IKS_2024', 'IKE_2024', 'IKL_2024', 'NILAI_IDM_2024']

        if string_cols is None:
            string_cols = ['KODE_PROV', 'KODE_KAB', 'KODE_KEC', 'KODE_DESA']

        # Convert to numeric
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

        # Convert to string and strip
        for col in string_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.strip()

        return df_clean

    def handle_status_encoding(self, df: pd.DataFrame,
                               status_col: str = 'STATUS_IDM_2024',
                               status_map: dict = None) -> pd.DataFrame:
        """Apply ordinal encoding to STATUS column"""
        df_clean = df.copy()

        if status_map is None:
            status_map = {
                'SANGAT TERTINGGAL': 1,
                'TERTINGGAL': 2,
                'BERKEMBANG': 3,
                'MAJU': 4,
                'MANDIRI': 5
            }

        if status_col in df_clean.columns:
            df_clean['STATUS_IDM_ORD'] = df_clean[status_col].str.upper().map(status_map)

        return df_clean

    def get_cleaning_summary(self) -> dict:
        """Return summary of cleaning operations"""
        return {
            'columns_dropped': self.cols_dropped,
            'rows_removed': self.rows_removed,
            'missing_values_handled': self.cleaning_report
        }
