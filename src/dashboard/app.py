from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import joblib
import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"
MODELING_PATH = DATA_DIR / "idm_2024_modeling.csv"
CLUSTERED_PATH = DATA_DIR / "idm_2024_clustered.csv"
PROV_STATS_PATH = DATA_DIR / "provinsi_stats_summary.csv"
MODEL_PATH = BASE_DIR / "models" / "classification_model.pkl"
ENCODER_PATH = BASE_DIR / "models" / "label_encoder_classification.pkl"

STATUS_ORDER = ["SANGAT TERTINGGAL", "TERTINGGAL", "BERKEMBANG", "MAJU", "MANDIRI"]
STATUS_PALETTE = {
    "SANGAT TERTINGGAL": "#b11226",
    "TERTINGGAL": "#dc2f02",
    "BERKEMBANG": "#f4a261",
    "MAJU": "#2a9d8f",
    "MANDIRI": "#264653",
}
CLUSTER_PALETTE = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261", "#264653"]
DIMENSI_OPTIONS = ["Ekonomi", "Lingkungan", "Sosial"]


st.set_page_config(
    page_title="Dashboard IDM 2024",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    header = pd.read_csv(path, nrows=0).columns.tolist()
    dtype_map = {col: str for col in header if col.startswith("KODE_")}
    df = pd.read_csv(path, dtype=dtype_map)

    numeric_columns = [
        "IKS_2024",
        "IKE_2024",
        "IKL_2024",
        "NILAI_IDM_2024",
        "jumlah_rekomendasi",
        "total_nilai_rekomendasi",
        "gap_iks_ike",
        "gap_iks_ikl",
        "intensitas_rekomendasi",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


@st.cache_data(show_spinner=False)
def load_prov_stats() -> pd.DataFrame:
    if PROV_STATS_PATH.exists():
        df_stats = pd.read_csv(PROV_STATS_PATH)
        if "NAMA_PROVINSI" in df_stats.columns:
            return df_stats

    modeling = load_csv(MODELING_PATH)
    if modeling.empty:
        return pd.DataFrame()

    def get_mode_local(series: pd.Series) -> str:
        mode = series.mode(dropna=True)
        return mode.iloc[0] if not mode.empty else np.nan

    grouped = modeling.groupby("NAMA_PROVINSI").agg(
        IKS_median=("IKS_2024", "median"),
        IKE_median=("IKE_2024", "median"),
        IKL_median=("IKL_2024", "median"),
        NILAI_IDM_median=("NILAI_IDM_2024", "median"),
        STATUS_IDM_modus=("STATUS_IDM_2024", get_mode_local),
        jumlah_desa=("KODE_DESA", "count"),
    ).reset_index()
    grouped.to_csv(PROV_STATS_PATH, index=False)
    return grouped


@st.cache_resource(show_spinner=False)
def load_classification_artifacts() -> tuple[object, object]:
    if not MODEL_PATH.exists() or not ENCODER_PATH.exists():
        return None, None
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, encoder


def format_percent(value: float) -> str:
    return f"{value:.1f}%"


def mode_value(series: pd.Series) -> str:
    mode = series.mode(dropna=True)
    return mode.iloc[0] if not mode.empty else np.nan


def build_status_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "STATUS_IDM_2024" not in df.columns:
        return pd.DataFrame(columns=["STATUS_IDM_2024", "JUMLAH", "PERSENTASE"])

    summary = df["STATUS_IDM_2024"].value_counts().reindex(STATUS_ORDER, fill_value=0).reset_index()
    summary.columns = ["STATUS_IDM_2024", "JUMLAH"]
    summary["PERSENTASE"] = (summary["JUMLAH"] / len(df) * 100).round(2)
    return summary


def build_dimensi_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty or "dimensi_terendah" not in df.columns:
        empty = pd.DataFrame(columns=["DIMENSI", "JUMLAH", "PERSENTASE"])
        return empty, empty

    nasional = df["dimensi_terendah"].value_counts().reset_index()
    nasional.columns = ["DIMENSI", "JUMLAH"]
    nasional["PERSENTASE"] = (nasional["JUMLAH"] / len(df) * 100).round(2)

    tertinggal = df[df["STATUS_IDM_2024"].isin(["TERTINGGAL", "SANGAT TERTINGGAL"])]
    if tertinggal.empty:
        tertinggal_summary = pd.DataFrame(columns=["DIMENSI", "JUMLAH", "PERSENTASE"])
    else:
        tertinggal_summary = tertinggal["dimensi_terendah"].value_counts().reset_index()
        tertinggal_summary.columns = ["DIMENSI", "JUMLAH"]
        tertinggal_summary["PERSENTASE"] = (tertinggal_summary["JUMLAH"] / len(tertinggal) * 100).round(2)

    return nasional, tertinggal_summary


def build_prov_tertinggal(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "NAMA_PROVINSI" not in df.columns:
        return pd.DataFrame(columns=["NAMA_PROVINSI", "total_desa", "desa_tertinggal", "persentase_tertinggal"])

    working = df.copy()
    working["is_tertinggal"] = working["STATUS_IDM_2024"].isin(["TERTINGGAL", "SANGAT TERTINGGAL"])
    grouped = working.groupby("NAMA_PROVINSI").agg(
        total_desa=("KODE_DESA", "count"),
        desa_tertinggal=("is_tertinggal", "sum"),
    ).reset_index()
    grouped["persentase_tertinggal"] = (grouped["desa_tertinggal"] / grouped["total_desa"] * 100).round(2)
    return grouped.sort_values("persentase_tertinggal", ascending=False)


def build_correlation(df: pd.DataFrame) -> pd.DataFrame:
    features = [
        "IKS_2024",
        "IKE_2024",
        "IKL_2024",
        "NILAI_IDM_2024",
        "jumlah_rekomendasi",
        "gap_iks_ike",
        "gap_iks_ikl",
    ]
    available = [column for column in features if column in df.columns]
    if len(available) < 2:
        return pd.DataFrame()
    return df[available].corr()


def get_default_prediction_values(df: pd.DataFrame) -> dict[str, float]:
    defaults = {
        "jumlah_rekomendasi": 0.0,
        "total_nilai_rekomendasi": 0.0,
        "gap_iks_ike": 0.0,
        "gap_iks_ikl": 0.0,
        "intensitas_rekomendasi": 0.0,
    }
    for key in defaults:
        if key in df.columns:
            value = pd.to_numeric(df[key], errors="coerce").median()
            if pd.notna(value):
                defaults[key] = float(value)
    return defaults


def add_css() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(38, 70, 83, 0.14), transparent 28%),
                    radial-gradient(circle at top right, rgba(42, 157, 143, 0.12), transparent 26%),
                    linear-gradient(180deg, #f7f9fb 0%, #eef3f7 100%);
            }
            .block-container {
                padding-top: 1.4rem;
                padding-bottom: 2.5rem;
            }
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #17324d 0%, #0f2235 100%);
            }
            [data-testid="stSidebar"] * {
                color: #f7fbff;
            }
            .hero-card {
                background: linear-gradient(135deg, rgba(15, 34, 53, 0.97), rgba(42, 157, 143, 0.82));
                color: white;
                padding: 1.5rem 1.7rem;
                border-radius: 22px;
                box-shadow: 0 14px 40px rgba(14, 30, 48, 0.18);
                margin-bottom: 1rem;
            }
            .hero-card h1 {
                margin: 0;
                font-size: 2.1rem;
                line-height: 1.05;
            }
            .hero-card p {
                margin: 0.55rem 0 0 0;
                opacity: 0.9;
                font-size: 0.98rem;
                max-width: 58rem;
            }
            .metric-card {
                background: rgba(255, 255, 255, 0.82);
                border: 1px solid rgba(15, 34, 53, 0.06);
                border-radius: 18px;
                padding: 1rem 1.05rem;
                box-shadow: 0 10px 24px rgba(15, 34, 53, 0.06);
            }
            .metric-label {
                font-size: 0.8rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                color: #5b6875;
                margin-bottom: 0.35rem;
            }
            .metric-value {
                font-size: 1.9rem;
                font-weight: 800;
                color: #10263b;
                line-height: 1;
            }
            .metric-note {
                margin-top: 0.2rem;
                color: #607081;
                font-size: 0.88rem;
            }
            .section-title {
                font-size: 1.1rem;
                font-weight: 700;
                margin: 1rem 0 0.2rem 0;
                color: #10263b;
            }
            .panel-card {
                background: rgba(255, 255, 255, 0.9);
                border: 1px solid rgba(16, 38, 59, 0.08);
                border-radius: 20px;
                padding: 1rem 1.15rem;
                box-shadow: 0 10px 30px rgba(16, 38, 59, 0.08);
            }
            .panel-title {
                font-size: 1rem;
                font-weight: 700;
                color: #10263b;
                margin-bottom: 0.35rem;
            }
            .panel-subtitle {
                color: #607081;
                font-size: 0.9rem;
                margin-bottom: 0.85rem;
            }
            .insight-card {
                background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(234, 243, 247, 0.98));
                border: 1px solid rgba(16, 38, 59, 0.08);
                border-radius: 18px;
                padding: 0.95rem 1rem;
                margin-bottom: 0.65rem;
            }
            .insight-label {
                font-size: 0.78rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                color: #5b6875;
                margin-bottom: 0.25rem;
            }
            .insight-value {
                font-size: 1rem;
                font-weight: 700;
                color: #10263b;
                line-height: 1.35;
            }
            .recommendation-box {
                background: linear-gradient(135deg, rgba(42, 157, 143, 0.08), rgba(38, 70, 83, 0.06));
                border: 1px solid rgba(42, 157, 143, 0.18);
                border-radius: 18px;
                padding: 1rem 1.1rem;
                margin-top: 0.75rem;
            }
            .recommendation-box h4 {
                margin: 0 0 0.35rem 0;
                color: #10263b;
            }
            .prediction-banner {
                background: linear-gradient(135deg, rgba(255, 255, 255, 0.96), rgba(234, 243, 247, 0.96));
                color: #10263b;
                border: 1px solid rgba(16, 38, 59, 0.08);
                border-radius: 20px;
                padding: 1rem 1.15rem;
                margin: 0.25rem 0 1rem 0;
                box-shadow: 0 12px 30px rgba(16, 38, 59, 0.14);
            }
            .prediction-banner h3 {
                margin: 0;
                font-size: 1.05rem;
                font-weight: 800;
                color: #10263b;
            }
            .prediction-banner p {
                margin: 0.35rem 0 0 0;
                color: #5b6875;
                font-size: 0.92rem;
                line-height: 1.45;
            }
            .subsection-label {
                font-size: 0.82rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                font-weight: 700;
                color: #27455f;
                margin: 0.25rem 0 0.5rem 0;
            }
            .probability-card {
                background: rgba(255, 255, 255, 0.93);
                border: 1px solid rgba(16, 38, 59, 0.08);
                border-radius: 18px;
                padding: 1rem 1rem 0.85rem 1rem;
                box-shadow: 0 10px 24px rgba(16, 38, 59, 0.06);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_panel(title: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div class="panel-card">
            <div class="panel-title">{title}</div>
            <div class="panel-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_insight(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="insight-card">
            <div class="insight-label">{label}</div>
            <div class="insight-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plot_status_bar(status_summary: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    if status_summary.empty:
        ax.text(0.5, 0.5, "Data status tidak tersedia", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return fig

    palette = [STATUS_PALETTE.get(status, "#457B9D") for status in status_summary["STATUS_IDM_2024"]]
    sns.barplot(data=status_summary, x="STATUS_IDM_2024", y="JUMLAH", palette=palette, ax=ax)
    ax.set_title("Proporsi Status Indeks Desa Membangun (IDM) Nasional 2024")
    ax.set_xlabel("Status Desa")
    ax.set_ylabel("Jumlah Desa")
    ax.tick_params(axis="x", rotation=15)
    for idx, row in status_summary.iterrows():
        ax.text(idx, row["JUMLAH"] + max(status_summary["JUMLAH"]) * 0.015, f"{row['PERSENTASE']:.1f}%", ha="center", fontsize=9)
    fig.tight_layout()
    return fig


def plot_top_prov(prov_tertinggal: pd.DataFrame, top_n: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 6))
    top_frame = prov_tertinggal.head(top_n)
    if top_frame.empty:
        ax.text(0.5, 0.5, "Data provinsi tidak tersedia", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return fig

    sns.barplot(data=top_frame, x="persentase_tertinggal", y="NAMA_PROVINSI", palette="Reds_r", ax=ax)
    ax.set_title(f"Top {top_n} Provinsi dengan Persentase Desa Tertinggal/Sangat Tertinggal Tertinggi")
    ax.set_xlabel("Persentase Desa Tertinggal (%)")
    ax.set_ylabel("Provinsi")
    fig.tight_layout()
    return fig


def plot_dimensi_comparison(nasional: pd.DataFrame, tertinggal: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if nasional.empty:
        axes[0].text(0.5, 0.5, "Data dimensi nasional tidak tersedia", ha="center", va="center", transform=axes[0].transAxes)
        axes[0].axis("off")
    else:
        sns.barplot(data=nasional, x="DIMENSI", y="PERSENTASE", ax=axes[0], palette="Blues_r")
        axes[0].set_title("Dimensi Terendah Nasional")
        axes[0].set_xlabel("Dimensi")
        axes[0].set_ylabel("Persentase Desa (%)")

    if tertinggal.empty:
        axes[1].text(0.5, 0.5, "Data dimensi desa tertinggal tidak tersedia", ha="center", va="center", transform=axes[1].transAxes)
        axes[1].axis("off")
    else:
        sns.barplot(data=tertinggal, x="DIMENSI", y="PERSENTASE", ax=axes[1], palette="Oranges_r")
        axes[1].set_title("Dimensi Terendah pada Desa Tertinggal & Sangat Tertinggal")
        axes[1].set_xlabel("Dimensi")
        axes[1].set_ylabel("Persentase Desa (%)")

    fig.tight_layout()
    return fig


def plot_correlation_heatmap(corr_matrix: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    if corr_matrix.empty:
        ax.text(0.5, 0.5, "Korelasi tidak tersedia", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return fig

    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Heatmap Korelasi Indikator IDM")
    fig.tight_layout()
    return fig


def plot_scatter_iks_ike(df: pd.DataFrame, sample_size: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    required = {"IKE_2024", "IKS_2024", "STATUS_IDM_2024"}
    if df.empty or not required.issubset(df.columns):
        ax.text(0.5, 0.5, "Data scatter tidak tersedia", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return fig

    sample_size = min(sample_size, len(df))
    sample = df.sample(sample_size, random_state=42) if sample_size < len(df) else df
    sns.scatterplot(data=sample, x="IKE_2024", y="IKS_2024", hue="STATUS_IDM_2024", palette="Set1", alpha=0.45, ax=ax)
    ax.set_title(f"Hubungan IKE vs IKS berdasarkan Status IDM (Sampel {len(sample):,} Desa)")
    ax.set_xlabel("IKE 2024")
    ax.set_ylabel("IKS 2024")
    ax.legend(title="Status IDM", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig


def plot_cluster_2d(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    required = {"IKL_2024", "IKE_2024", "KLASTER"}
    if df.empty or not required.issubset(df.columns):
        ax.text(0.5, 0.5, "Data klaster tidak tersedia", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return fig

    clusters = sorted(df["KLASTER"].dropna().unique())
    for cluster in clusters:
        mask = df["KLASTER"] == cluster
        ax.scatter(
            df.loc[mask, "IKL_2024"],
            df.loc[mask, "IKE_2024"],
            s=12,
            alpha=0.25,
            color=CLUSTER_PALETTE[(int(cluster) - 1) % len(CLUSTER_PALETTE)],
            label=f"Klaster {int(cluster)}",
        )
    ax.set_title("Visualisasi Klaster Desa: IKL vs IKE")
    ax.set_xlabel("IKL 2024")
    ax.set_ylabel("IKE 2024")
    ax.legend(title="Klaster", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig


def plot_cluster_3d(df: pd.DataFrame) -> plt.Figure:
    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection="3d")
    required = {"IKS_2024", "IKE_2024", "IKL_2024", "KLASTER"}
    if df.empty or not required.issubset(df.columns):
        ax.text2D(0.35, 0.5, "Data klaster 3D tidak tersedia", transform=ax.transAxes)
        return fig

    clusters = sorted(df["KLASTER"].dropna().unique())
    for cluster in clusters:
        mask = df["KLASTER"] == cluster
        ax.scatter(
            df.loc[mask, "IKL_2024"],
            df.loc[mask, "IKE_2024"],
            df.loc[mask, "IKS_2024"],
            s=8,
            alpha=0.2,
            color=CLUSTER_PALETTE[(int(cluster) - 1) % len(CLUSTER_PALETTE)],
            label=f"Klaster {int(cluster)}",
        )

    ax.set_title("Visualisasi 3D Klaster Desa IDM 2024")
    ax.set_xlabel("IKL 2024")
    ax.set_ylabel("IKE 2024")
    ax.set_zlabel("IKS 2024")
    ax.legend(title="Klaster", bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    return fig


def build_cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "KLASTER" not in df.columns:
        return pd.DataFrame()

    columns = [column for column in ["IKS_2024", "IKE_2024", "IKL_2024", "NILAI_IDM_2024", "jumlah_rekomendasi"] if column in df.columns]
    summary = df.groupby(["KLASTER_LABEL" if "KLASTER_LABEL" in df.columns else "KLASTER"]).agg({
        **{column: "mean" for column in columns},
        "STATUS_IDM_2024": mode_value,
    }).reset_index()
    return summary

def detect_lowest_dimension(iks, ike, ikl):
    dimensions = {
        "Sosial": iks,
        "Ekonomi": ike,
        "Lingkungan": ikl,
    }
    return min(dimensions, key=dimensions.get)


def percentile_rank(df: pd.DataFrame, column: str, value: float) -> float:
    if column not in df.columns or df.empty:
        return 0.0
    return round((df[column] < value).mean() * 100, 1)


def recommendation_engine(lowest_dimension: str) -> list[str]:
    recommendations = {
        "Sosial": [
            "Perkuat akses pendidikan dasar dan pelatihan masyarakat",
            "Tingkatkan layanan kesehatan primer dan posyandu",
        ],
        "Ekonomi": [
            "Dorong pengembangan UMKM desa dan akses pembiayaan",
            "Perluas akses pasar dan digitalisasi ekonomi lokal",
        ],
        "Lingkungan": [
            "Perbaiki sanitasi, air bersih, dan pengelolaan limbah",
            "Perkuat mitigasi risiko lingkungan dan ketahanan bencana",
        ],
    }
    return recommendations.get(lowest_dimension, [])

def main() -> None:
    add_css()

    modeling_df = load_csv(MODELING_PATH)
    clustered_df = load_csv(CLUSTERED_PATH)
    prov_stats_df = load_prov_stats()

    st.markdown(
        """
        <div class="hero-card">
            <h1>Dashboard Streamlit IDM 2024</h1>
            <p>
                Ringkasan interaktif untuk eksplorasi status desa, provinsi dengan ketertinggalan tertinggi,
                dimensi penghambat utama, serta hasil clustering yang sudah diekspor dari notebook.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if modeling_df.empty:
        st.error(f"Dataset tidak ditemukan di {MODELING_PATH}")
        st.stop()

    st.sidebar.markdown("### Navigasi")
    page = st.sidebar.radio(
        "Pilih halaman",
        ["Ringkasan Nasional", "Analisis Indikator", "Clustering", "Predict Simulation", "Data Explorer"],
        index=0,
    )

    if page != "Predict Simulation":
        st.sidebar.markdown("### Filter Data")
        province_options = sorted(modeling_df["NAMA_PROVINSI"].dropna().unique().tolist()) if "NAMA_PROVINSI" in modeling_df.columns else []
        status_options = STATUS_ORDER if "STATUS_IDM_2024" in modeling_df.columns else []

        selected_provinces = st.sidebar.multiselect(
            "Provinsi",
            province_options,
            default=province_options[: min(5, len(province_options))],
        )
        selected_status = st.sidebar.multiselect(
            "Status IDM",
            status_options,
            default=status_options,
        )
        sample_size = st.sidebar.slider("Ukuran sampel scatter", min_value=500, max_value=10000, value=5000, step=500)
    else:
        selected_provinces = []
        selected_status = STATUS_ORDER
        sample_size = 5000

    pred_model, pred_encoder = load_classification_artifacts()

    filtered_df = modeling_df.copy()
    if selected_provinces and "NAMA_PROVINSI" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["NAMA_PROVINSI"].isin(selected_provinces)]
    if selected_status and "STATUS_IDM_2024" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["STATUS_IDM_2024"].isin(selected_status)]

    status_summary = build_status_summary(filtered_df)
    prov_tertinggal = build_prov_tertinggal(filtered_df)
    dimensi_nasional, dimensi_tertinggal = build_dimensi_summary(filtered_df)
    corr_matrix = build_correlation(filtered_df)

    total_desa = len(filtered_df)
    total_provinsi = filtered_df["NAMA_PROVINSI"].nunique() if "NAMA_PROVINSI" in filtered_df.columns else 0
    persen_tertinggal = 0.0
    persen_mandiri = 0.0
    rata_idm = 0.0
    if not filtered_df.empty:
        persen_tertinggal = filtered_df["STATUS_IDM_2024"].isin(["TERTINGGAL", "SANGAT TERTINGGAL"]).mean() * 100 if "STATUS_IDM_2024" in filtered_df.columns else 0.0
        persen_mandiri = (filtered_df["STATUS_IDM_2024"] == "MANDIRI").mean() * 100 if "STATUS_IDM_2024" in filtered_df.columns else 0.0
        rata_idm = filtered_df["NILAI_IDM_2024"].mean() if "NILAI_IDM_2024" in filtered_df.columns else 0.0

    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_metric("Total Desa", f"{total_desa:,}", "Setelah filter sidebar")
    with metric_cols[1]:
        render_metric("Provinsi", f"{total_provinsi:,}", "Provinsi dalam cakupan data")
    with metric_cols[2]:
        render_metric("Rata-rata IDM", f"{rata_idm:.3f}", "Nilai indeks rata-rata")
    with metric_cols[3]:
        render_metric("Rasio Tertinggal", format_percent(persen_tertinggal), f"Mandiri: {format_percent(persen_mandiri)}")

    if page == "Ringkasan Nasional":
        st.markdown('<div class="section-title">Gambaran Umum Nasional</div>', unsafe_allow_html=True)
        left, right = st.columns([1.15, 0.85])
        with left:
            st.pyplot(plot_status_bar(status_summary), clear_figure=True, use_container_width=True)
        with right:
            st.dataframe(
                status_summary,
                use_container_width=True,
                hide_index=True,
            )

        st.markdown('<div class="section-title">Provinsi dengan Ketertinggalan Tertinggi</div>', unsafe_allow_html=True)
        max_top = len(prov_tertinggal)
        if max_top == 0:
            top_n = 0
        elif max_top == 1:
            top_n = 1
            st.caption("Hanya ada 1 provinsi pada hasil filter saat ini.")
        elif max_top <= 5:
            top_n = max_top
            st.caption(f"Menampilkan seluruh {max_top} provinsi pada hasil filter saat ini.")
        else:
            max_top = min(20, max_top)
            default_top = min(10, max_top)
            top_n = st.slider("Jumlah provinsi teratas", min_value=5, max_value=max_top, value=default_top)
        left, right = st.columns([1.15, 0.85])
        with left:
            st.pyplot(plot_top_prov(prov_tertinggal, top_n), clear_figure=True, use_container_width=True)
        with right:
            st.dataframe(prov_tertinggal.head(top_n), use_container_width=True, hide_index=True)

        st.markdown('<div class="section-title">Ringkasan Provinsi</div>', unsafe_allow_html=True)
        if prov_stats_df.empty:
            st.info("File ringkasan provinsi belum tersedia.")
        else:
            st.dataframe(
                prov_stats_df.sort_values("NILAI_IDM_median", ascending=True),
                use_container_width=True,
                hide_index=True,
            )

    elif page == "Analisis Indikator":
        st.markdown('<div class="section-title">Dimensi Penghambat</div>', unsafe_allow_html=True)
        left, right = st.columns(2)
        with left:
            st.pyplot(plot_dimensi_comparison(dimensi_nasional, dimensi_tertinggal), clear_figure=True, use_container_width=True)
        with right:
            st.dataframe(
                dimensi_nasional,
                use_container_width=True,
                hide_index=True,
            )
            st.dataframe(
                dimensi_tertinggal,
                use_container_width=True,
                hide_index=True,
            )

        st.markdown('<div class="section-title">Relasi IKE dan IKS</div>', unsafe_allow_html=True)
        st.pyplot(plot_scatter_iks_ike(filtered_df, sample_size), clear_figure=True, use_container_width=True)

        st.markdown('<div class="section-title">Korelasi Variabel</div>', unsafe_allow_html=True)
        st.pyplot(plot_correlation_heatmap(corr_matrix), clear_figure=True, use_container_width=True)

    elif page == "Clustering":
        if clustered_df.empty:
            st.warning("File clustering tidak ditemukan. Jalankan notebook clustering terlebih dahulu untuk membuat model dan dataset terklaster.")
        else:
            cluster_view = clustered_df.copy()
            if selected_provinces and "NAMA_PROVINSI" in cluster_view.columns:
                cluster_view = cluster_view[cluster_view["NAMA_PROVINSI"].isin(selected_provinces)]
            if selected_status and "STATUS_IDM_2024" in cluster_view.columns:
                cluster_view = cluster_view[cluster_view["STATUS_IDM_2024"].isin(selected_status)]

            if "KLASTER" in cluster_view.columns:
                cluster_summary = build_cluster_summary(cluster_view)
                cluster_count = int(cluster_view["KLASTER"].nunique())
                cluster_size_text = f"{cluster_count} klaster"
            else:
                cluster_summary = pd.DataFrame()
                cluster_size_text = "Data klaster tidak lengkap"

            st.markdown('<div class="section-title">Ringkasan Klaster</div>', unsafe_allow_html=True)
            cluster_metric_cols = st.columns(3)
            with cluster_metric_cols[0]:
                render_metric("Baris Terklaster", f"{len(cluster_view):,}", "Setelah filter sidebar")
            with cluster_metric_cols[1]:
                render_metric("Jumlah Klaster", cluster_size_text, "Berdasarkan file model")
            with cluster_metric_cols[2]:
                render_metric("Label Klaster", "KLASTER / KLASTER_LABEL", "Kolom hasil K-Means")

            cluster_tabs = st.tabs(["2D Scatter", "3D Scatter", "Profil Klaster"])
            with cluster_tabs[0]:
                render_panel("Sebaran IKL vs IKE", "Melihat pemisahan klaster secara visual di dua dimensi")
                st.pyplot(plot_cluster_2d(cluster_view), clear_figure=True, use_container_width=True)
            with cluster_tabs[1]:
                render_panel("Sebaran 3D", "IKS, IKE, dan IKL ditampilkan bersamaan untuk melihat struktur klaster")
                st.pyplot(plot_cluster_3d(cluster_view), clear_figure=True, use_container_width=True)
            with cluster_tabs[2]:
                render_panel("Profil Rata-rata per Klaster", "Ringkasan statistik untuk tiap kelompok hasil K-Means")
                if cluster_summary.empty:
                    st.info("Ringkasan klaster belum tersedia.")
                else:
                    st.dataframe(cluster_summary, use_container_width=True, hide_index=True)

    elif page == "Predict Simulation":
        st.markdown(
            '<div class="section-title">Predict Simulation</div>',
            unsafe_allow_html=True
        )

        if pred_model is None or pred_encoder is None:
            st.error("Model klasifikasi belum tersedia.")
        else:
            st.markdown("### Simulasi Status IDM")
            st.caption(
                "Masukkan nilai IKS, IKE, dan IKL untuk melihat prediksi status desa berdasarkan model klasifikasi."
            )

            left_col, right_col = st.columns([0.95, 1.05], gap="large")

            # =========================
            # LEFT PANEL → INPUT
            # =========================
            with left_col:
                with st.container(border=True):
                    st.markdown("### Input Indikator")
                    st.caption("Atur nilai dari 0 sampai 1")

                    with st.form("prediction_form", border=False):
                        iks = st.slider(
                            "Indeks Ketahanan Sosial (IKS)",
                            0.0, 1.0, 0.50, 0.01
                        )

                        ike = st.slider(
                            "Indeks Ketahanan Ekonomi (IKE)",
                            0.0, 1.0, 0.50, 0.01
                        )

                        ikl = st.slider(
                            "Indeks Ketahanan Lingkungan (IKL)",
                            0.0, 1.0, 0.50, 0.01
                        )

                        gap_iks_ike = abs(iks - ike)
                        gap_iks_ikl = abs(iks - ikl)
                        lowest_dimension = detect_lowest_dimension(iks, ike, ikl)

                        st.divider()
                        st.markdown("#### Quick Summary")

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Gap IKS-IKE", f"{gap_iks_ike:.3f}")
                        c2.metric("Gap IKS-IKL", f"{gap_iks_ikl:.3f}")
                        c3.metric("Prioritas", lowest_dimension)

                        submitted = st.form_submit_button(
                            "Run Prediction",
                            use_container_width=True
                        )

            # =========================
            # RIGHT PANEL → OUTPUT
            # =========================
            with right_col:
                with st.container(border=True):
                    st.markdown("### Prediction Result")

                    if submitted:
                        defaults = get_default_prediction_values(modeling_df)

                        model_input = pd.DataFrame([{
                            "IKS_2024": iks,
                            "IKE_2024": ike,
                            "IKL_2024": ikl,
                            "jumlah_rekomendasi": defaults["jumlah_rekomendasi"],
                            "total_nilai_rekomendasi": defaults["total_nilai_rekomendasi"],
                            "gap_iks_ike": gap_iks_ike,
                            "gap_iks_ikl": gap_iks_ikl,
                            "intensitas_rekomendasi": defaults["intensitas_rekomendasi"],
                            "dimensi_terendah_Ekonomi": 1 if lowest_dimension == "Ekonomi" else 0,
                            "dimensi_terendah_Lingkungan": 1 if lowest_dimension == "Lingkungan" else 0,
                            "dimensi_terendah_Sosial": 1 if lowest_dimension == "Sosial" else 0,
                        }])

                        prediction = pred_model.predict(model_input)[0]
                        predicted_label = pred_encoder.inverse_transform([prediction])[0]

                        probabilities = pred_model.predict_proba(model_input)[0]

                        proba_df = pd.DataFrame({
                            "Status IDM": pred_encoder.classes_,
                            "Probabilitas": probabilities,
                        }).sort_values("Probabilitas", ascending=False)

                        top_conf = proba_df.iloc[0]["Probabilitas"]

                        # MAIN RESULT
                        st.success(f"Prediksi utama: **{predicted_label}**")
                        st.progress(float(top_conf))
                        st.caption(f"Confidence Score: {top_conf*100:.2f}%")

                        st.divider()

                        # PROBABILITY CHART
                        st.markdown("#### Probability Distribution")
                        st.bar_chart(
                            proba_df.set_index("Status IDM"),
                            use_container_width=True
                        )

                        st.divider()

                        # BENCHMARK
                        st.markdown("#### National Benchmark")

                        iks_pct = percentile_rank(modeling_df, "IKS_2024", iks)
                        ike_pct = percentile_rank(modeling_df, "IKE_2024", ike)
                        ikl_pct = percentile_rank(modeling_df, "IKL_2024", ikl)

                        b1, b2, b3 = st.columns(3)
                        b1.metric("IKS", f"{iks_pct:.1f}%")
                        b2.metric("IKE", f"{ike_pct:.1f}%")
                        b3.metric("IKL", f"{ikl_pct:.1f}%")

                        st.caption(
                            "Menunjukkan posisi indikator dibanding distribusi nasional."
                        )

                        st.divider()

                        # RECOMMENDATION
                        st.markdown("#### Intervention Recommendation")

                        st.warning(
                            f"Fokus utama saat ini ada pada dimensi **{lowest_dimension}**."
                        )

                        for rec in recommendation_engine(lowest_dimension):
                            st.markdown(f"• {rec}")

                        # BALANCED DETECTION
                        if max(iks, ike, ikl) - min(iks, ike, ikl) < 0.1:
                            st.success(
                                "Ketiga indikator cukup seimbang. Fokus pada maintenance dan scaling."
                            )

                    else:
                        st.info(
                            "Masukkan nilai indikator lalu klik **Run Prediction**."
                        )
    else:
        st.markdown('<div class="section-title">Data Explorer</div>', unsafe_allow_html=True)
        st.caption("Gunakan filter sidebar untuk memperkecil data, lalu unduh hasilnya bila perlu.")
        st.dataframe(filtered_df, use_container_width=True, height=520)

        csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Unduh data terfilter CSV",
            data=csv_bytes,
            file_name="idm_2024_filtered.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()