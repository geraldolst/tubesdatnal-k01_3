# Tugas Besar II4013 Data Analytics | Analisis IDM 2024
> K-01 | Kelompok 3

Dashboard ini dikembangkan untuk menganalisis data Indeks Desa Membangun (IDM) 2024 melalui visualisasi interaktif, clustering desa, dan simulasi prediksi status IDM.

## Main Features
- Ringkasan Nasional: menampilkan distribusi status IDM, jumlah desa, rata-rata IDM, dan rasio desa tertinggal.
- Analisis Indikator: menampilkan dimensi penghambat utama, hubungan antar indikator, dan korelasi variabel.
- Clustering: menampilkan hasil segmentasi desa berdasarkan kemiripan nilai IKS, IKE, dan IKL.
- Predict Simulation: melakukan simulasi prediksi status IDM berdasarkan input nilai IKS, IKE, dan IKL menggunakan model klasifikasi.
- Data Explorer: menampilkan dataset terproses dan memungkinkan pengguna mengunduh data hasil filter.

## How to Run
#### 1. Akses Secara Online
Sistem dapat diakses langsung melalui browser tanpa memerlukan instalasi tambahan melalui tautan berikut: <br>
https://idm-analysis.streamlit.app/

#### 2. Akses Secara Lokal
- Clone repository
- Install dependensi
- Menjalankan command pada terminal `streamlit run src/dashboard/app.py`

## Struktur Folder
```text
TUBESDATNAL-K01_3/
│
├── data/
│   ├── raw/
│   │   ├── idm_2024_raw.csv
│   │   └── rekomendasi_provinsi/
│   │
│   └── processed/
│       ├── idm_2024_modeling.csv
│       ├── idm_2024_clustered.csv
│       └── provinsi_stats_summary.csv
│
├── models/
│   ├── classification_model.pkl
│   ├── kmeans_model.pkl
│   ├── label_encoder_classification.pkl
│   └── scaler_clustering.pkl
│
├── notebooks/
│   ├── 02_scrub.ipynb
│   ├── 03_EDA.ipynb
│   ├── 04_model_clustering.ipynb
│   └── 05_model_classification.ipynb
│
├── src/
│   └── dashboard/
│       └── app.py
│
├── requirements.txt
└── README.md
```

## Dependensi
```text
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
```

Untuk instalasi seluruh dependensi, cukup install requirements yang sudah tersedia:
`pip install -r requirements.txt`

## Contributors
|          **Nama**        |  **NIM** |   **Peran**   |
|--------------------------|----------|----------------|
| [Michael Jeremy Bungaran](https://github.com/chaeljer18) | 18221136 |Documentation and Insight Lead|
| [Muhammad Daffa Al Ghifari](http://github.com/Daghif)       | 18223016 |Data Preprocessing|
| [Valereo Jibril Al Buchori](https://github.com/valereoo)    | 18223030 |Data Engineer |
| [Anggita Najmi Layali](https://github.com/gitaa001)    | 18223122 |Visualization / Dashboard Developer|
| [Geraldo Linggom S. T.](https://github.com/geraldolst)   | 18223136 |Data Analyst / Modeler|
