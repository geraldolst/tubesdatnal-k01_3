# IMPLEMENTASI BACKEND IDM - ARCHITECTURE & INTEGRATION GUIDE

## 📋 RINGKASAN

Struktur backend telah dibuat untuk production-ready ML pipeline dengan modular architecture. Semua model diserialisasi ke format `.pkl` (pickle) untuk deployment.

## 🏗️ STRUKTUR BACKEND

```
backend/
├── idm_backend/                          # Main Python package
│   ├── __init__.py
│   ├── config.py                         # Centralized configuration
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── cleaner.py                   # Class: IDMCleaner
│   │   └── feature_engineer.py          # Class: FeatureEngineer
│   ├── models/
│   │   ├── __init__.py
│   │   ├── clustering.py                # Class: ClusteringModel (K-Means)
│   │   └── classification.py            # Class: ClassificationModel (Random Forest)
│   ├── api/
│   │   ├── __init__.py
│   │   └── prediction.py                # Class: PredictionAPI (Frontend-facing)
│   └── utils/
│       └── __init__.py
├── models/                               # Output: Serialized models
│   ├── kmeans_model.pkl                 # K-Means trained model
│   ├── scaler_features.pkl              # StandardScaler for features
│   ├── classifier_model.pkl             # Random Forest classifier
│   └── label_encoder.pkl                # Status label encoder
├── data/                                 # Datasets
│   ├── idm_2024_raw.csv                 # Input raw data
│   └── idm_2024_clean.csv               # Cleaned data after preprocessing
├── notebooks/
│   └── notebook_with_backend.py         # Simplified notebook using backend modules
├── train_pipeline.py                    # Standalone training script
├── requirements.txt                     # Python dependencies
└── README.md                            # Full documentation
```

## 🔄 WORKFLOW

### Phase 1: Training (One-time or scheduled)

```
Raw Data (CSV)
     ↓
[train_pipeline.py]
     ├─→ IDMCleaner.drop_unused_columns()
     ├─→ IDMCleaner.remove_duplicates()
     ├─→ IDMCleaner.handle_missing_values()
     ├─→ IDMCleaner.fix_data_types()
     ├─→ IDMCleaner.handle_status_encoding()
     ├─→ FeatureEngineer.apply_all_features()
     ├─→ ClusteringModel.train() → kmeans_model.pkl
     ├─→ ClassificationModel.train() → classifier_model.pkl
     └─→ Saves: scaler_features.pkl, label_encoder.pkl
     ↓
Clean Data (CSV) + 4 .pkl Model Files
```

### Phase 2: Prediction (For Frontend)

```
Frontend Request (IKS, IKE, IKL values)
     ↓
[PredictionAPI.load_models()]
     ├─→ Load kmeans_model.pkl
     ├─→ Load classifier_model.pkl
     ├─→ Load scaler_features.pkl
     ├─→ Load label_encoder.pkl
     ↓
[PredictionAPI.predict_single_village(iks, ike, ikl)]
     ├─→ ClusteringModel.predict_with_distance()
     ├─→ ClassificationModel.predict_proba()
     ↓
Result JSON:
{
  'iks': 0.75,
  'ike': 0.65,
  'ikl': 0.70,
  'idm': 0.7,
  'clustering': {
    'cluster': 2,
    'confidence_scores': [1.2, 0.5, 2.1],
    'closest_centroid': 0.5
  },
  'classification': {
    'predicted_status': 'BERKEMBANG',
    'confidence': 0.92,
    'probabilities': {...}
  }
}
     ↓
Frontend Displays Results
```

## 📦 MODULES & CLASSES

### 1. **IDMCleaner** (`preprocessing/cleaner.py`)

Handles all data cleaning operations:

```python
cleaner = IDMCleaner()
df = cleaner.drop_unused_columns(df_raw)
df = cleaner.remove_duplicates(df)
df = cleaner.handle_missing_values(df, ['IKS_2024', 'IKE_2024', ...])
df = cleaner.fix_data_types(df)
df = cleaner.handle_status_encoding(df)
summary = cleaner.get_cleaning_summary()
```

### 2. **FeatureEngineer** (`preprocessing/feature_engineer.py`)

Creates additional features for modeling:

```python
df_eng = FeatureEngineer.apply_all_features(df)
# Creates:
# - dimensi_terendah: 'Sosial'|'Ekonomi'|'Lingkungan'
# - gap_iks_ike, gap_iks_ikl, gap_ike_ikl: float differences
# - intensitas_rekomendasi: normalized recommendation count
# - idm_category: 'Sangat_Rendah'|'Rendah'|'Tinggi'|'Sangat_Tinggi'
```

### 3. **ClusteringModel** (`models/clustering.py`)

K-Means clustering wrapper:

```python
model = ClusteringModel(n_clusters=3, random_state=42)
metrics = model.train(X_array, feature_names=['IKS_2024', 'IKE_2024', 'IKL_2024'])

# Predictions
labels = model.predict(X_new)
labels, distances = model.predict_with_distance(X_new)

# Serialization
model.save('kmeans_model.pkl', 'scaler.pkl')
model = ClusteringModel.load('kmeans_model.pkl', 'scaler.pkl')
```

### 4. **ClassificationModel** (`models/classification.py`)

Random Forest classifier wrapper:

```python
model = ClassificationModel(model_type='RandomForest')
metrics = model.train(X_df, y_series, feature_names=[...])

# Predictions
predictions = model.predict(X_new)  # Returns status labels
probabilities = model.predict_proba(X_new)  # Returns probabilities

# Feature importance
importance_df = model.get_feature_importance(top_n=10)

# Serialization
model.save('classifier.pkl', 'scaler.pkl', 'encoder.pkl')
model = ClassificationModel.load('classifier.pkl', 'scaler.pkl', 'encoder.pkl')
```

### 5. **PredictionAPI** (`api/prediction.py`)

Unified interface for frontend integration:

```python
api = PredictionAPI()
api.load_models()  # Loads all 4 .pkl files

# Single prediction
result = api.predict_single_village(iks=0.75, ike=0.65, ikl=0.70)
# Returns: {iks, ike, ikl, idm, clustering, classification}

# Batch predictions
df_results = api.predict_batch(df_with_features)
# Returns: original + predicted_cluster, predicted_status, status_confidence

# Get model info
cluster_info = api.get_cluster_info()
feature_importance = api.get_feature_importance()
```

## 🚀 DEPLOYMENT WORKFLOW

### Step 1: Initial Setup

```bash
cd backend
pip install -r requirements.txt
```

### Step 2: Prepare Data

```bash
# Copy raw data to data/ folder
cp /path/to/IDM_2024.csv data/idm_2024_raw.csv
```

### Step 3: Train Models

```bash
python train_pipeline.py
```

Output:
- `models/kmeans_model.pkl`
- `models/classifier_model.pkl`
- `models/scaler_features.pkl`
- `models/label_encoder.pkl`
- `data/idm_2024_clean.csv`

### Step 4: Integrate with Flask

```python
# app.py
from flask import Flask, request, jsonify
from idm_backend.api import PredictionAPI

app = Flask(__name__)
api = PredictionAPI()
api.load_models()

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    result = api.predict_single_village(
        iks=data['IKS_2024'],
        ike=data['IKE_2024'],
        ikl=data['IKL_2024']
    )
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
```

### Step 5: Frontend Integration

```javascript
// React example
const [prediction, setPrediction] = useState(null);

const fetchPrediction = async (iks, ike, ikl) => {
  const response = await fetch('http://backend:5000/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      IKS_2024: iks,
      IKE_2024: ike,
      IKL_2024: ikl
    })
  });
  const result = await response.json();
  setPrediction(result);
};

// Display results
<div>
  <h3>Cluster: {prediction?.clustering?.cluster}</h3>
  <h3>Status: {prediction?.classification?.predicted_status}</h3>
  <p>Confidence: {prediction?.classification?.confidence?.toFixed(2)}</p>
</div>
```

## 📊 MODEL FILES (.pkl)

| File | Size | Contains |
|------|------|----------|
| `kmeans_model.pkl` | ~500KB | K-Means model (centroids, labels) |
| `classifier_model.pkl` | 10-15MB | Random Forest (300+ decision trees) |
| `scaler_features.pkl` | ~50KB | StandardScaler (mean, std) |
| `label_encoder.pkl` | ~1KB | Status label mapping |

**Total Size: ~15MB** (acceptable for most servers)

## 🔧 CONFIGURATION

Edit `idm_backend/config.py`:

```python
# Clustering
KMEANS_K = 3  # Number of clusters
KMEANS_RANDOM_STATE = 42

# Classification
TEST_SIZE = 0.2  # 20% test data
CV_FOLDS = 5  # 5-fold cross-validation

# Paths
MODELS_DIR = Path('backend/models')
DATA_DIR = Path('backend/data')
```

## 🎯 IMPORTS REFERENCE

### For Training (in train_pipeline.py)
```python
from idm_backend import config, IDMCleaner, FeatureEngineer
from idm_backend.models import ClusteringModel, ClassificationModel
```

### For Prediction (in Flask/API)
```python
from idm_backend.api import PredictionAPI
```

### For Direct Model Usage
```python
from idm_backend.models import ClusteringModel, ClassificationModel
from idm_backend.preprocessing import IDMCleaner, FeatureEngineer
```

## ✅ PRODUCTION CHECKLIST

- [x] Modular code structure
- [x] Model serialization (.pkl format)
- [x] Unified PredictionAPI
- [x] Configuration management
- [x] Feature engineering pipeline
- [x] Data cleaning pipeline
- [x] Cross-validation in training
- [x] Performance metrics
- [ ] Docker containerization (optional)
- [ ] Database logging (optional)
- [ ] Model versioning (MLflow) (optional)
- [ ] Monitoring/alerting (optional)

## 📝 NEXT STEPS

1. **Copy raw data** to `backend/data/idm_2024_raw.csv`
2. **Run training**: `python backend/train_pipeline.py`
3. **Verify models**: Check that 4 .pkl files exist in `backend/models/`
4. **Create Flask API** using PredictionAPI
5. **Connect Frontend** to Flask endpoints
6. **Deploy** to production server

## 📞 TROUBLESHOOTING

**Q: ImportError: No module named 'idm_backend'**
A: Add backend path: `sys.path.insert(0, 'backend')`

**Q: Models won't load**
A: Ensure all 4 .pkl files exist and paths in config.py are correct

**Q: Slow predictions**
A: Models use scikit-learn (fast). If still slow, consider reducing CV_FOLDS during training

**Q: Memory issues during training**
A: Reduce dataset size or use data sampling

---

**Status: READY FOR DEPLOYMENT** ✅

Backend structure is production-ready and can be integrated with any frontend framework (React, Vue, Angular, etc.) via Flask/FastAPI REST API.
