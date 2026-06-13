# IDM 2024 Backend - Unified ML Pipeline

**Production-ready ML architecture with comprehensive documentation and full justifications**

## 📊 Overview

This backend implements a complete, documented ML pipeline for IDM 2024 data processing and predictions:

- **Input**: Raw CSV data (75k+ villages)
- **Output**: 5 trained models in `.pkl` format
- **Architecture**: Single unified pipeline (not fragmented)
- **Documentation**: Justifications for every cleaning & engineering step

## 🏗️ Unified Structure

```
backend/
├── pipeline.py              ⭐ Main pipeline (all 6 phases in one file)
│   ├─ Phase 1: Data Loading & Exploration
│   ├─ Phase 2: Data Cleaning & Preparation  
│   ├─ Phase 3: Feature Engineering
│   ├─ Phase 4: Clustering Pipeline (K-Means)
│   ├─ Phase 5: Classification Pipeline (Random Forest)
│   └─ Phase 6: Model Serialization (.pkl files)
│
├── api.py                   ⭐ Prediction API (for frontend)
│   └─ PredictionAPI class (loads .pkl files & makes predictions)
│
├── models/                  📦 Output models
│   ├── kmeans_model.pkl                  (K-Means clustering)
│   ├── scaler_clustering.pkl             (Feature scaler for clustering)
│   ├── classifier_model.pkl              (Random Forest classifier)
│   ├── scaler_classification.pkl         (Feature scaler for classification)
│   └── label_encoder.pkl                 (Status label encoder)
│
├── data/                    📊 Data files
│   ├── idm_2024_raw.csv                  (Input)
│   └── idm_2024_clean.csv                (Output after cleaning)
│
└── README.md               (This file)
```

## 🔄 Pipeline Phases with Justifications

### Phase 1: Data Loading & Exploration
```python
DataLoader.load_data()
```
- Load raw CSV
- Examine structure, dtypes, memory usage
- Analyze missing values

**Output**: Initial DataFrame shape, missing value report

---

### Phase 2: Data Cleaning & Preparation

#### Step 2.1: Drop Unused Columns
```python
# Drop: Keterangan, Unnamed: 14
```
**JUSTIFICATION**: These columns contain no meaningful information for ML modeling:
- `Keterangan`: Sparse, unstructured notes (only 5k/75k rows)
- `Unnamed: 14`: Artifact from Excel import (only 1 row filled)

**Decision**: Removing reduces noise and memory footprint without losing information.

---

#### Step 2.2: Remove Duplicates
```python
df.drop_duplicates()
```
**JUSTIFICATION**: Duplicate rows violate data integrity assumptions.
- K-Means and Random Forest assume each observation is independent
- Duplicates inflate certain villages' influence on model fitting

**Decision**: Remove completely (count: 0 in this dataset)

---

#### Step 2.3: Handle Missing Values

##### 2.3a Remove rows with ALL indices NaN
```python
mask_all_nan = df[['IKS_2024', 'IKE_2024', 'IKL_2024', 'NILAI_IDM_2024']].isnull().all(axis=1)
```
**JUSTIFICATION**: Villages with no index values are non-active:
- No population (abandoned villages)
- Under land acquisition (HGU)
- Active mining operations
- Cannot be analyzed without any dimension data

**Decision**: Remove completely (preserves causality).

---

##### 2.3b Fill NAMA_DESA with 'Tidak_Diketahui'
```python
df['NAMA_DESA'].fillna('Tidak_Diketahui')
```
**JUSTIFICATION**: Village name is only for identification, doesn't affect numerical analysis.
- Preserves referential integrity for manual lookup
- Placeholder value doesn't introduce bias

**Decision**: Fill with placeholder.

---

##### 2.3c Fill numeric indices with MEDIAN
```python
for col in ['IKS_2024', 'IKE_2024', 'IKL_2024', 'NILAI_IDM_2024']:
    df[col].fillna(df[col].median())
```
**JUSTIFICATION for MEDIAN (not MEAN or forward-fill)**:

1. **Why not MEAN?** 
   - IDM distributions are often left-skewed (many villages in "Tertinggal" status)
   - Single extreme outlier can skew mean significantly
   - Median is robust to skewness

2. **Why not forward-fill?**
   - No temporal structure in data (not time series)
   - Introduces autocorrelation artifacts

3. **Why not deletion?**
   - Would lose 4 valid rows with partial data
   - Reduces sample size for training

**Decision**: Use median (robust, preserves distributional shape).

---

#### Step 2.4: Fix Data Types

**For numeric columns** (`IKS_2024`, `IKE_2024`, `IKL_2024`):
```python
df[col] = pd.to_numeric(df[col], errors='coerce')  # → float64
```
**JUSTIFICATION**: Required for mathematical operations:
- StandardScaler requires float arrays
- K-Means computes Euclidean distances (floating point)
- Random Forest builds splits on numeric ranges

---

**For code columns** (`KODE_PROV`, `KODE_DESA`):
```python
df[col] = df[col].astype(str).str.strip()  # → string
```
**JUSTIFICATION**: Prevent numeric interpretation:
- `KODE_DESA = 1101012001` has leading digit
- If treated as integer: `1101012001` → loses leading zeros
- Causes JOIN failures when matching against reference data
- Example: `01` becomes `1`, matching fails

**Decision**: Keep as strings to preserve structure.

---

#### Step 2.5: Ordinal Encode STATUS_IDM_2024
```python
STATUS_MAP = {
    'SANGAT TERTINGGAL': 1,
    'TERTINGGAL': 2,
    'BERKEMBANG': 3,
    'MAJU': 4,
    'MANDIRI': 5
}
```
**JUSTIFICATION for ORDINAL (not one-hot)**:
- Status classes are hierarchical (1 < 2 < 3 < 4 < 5)
- This hierarchy is semantically meaningful
- Random Forest can split on this ordering
- Reduces dimensionality (1 column vs 5 columns)

**Decision**: Use ordinal encoding.

---

### Phase 3: Feature Engineering

#### Feature 1: `dimensi_terendah` (Bottleneck Identification)
```python
def get_lowest_dimension(row):
    min_idx = min(row['IKS_2024'], row['IKE_2024'], row['IKL_2024'])
    return 'Sosial' if row['IKS_2024'] == min_idx else ...
```
**PURPOSE**: Answer "What's this village's main development constraint?"

**Use Cases**:
- Policy makers: Target interventions to lowest dimension
- Clustering: Profiles villages by constraint type
- Classification: Feature may predict status

**Example**: 
- Village A: IKS=0.8, IKE=0.6, IKL=0.7 → `dimensi_terendah='Ekonomi'`
- Village B: IKS=0.5, IKE=0.8, IKL=0.7 → `dimensi_terendah='Sosial'`

---

#### Features 2-4: Dimension Gaps (`gap_iks_ike`, `gap_iks_ikl`, `gap_ike_ikl`)
```python
df['gap_iks_ike'] = df['IKS_2024'] - df['IKE_2024']
df['gap_iks_ikl'] = df['IKS_2024'] - df['IKL_2024']
```
**PURPOSE**: Quantify inequality between dimensions

**Interpretation**:
- Large positive gap: Social development outpaces economic
- Large negative gap: Economic crisis with strong social fabric
- Small gap: Balanced development

**Example**:
- Village with gap_iks_ike = 0.25: Social much stronger than economy
  - Insight: Economy is bottleneck (not social support)
  - Suggests economic policies would help most

---

### Phase 4: Clustering Pipeline (K-Means)

**Input Features**: `['IKS_2024', 'IKE_2024', 'IKL_2024']`

#### Why K-Means?
```python
Why not Hierarchical? Slower on 75k+ data
Why not DBSCAN?     Doesn't work well with equal-sized clusters
Why K-Means?        Fast, interpretable centroids, proven
```

#### Why StandardScaler?
```python
X_scaled = StandardScaler().fit_transform(X)
```
**JUSTIFICATION**: Without scaling:
- IKE typically ranges [0.1, 1.0] (range=0.9)
- IKS typically ranges [0.5, 1.0] (range=0.5)
- Euclidean distance: dist = sqrt((iks_diff)² + (ike_diff)² + (ikl_diff)²)
- Larger range IKE overwhelms smaller range IKS
- Solution: Standardize all features to mean=0, std=1

**Result**: Each dimension contributes equally to distance

---

#### K=3 Justification
```python
Elbow Method + Silhouette Score determined k=3 as optimal
```

---

### Phase 5: Classification Pipeline (Random Forest)

**Input Features**: `['IKS_2024', 'IKE_2024', 'IKL_2024']`
**Target**: `STATUS_IDM_2024` (5 classes)

#### Why Random Forest?
```python
✓ Handles non-linear relationships (IKS→STATUS not linear)
✓ Feature importance ranking (answer: which dimension matters most?)
✓ Robust to outliers (no distance-based assumptions)
✓ Handles class imbalance naturally via class_weight='balanced'
```

#### Why Stratified K-Fold?
```python
StratifiedKFold(n_splits=5, shuffle=True)
```
**JUSTIFICATION**: Maintains class distribution in each fold
- Some villages are rare (e.g., "Mandiri" only ~5%)
- Random split might put all "Mandiri" in test set
- Stratified ensures even rare classes in train AND test

---

### Phase 6: Model Serialization (.pkl files)

**Why pickle?**
```python
✓ Python native format (scikit-learn objects)
✓ Preserves full model state (centroids, trees, scalers)
✓ Fast loading
✗ Not human-readable (but that's OK for binary models)
```

**Generated Files**:
| File | Size | Contains |
|------|------|----------|
| `kmeans_model.pkl` | ~500KB | K-Means centroids, labels, n_clusters |
| `scaler_clustering.pkl` | ~50KB | mean_, scale_, n_features |
| `classifier_model.pkl` | 10-15MB | 200 decision trees |
| `scaler_classification.pkl` | ~50KB | mean_, scale_ |
| `label_encoder.pkl` | ~1KB | Class labels mapping |

---

## 🚀 Usage

### 1️⃣ Prepare Data
```bash
cp /path/to/IDM_2024.csv backend/data/idm_2024_raw.csv
```

### 2️⃣ Train Models
```bash
cd backend
python pipeline.py
```

**Output**: 5 .pkl files in `models/` folder

### 3️⃣ Use Predictions (in Flask/FastAPI)
```python
from api import PredictionAPI

api = PredictionAPI()
api.load_models()

# Single prediction
result = api.predict_single(iks=0.75, ike=0.65, ikl=0.70)
print(result)
# {
#   'input': {...},
#   'clustering': {'cluster': 2, 'confidence_scores': [...], ...},
#   'classification': {'predicted_status': 'BERKEMBANG', 'confidence': 0.92, ...}
# }

# Batch prediction
df_results = api.predict_batch(df_with_features)
```

---

## 🔗 Integration Examples

### Flask API
```python
from flask import Flask, request, jsonify
from api import PredictionAPI

app = Flask(__name__)
api = PredictionAPI()
api.load_models()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    result = api.predict_single(
        iks=data['IKS_2024'],
        ike=data['IKE_2024'],
        ikl=data['IKL_2024']
    )
    return jsonify(result)
```

### React Frontend
```javascript
const predictVillage = async (iks, ike, ikl) => {
  const response = await fetch('http://backend:5000/predict', {
    method: 'POST',
    body: JSON.stringify({ IKS_2024: iks, IKE_2024: ike, IKL_2024: ikl })
  });
  return response.json();
};

// Usage
const result = await predictVillage(0.75, 0.65, 0.70);
console.log('Cluster:', result.clustering.cluster);
console.log('Status:', result.classification.predicted_status);
```

---

## ✅ Quality Checklist

- [x] **Documented**: Every cleaning/engineering step has justification
- [x] **Reproducible**: Fixed random_state for all models
- [x] **Tested**: Cross-validation, train/test split, metrics
- [x] **Modular**: Single pipeline.py + api.py
- [x] **Serialized**: 5 .pkl model files ready for deployment
- [x] **Scalable**: Handles 75k+ villages efficiently

---

## 📝 Notes

1. **No hardcoded paths**: All paths in `Config` class
2. **No magic numbers**: All parameters explained in comments
3. **No duplicated code**: Single pipeline, two branches (clustering vs classification)
4. **Production-ready**: Error handling, logging, clear output

---

**Status**: ✅ Ready for deployment to frontend
