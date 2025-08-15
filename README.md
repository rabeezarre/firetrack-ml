# FireTrack-ML — Wildfire Risk Prediction

Machine learning module for the [FireTrack](https://github.com/rabeezarre/firetrack-project) system, used to estimate **wildfire risk** from environmental data.  
It combines **temperature**, **humidity**, and **altitude** features with an **unsupervised clustering model** to classify locations into **Low / Medium / High** risk.

## Features
- Fetches historical weather data from **Open-Meteo** for training.
- Preprocesses and scales features for clustering.
- Exports a trained **K-Means** model and scaler for API or offline use.
- Provides a FastAPI service for real-time predictions.

## Quick Start

**Install dependencies**
```bash
pip install -r requirements.txt
# or
pip install numpy pandas scikit-learn joblib requests fastapi uvicorn pydantic
```

**Run API**
```bash
uvicorn api.app:app --reload --port 8001
```

**Example request**
```bash
curl "http://localhost:8001/predict?temperature=32.5&humidity=18&altitude=450"
# {"risk":"High","cluster":2,"distance":0.27}
```

## Training
1. Fetch weather data:
```bash
python scripts/fetch_open_meteo.py --lat 43.7 --lon 7.2 --start 2023-03-01 --end 2023-09-01 --out data.csv
```
2. Train and export model:
```bash
python scripts/train.py --input data.csv --model-out models/risk_kmeans.joblib --scaler-out models/feature_scaler.joblib
```

## Integration
This module is consumed by the FireTrack backend to provide risk scores for map locations and trigger alerts.
