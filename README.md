# SP25 Taxi Demand Forecasting - MLOps Project

## Overview
An end-to-end MLOps solution for predicting NYC taxi demand in real-time using historical data, automated pipelines, and live visualizations. Combines batch processing with upcoming streaming integration for operational efficiency and model monitoring.

## Architecture

### Part 1: Static Data Pipeline (Model Development)
- **Data Source:** NYC Yellow Taxi Trip Data (2023)
- **Core Components:**
  - Data ingestion and validation
  - Hourly time series aggregation
  - Feature engineering with sliding windows
  - Model training (XGBoost/LightGBM) with MLflow tracking
  - Hyperparameter optimization

### Part 2: Automated Batch Pipeline
- **MLOps Infrastructure:**
  - Hopsworks (Feature Store + Model Registry)
  - GitHub Actions (hourly scheduling)
  - Prediction storage and monitoring
- **Key Metrics:**
  - Hourly MAE calculation
  - Prediction performance history

### Part 3: Real-Time Visualization
- **Streamlit Dashboards:**
  - Public: Interactive NYC taxi zone map with predictions
  - Internal: MAE tracking and model performance
- **Visualization Tools:**
  - Plotly for trends
  - Folium for geospatial data

## Tech Stack
- **Core:** Python 3.9+
- **ML:** XGBoost, LightGBM, Scikit-learn
- **Data:** Pandas, NumPy, GeoPandas
- **MLOps:** Hopsworks, MLflow, GitHub Actions
- **Viz:** Streamlit, Plotly, Folium

## Project Structure

## Quick Start
1. **Environment Setup:**
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

2. **Configure Secrets (.env):**
HOPSWORKS_API_KEY=your_api_key
MLFLOW_TRACKING_URI=your_mlflow_uri

3. **Run Pipelines:**
Feature pipeline
python src/feature_pipeline.py

Model training
python src/model_training_pipeline.py

Inference
python src/inference_pipeline.py


4. **Launch Dashboards:**
streamlit run streamlit/frontend_v2.py
streamlit run streamlit/frontend_monitor.py


## Roadmap
- **Part 4 (Upcoming):**
  - AWS Kinesis integration for streaming data
  - Real-time prediction updates
  - Cloud-native deployment

## License
MIT License © 2024 Vasu Krishna
