# Churn Enterprise Prediction System

An end-to-end customer churn prediction platform built with Python, Scikit-Learn, and Flask.

## Overview
This system enables enterprise organizations to predict customer churn risks using machine learning classification pipelines (such as Random Forest and XGBoost) and provides an interactive web interface for churn scoring and insights.

## Project Structure
- `app.py`: Flask web application for inference and UI.
- `build_pipeline.py`: Model pipeline construction, feature transformation, and training workflow.
- `preprocessing.py`: Feature engineering and data cleaning utilities.
- `understand.ipynb`: Exploratory data analysis (EDA) and model experimentation notebook.
- `models/`: Saved model pipelines and preprocessing artifacts.
- `data/`: Raw and processed dataset files.
- `templates/` & `static/`: Frontend interface assets.

## Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/raohuzaifa6262-design/Churn_Enterprise.git
   cd Churn_Enterprise
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```
3. Install required packages:
   ```bash
   pip install flask scikit-learn pandas numpy xgboost
   ```
4. Run the web application:
   ```bash
   python app.py
   ```
