# =========================================
# CREDIT RISK PREDICTION 
# =========================================

import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib

from flask import Flask, request, jsonify

# =========================================
# CONFIG FLAGS
# =========================================
RUN_TRAINING = True      # Set False when only running API
MODEL_PATH = 'credit_risk_model.pkl'

# =========================================
# FEATURE ENGINEERING 
# =========================================
def feature_engineering(df):
    df = df.copy()

    # Fill missing values safely
    df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())
    df['NumberOfDependents'] = df['NumberOfDependents'].fillna(0)

    # Create features
    df['PaymentBurden'] = df['MonthlyIncome'] / (df['DebtRatio'] + 1e-6)
    df['RecentDelinquency'] = (
        df['NumberOfTime30-59DaysPastDueNotWorse'] * 0.5 +
        df['NumberOfTime60-89DaysPastDueNotWorse'] * 0.7 +
        df['NumberOfTimes90DaysLate']
    )
    df['AgeDebtInteraction'] = df['age'] / (df['DebtRatio'] + 1e-6)

    return df

# =========================================
# FEATURES
# =========================================
features = [
    'RevolvingUtilizationOfUnsecuredLines',
    'age',
    'DebtRatio',
    'MonthlyIncome',
    'NumberOfOpenCreditLinesAndLoans',
    'NumberOfTimes90DaysLate',
    'PaymentBurden',
    'RecentDelinquency',
    'AgeDebtInteraction'
]

# =========================================
# MODEL TRAINING
# =========================================
if RUN_TRAINING:
    print("Loading training data...")
    train = pd.read_csv('cs-training.csv')

    print("Running training pipeline...")
    train_fe = feature_engineering(train)

    X = train_fe[features]
    y = train_fe['SeriousDlqin2yrs']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', RandomForestClassifier(
            n_estimators=200,
            class_weight='balanced',
            random_state=42
        ))
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("ROC-AUC:", roc_auc_score(y_test, y_proba))
    print(classification_report(y_test, y_pred))

    joblib.dump(pipeline, MODEL_PATH)
    print("Model saved:", MODEL_PATH)

# =========================================
# SAFE MODEL LOADING
# =========================================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file not found. Set RUN_TRAINING=True to train it first.")

model = joblib.load(MODEL_PATH)

# =========================================
# API
# =========================================
print("Starting API...")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'Empty JSON payload'}), 400

        df = pd.DataFrame([data])

        # Validate required raw fields
        required_raw = [
            'RevolvingUtilizationOfUnsecuredLines',
            'age',
            'DebtRatio',
            'MonthlyIncome',
            'NumberOfOpenCreditLinesAndLoans',
            'NumberOfTimes90DaysLate',
            'NumberOfTime30-59DaysPastDueNotWorse',
            'NumberOfTime60-89DaysPastDueNotWorse',
            'NumberOfDependents'
        ]

        missing = set(required_raw) - set(df.columns)
        if missing:
            return jsonify({'error': f'Missing fields: {list(missing)}'}), 400

        # Feature engineering
        df = feature_engineering(df)
        df = df[features]

        # Prediction
        proba = model.predict_proba(df)[0][1]

        return jsonify({
            'risk_score': float(proba),
            'risk_category': (
                'High' if proba > 0.7 else
                'Medium' if proba > 0.3 else
                'Low'
            )
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =========================================
# RUN SERVER
# =========================================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


