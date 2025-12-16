# =========================================
# CREDIT RISK PREDICTION 
# =========================================

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
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
RUN_EDA = True
RUN_TRAINING = True
RUN_API = True

MODEL_PATH = 'credit_risk_model.pkl'

# =========================================
# LOAD DATA (ONLY ONCE)
# =========================================
print("Loading data...")
data = pd.read_csv('cs-training.csv')

# =========================================
# 1. EDA + VISUALIZATION
# =========================================
def run_eda(df):
    print("Running EDA...")

    os.makedirs("eda_outputs", exist_ok=True)

    fig, ax = plt.subplots(2, 2, figsize=(16, 12))

    # Credit Utilization
    sns.kdeplot(
        data=df,
        x='RevolvingUtilizationOfUnsecuredLines',
        hue='SeriousDlqin2yrs',
        fill=True,
        ax=ax[0, 0]
    )
    ax[0, 0].set_title('Credit Utilization vs Default')

    # Debt Ratio
    sns.boxplot(
        data=df,
        x='SeriousDlqin2yrs',
        y='DebtRatio',
        showfliers=False,
        ax=ax[0, 1]
    )
    ax[0, 1].set_title('Debt Ratio Distribution')

    # Age Risk
    df['AgeGroup'] = pd.cut(df['age'], bins=[20,30,40,50,60,70,80,90])
    risk_by_age = df.groupby('AgeGroup', observed=True)['SeriousDlqin2yrs'].mean()

    sns.barplot(
        x=risk_by_age.index.astype(str),
        y=risk_by_age.values,
        ax=ax[1, 0]
    )
    ax[1, 0].set_title('Default Risk by Age Group')
    ax[1, 0].tick_params(axis='x', rotation=45)

    # Late Payments
    late = (
        df['NumberOfTime30-59DaysPastDueNotWorse'] +
        df['NumberOfTime60-89DaysPastDueNotWorse'] +
        df['NumberOfTimes90DaysLate']
    )

    sns.countplot(
        x=late[late > 0],
        hue=df.loc[late > 0, 'SeriousDlqin2yrs'],
        ax=ax[1, 1]
    )
    ax[1, 1].set_title('Late Payments vs Default')

    plt.tight_layout()
    plt.savefig('eda_outputs/financial_risk_analysis.png', dpi=300)
    plt.close()

    # Statistical report
    with open('eda_outputs/eda_report.txt', 'w') as f:
        f.write("SKEWNESS:\n")
        f.write(df.select_dtypes('number').skew().to_string())
        f.write("\n\nKURTOSIS:\n")
        f.write(df.select_dtypes('number').kurtosis().to_string())
        f.write("\n\nANOVA (Age vs Default):\n")
        f.write(str(stats.f_oneway(
            df[df['SeriousDlqin2yrs'] == 0]['age'],
            df[df['SeriousDlqin2yrs'] == 1]['age']
        )))

    print("EDA complete → saved to /eda_outputs")

if RUN_EDA:
    run_eda(data)

# =========================================
# 2. FEATURE ENGINEERING
# =========================================
def feature_engineering(df):
    df = df.copy()

    df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())
    df['NumberOfDependents'] = df['NumberOfDependents'].fillna(0)

    df['PaymentBurden'] = df['MonthlyIncome'] / (df['DebtRatio'] + 1e-6)
    df['RecentDelinquency'] = (
        df['NumberOfTime30-59DaysPastDueNotWorse'] * 0.5 +
        df['NumberOfTime60-89DaysPastDueNotWorse'] * 0.7 +
        df['NumberOfTimes90DaysLate']
    )
    df['AgeDebtInteraction'] = df['age'] / (df['DebtRatio'] + 1e-6)

    return df

# =========================================
# 3. FEATURES
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
# 4. MODEL TRAINING
# =========================================
if RUN_TRAINING:
    print("Training model...")

    df_fe = feature_engineering(data)

    X = df_fe[features]
    y = df_fe['SeriousDlqin2yrs']

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
# 5. LOAD MODEL SAFELY
# =========================================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model not found. Train it first.")

model = joblib.load(MODEL_PATH)

# =========================================
# 6. API
# =========================================
if RUN_API:
    print("Starting API...")

    app = Flask(__name__)

    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            data = request.get_json()
            df = pd.DataFrame([data])

            df = feature_engineering(df)
            df = df[features]

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

    if __name__ == '__main__':
        app.run(host='0.0.0.0', port=5000)
