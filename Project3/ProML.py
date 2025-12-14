# =========================================
# CREDIT RISK PREDICTION - SINGLE FILE
# =========================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib

from flask import Flask, request, jsonify

# =========================================
# 1. LOAD DATA
# =========================================
print("Loading data...")
train = pd.read_csv('cs-training.csv')

# =========================================
# 2. EDA
# =========================================
def comprehensive_eda(df):
    print("Running EDA...")
    df = df.copy()

    fig, ax = plt.subplots(2, 2, figsize=(15, 12))

    sns.kdeplot(
        data=df,
        x='RevolvingUtilizationOfUnsecuredLines',
        hue='SeriousDlqin2yrs',
        fill=True,
        common_norm=False,
        ax=ax[0, 0]
    )
    ax[0, 0].set_title('Credit Utilization by Risk')

    sns.boxplot(
        data=df,
        x='SeriousDlqin2yrs',
        y='DebtRatio',
        showfliers=False,
        ax=ax[0, 1]
    )
    ax[0, 1].set_title('Debt Ratio Distribution')

    df['AgeGroup'] = pd.cut(df['age'], bins=[20,30,40,50,60,70,80,90])
    risk_by_age = df.groupby('AgeGroup', observed=True)['SeriousDlqin2yrs'].mean()

    sns.barplot(
        x=risk_by_age.index.astype(str),
        y=risk_by_age.values,
        ax=ax[1, 0]
    )
    ax[1, 0].set_title('Default Risk by Age Group')
    ax[1, 0].tick_params(axis='x', rotation=45)

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
    plt.savefig('financial_risk_analysis_visualization.png', dpi=300)
    plt.close()

    with open('eda_report.txt', 'w') as f:
        f.write("Skewness:\n")
        f.write(df.select_dtypes('number').skew().to_string())
        f.write("\n\nKurtosis:\n")
        f.write(df.select_dtypes('number').kurtosis().to_string())
        f.write("\n\nANOVA (Age vs Default):\n")
        f.write(str(stats.f_oneway(
            df[df['SeriousDlqin2yrs'] == 0]['age'],
            df[df['SeriousDlqin2yrs'] == 1]['age']
        )))

    print("EDA files saved.")

comprehensive_eda(train)

# =========================================
# 3. FEATURE ENGINEERING
# =========================================
print("Feature engineering...")

train['MonthlyIncome'].fillna(train['MonthlyIncome'].median(), inplace=True)
train['NumberOfDependents'].fillna(0, inplace=True)

train['PaymentBurden'] = train['MonthlyIncome'] / (train['DebtRatio'] + 1e-6)
train['RecentDelinquency'] = (
    train['NumberOfTime30-59DaysPastDueNotWorse'] * 0.5 +
    train['NumberOfTime60-89DaysPastDueNotWorse'] * 0.7 +
    train['NumberOfTimes90DaysLate']
)
train['AgeDebtInteraction'] = train['age'] / (train['DebtRatio'] + 1e-6)

# =========================================
# 4. MODEL TRAINING
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

X = train[features]
y = train['SeriousDlqin2yrs']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        random_state=42
    ))
])

print("Training model...")
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print(classification_report(y_test, y_pred))

joblib.dump(pipeline, 'credit_risk_model.pkl')
print("Model saved.")

# =========================================
# 5. API (SAME FILE)
# =========================================
print("Starting API...")

app = Flask(__name__)
model = joblib.load('credit_risk_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    proba = model.predict_proba(df)[0][1]

    return jsonify({
        'risk_score': float(proba),
        'risk_category': (
            'High' if proba > 0.7 else
            'Medium' if proba > 0.3 else
            'Low'
        )
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
