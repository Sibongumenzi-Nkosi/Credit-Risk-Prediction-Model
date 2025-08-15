# ================================
# CREDIT RISK PREDICTION PIPELINE 
# ================================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
train = pd.read_csv('cs-training.csv')
test = pd.read_csv('cs-test.csv')

# ===================
# 1.  EDA FUNCTION
# ===================
def comprehensive_eda(df):
    # Create copy
    print("Running EDA...")
    df_eda = df.copy()
    
    # 1. Financial Profile Analysis
    fig, ax = plt.subplots(2, 2, figsize=(15, 12))
    
    # Credit Utilization Plot
    sns.kdeplot(data=df_eda, x='RevolvingUtilizationOfUnsecuredLines', 
                hue='SeriousDlqin2yrs', ax=ax[0,0], 
                fill=True, common_norm=False, palette='viridis')
    ax[0,0].set_title('Credit Utilization by Risk Status')
    
    # Debt Ratio Plot
    sns.boxplot(data=df_eda, x='SeriousDlqin2yrs', y='DebtRatio',
                hue='SeriousDlqin2yrs', ax=ax[0,1], 
                showfliers=False, palette='coolwarm', legend=False)
    ax[0,1].set_title('Debt Ratio Distribution')
    
    # 2. Demographic Analysis
    df_eda['AgeGroup'] = pd.cut(df_eda['age'], bins=[20,30,40,50,60,70,80,90])
    risk_by_age = df_eda.groupby('AgeGroup', observed=True)['SeriousDlqin2yrs'].mean().reset_index()
    
    sns.barplot(data=risk_by_age, x='AgeGroup', y='SeriousDlqin2yrs',
                hue='AgeGroup', ax=ax[1,0], palette='rocket', legend=False)
    ax[1,0].set_title('Default Risk by Age Group')
    
    # 3. Payment Behavior Analysis
    late_payments = df_eda[['NumberOfTime30-59DaysPastDueNotWorse',
                           'NumberOfTime60-89DaysPastDueNotWorse',
                           'NumberOfTimes90DaysLate']].sum(axis=1)
    sns.countplot(x=late_payments[late_payments > 0], 
                 hue=df_eda['SeriousDlqin2yrs'], ax=ax[1,1], palette='mako')
    ax[1,1].set_title('Late Payment History vs Default Risk')
    
    plt.tight_layout()
    plt.savefig('financial_risk_analysis_visualization.png', dpi=300)
    print("")
    print("Visualization completed & saved to the file as financial_risk_analysis_visualization.")
    print("")

    
    # 4. Statistical Analysis Report 
    numeric_cols = df_eda.select_dtypes(include=['number']).columns
    with open('eda_report.txt', 'w') as f:
        f.write(f"Skewness Analysis:\n{df_eda[numeric_cols].skew().to_string()}\n\n")
        f.write(f"Kurtosis Analysis:\n{df_eda[numeric_cols].kurtosis().to_string()}\n\n")
        f.write("ANOVA Test (Age vs Default):\n")
        f.write(str(stats.f_oneway(
            df_eda[df_eda['SeriousDlqin2yrs']==0]['age'],
            df_eda[df_eda['SeriousDlqin2yrs']==1]['age']
        )))

# Execute EDA
comprehensive_eda(train)
print("EDA completed, report saved to the file as eda_report.")

# =======================
# 2. FEATURE ENGINEERING
# =======================
# Handle missing values first
train['MonthlyIncome'].fillna(train['MonthlyIncome'].median(), inplace=True)
train['NumberOfDependents'].fillna(0, inplace=True)

# Create features
train['PaymentBurden'] = train['MonthlyIncome'] / (train['DebtRatio'] + 1e-6)
train['CreditUsageStability'] = train['NumberOfOpenCreditLinesAndLoans'] / (train['NumberRealEstateLoansOrLines'] + 1)
train['RecentDelinquency'] = (
    train['NumberOfTime30-59DaysPastDueNotWorse'] * 0.5 +
    train['NumberOfTime60-89DaysPastDueNotWorse'] * 0.7 +
    train['NumberOfTimes90DaysLate'] * 1.0
)
train['IncomeRisk'] = train['MonthlyIncome'] * train['SeriousDlqin2yrs']
train['AgeDebtInteraction'] = train['age'] / (train['DebtRatio'] + 1e-6)

# ==================
# 3. MODEL PIPELINE 
# ==================
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Define features
features = ['RevolvingUtilizationOfUnsecuredLines', 'age', 'DebtRatio',
           'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
           'NumberOfTimes90DaysLate', 'PaymentBurden']

# Train-test split
X = train[features]
y = train['SeriousDlqin2yrs']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline
print("\nBuilding model pipeline...")
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(class_weight='balanced', n_estimators=100))
])

# Train model
print("Training model...")
pipeline.fit(X_train, y_train)

# Evaluate
from sklearn.metrics import classification_report, roc_auc_score
y_pred = pipeline.predict(X_test)
print("Model Performance:")
print(f"ROC-AUC: {roc_auc_score(y_test, pipeline.predict_proba(X_test)[:,1]):.4f}")
print(classification_report(y_test, y_pred))

# Save model
import joblib
joblib.dump(pipeline, 'credit_risk_model.txt')
print("Model saved as 'credit_risk_model.txt'")

# ==================
# 4. API DEPLOYMENT
# ==================
print("\nPreparing API deployment...")
try:
    api_code = """
from flask import Flask, request, jsonify
import joblib
import pandas as pd
from waitress import serve

app = Flask(__name__)
model = joblib.load('credit_risk_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        df = pd.DataFrame([data])
        proba = model.predict_proba(df)[0][1]
        return jsonify({
            'risk_score': float(proba),
            'risk_category': 'High' if proba > 0.7 else 'Medium' if proba > 0.3 else 'Low'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("API running at http://0.0.0.0:5000")
    serve(app, host="0.0.0.0", port=5000)
"""
    requirements = """flask==2.0.1
    waitress==2.1.2
    pandas==1.3.4
    scikit-learn==1.0.2
    joblib==1.1.0
    """

    with open('Project3.txt', 'w') as f:
        f.write(api_code)
    with open('Project3.txt', 'w') as f:
        f.write(requirements)
    
    

    print("API files are created in Project3")
    print("To run the API:")
    print("1. cd api")
    print("2. pip install Project3")
    print("3. python ProML.py")



except Exception as e:
    print(f"API setup error: {str(e)}")

print("\nPipeline execution complete!")
