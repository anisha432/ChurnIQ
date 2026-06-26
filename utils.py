"""
ChurnIQ - Core Utility Functions
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
import joblib
import os
import warnings
import urllib.request

warnings.filterwarnings('ignore')

TELCO_DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
LOCAL_DATA_PATH = "data/telco_churn.csv"
MODELS_DIR = "models"

BINARY_COLS = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
MULTI_CAT_COLS = [
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaymentMethod', 'tenure_group'
]
ENGINEERED_BINARY_COLS = ['has_premium_support', 'is_monthly_contract', 'is_auto_payment', 'has_internet']
NUMERICAL_COLS = [
    'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
    'charges_per_tenure', 'num_services', 'monthly_to_total_ratio'
] + ENGINEERED_BINARY_COLS
DROP_COLS = ['customerID', 'Churn']


def load_telco_data():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(LOCAL_DATA_PATH):
        try:
            urllib.request.urlretrieve(TELCO_DATA_URL, LOCAL_DATA_PATH)
            print(f"Dataset downloaded to {LOCAL_DATA_PATH}")
        except Exception as e:
            raise FileNotFoundError(f"Could not download dataset: {e}")
    df = pd.read_csv(LOCAL_DATA_PATH)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_data(df):
    df = df.copy()
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed > 0:
        print(f"Removed {removed} duplicate rows")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    missing_tc = df['TotalCharges'].isna().sum()
    if missing_tc > 0:
        df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'])
        print(f"Filled {missing_tc} missing TotalCharges values")
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    print(f"Data cleaned: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def engineer_features(df):
    df = df.copy()
    df['tenure_group'] = pd.cut(
        df['tenure'], bins=[-1, 12, 24, 48, 60, 999],
        labels=['0-12', '13-24', '25-48', '49-60', '60+']
    ).astype(str)
    df['charges_per_tenure'] = df['TotalCharges'] / (df['tenure'] + 1)
    service_cols = {
        'PhoneService': 'Yes', 'MultipleLines': 'Yes',
        'InternetService': ['DSL', 'Fiber optic'], 'OnlineSecurity': 'Yes',
        'OnlineBackup': 'Yes', 'DeviceProtection': 'Yes', 'TechSupport': 'Yes',
        'StreamingTV': 'Yes', 'StreamingMovies': 'Yes'
    }
    df['num_services'] = 0
    for col, val in service_cols.items():
        if isinstance(val, list):
            df['num_services'] += df[col].isin(val).astype(int)
        else:
            df['num_services'] += (df[col] == val).astype(int)
    df['has_premium_support'] = (
        (df['OnlineSecurity'] == 'Yes') & (df['TechSupport'] == 'Yes')
    ).astype(int)
    df['is_monthly_contract'] = (df['Contract'] == 'Month-to-month').astype(int)
    df['is_auto_payment'] = df['PaymentMethod'].str.contains('automatic', case=False, na=False).astype(int)
    df['has_internet'] = (df['InternetService'] != 'No').astype(int)
    df['monthly_to_total_ratio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)
    print(f"Feature engineering complete: {df.shape[1]} columns")
    return df


def encode_data(df, is_training=True, training_columns=None):
    df = df.copy()
    binary_map = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].map(binary_map).fillna(0).astype(int)
    existing_multi = [c for c in MULTI_CAT_COLS if c in df.columns]
    if existing_multi:
        df = pd.get_dummies(df, columns=existing_multi, drop_first=False, dtype=int)
    if is_training:
        return df
    else:
        if training_columns is not None:
            for col in training_columns:
                if col not in df.columns:
                    df[col] = 0
            df = df[training_columns]
        return df


def preprocess_for_training(df, test_size=0.2, random_state=42):
    df = df.copy()
    y = df['Churn']
    X = df.drop(columns=DROP_COLS, errors='ignore')
    X_encoded = encode_data(X, is_training=True)
    feature_columns = X_encoded.columns.tolist()
    scaler = StandardScaler()
    num_cols_present = [c for c in NUMERICAL_COLS if c in X_encoded.columns]
    X_encoded[num_cols_present] = scaler.fit_transform(X_encoded[num_cols_present])
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Preprocessing complete: Train={X_train.shape[0]}, Test={X_test.shape[0]}, Features={X_train.shape[1]}")
    return X_train, X_test, y_train, y_test, scaler, feature_columns, num_cols_present


def train_logistic_regression(X_train, y_train, random_state=42):
    model = LogisticRegression(max_iter=1000, random_state=random_state, class_weight='balanced', solver='lbfgs', C=1.0)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, random_state=42):
    model = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_split=10, min_samples_leaf=4, random_state=random_state, class_weight='balanced', n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob),
    }
    cm = confusion_matrix(y_test, y_pred)
    metrics['Confusion Matrix'] = cm
    metrics['y_pred'] = y_pred
    metrics['y_prob'] = y_prob
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    metrics['FPR'] = fpr
    metrics['TPR'] = tpr
    cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='roc_auc')
    metrics['CV Mean AUC'] = cv_scores.mean()
    metrics['CV Std AUC'] = cv_scores.std()
    print(f"  {model_name}: Acc={metrics['Accuracy']:.4f}, AUC={metrics['ROC-AUC']:.4f}, F1={metrics['F1 Score']:.4f}")
    return metrics


def get_feature_importance(model, feature_columns, model_name="Model"):
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
    else:
        return pd.DataFrame()
    fi_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': importance,
        'model': model_name
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    return fi_df


def select_best_model(results):
    return max(results, key=lambda x: x['ROC-AUC'])


def save_artifacts(model, scaler, feature_columns, num_cols, results, best_model_name):
    os.makedirs(MODELS_DIR, exist_ok=True)
    artifacts = {
        'model': model, 'scaler': scaler, 'feature_columns': feature_columns,
        'numerical_cols': num_cols, 'results': results, 'best_model_name': best_model_name
    }
    for name, obj in artifacts.items():
        path = os.path.join(MODELS_DIR, f"{name}.pkl")
        joblib.dump(obj, path)
        print(f"  Saved {path}")
    print(f"All artifacts saved to {MODELS_DIR}/")


def load_artifacts():
    artifacts = {}
    files = {
        'model': 'model.pkl', 'scaler': 'scaler.pkl',
        'feature_columns': 'feature_columns.pkl', 'numerical_cols': 'numerical_cols.pkl',
        'results': 'training_results.pkl', 'best_model_name': 'best_model_name.pkl'
    }
    for key, filename in files.items():
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path):
            artifacts[key] = joblib.load(path)
        else:
            return None
    return artifacts


def prepare_single_prediction(input_dict, feature_columns, numerical_cols, scaler):
    df = pd.DataFrame([input_dict])
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'])
    df = engineer_features(df)
    if 'Churn' in df.columns:
        df = df.drop(columns=['Churn'])
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    df_encoded = encode_data(df, is_training=False, training_columns=feature_columns)
    num_present = [c for c in numerical_cols if c in df_encoded.columns]
    df_encoded[num_present] = scaler.transform(df_encoded[num_present])
    return df_encoded


def get_risk_category(probability):
    if probability < 0.2:
        return 'Low', '#00B894', '🟢'
    elif probability < 0.4:
        return 'Medium', '#FDCB6E', '🟡'
    elif probability < 0.65:
        return 'High', '#E17055', '🟠'
    else:
        return 'Critical', '#D63031', '🔴'


def get_confidence_score(probability):
    return min(abs(probability - 0.5) * 2, 1.0)


def calculate_kpis(df):
    total_customers = len(df)
    total_churned = df['Churn'].sum()
    churn_rate = total_churned / total_customers
    avg_tenure = df['tenure'].mean()
    avg_monthly = df['MonthlyCharges'].mean()
    avg_total = df['TotalCharges'].mean()
    monthly_contract = df[df['Contract'] == 'Month-to-month']
    monthly_churn_rate = monthly_contract['Churn'].mean() if len(monthly_contract) > 0 else 0
    new_customers = df[df['tenure'] <= 12]
    new_churn_rate = new_customers['Churn'].mean() if len(new_customers) > 0 else 0
    seniors = df[df['SeniorCitizen'] == 1]
    senior_churn_rate = seniors['Churn'].mean() if len(seniors) > 0 else 0
    fiber = df[df['InternetService'] == 'Fiber optic']
    fiber_churn_rate = fiber['Churn'].mean() if len(fiber) > 0 else 0
    revenue_at_risk = df[df['Churn'] == 1]['MonthlyCharges'].sum()
    return {
        'Total Customers': f"{total_customers:,}",
        'Churned Customers': f"{int(total_churned):,}",
        'Overall Churn Rate': f"{churn_rate:.1%}",
        'Avg Tenure (months)': f"{avg_tenure:.1f}",
        'Avg Monthly Charges': f"${avg_monthly:.2f}",
        'Avg Total Charges': f"${avg_total:.2f}",
        'Monthly Contract Churn': f"{monthly_churn_rate:.1%}",
        'New Customer Churn (≤12mo)': f"{new_churn_rate:.1%}",
        'Senior Citizen Churn': f"{senior_churn_rate:.1%}",
        'Fiber Optic Churn': f"{fiber_churn_rate:.1%}",
        'Monthly Revenue at Risk': f"${revenue_at_risk:,.2f}",
        'Annual Revenue at Risk': f"${revenue_at_risk * 12:,.2f}",
    }


def generate_business_insights(df, feature_importance_df):
    insights = []
    if len(feature_importance_df) > 0:
        top_feature = feature_importance_df.iloc[0]['feature']
        top_importance = feature_importance_df.iloc[0]['importance']
        insights.append({'icon': '⚡', 'title': f'Primary Risk Factor: {top_feature}',
            'detail': f'This feature has the highest importance score ({top_importance:.4f}). Customers with unfavorable values are significantly more likely to churn. Prioritize interventions targeting this dimension.',
            'priority': 'Critical'})
    monthly_df = df[df['Contract'] == 'Month-to-month']
    yearly_df = df[df['Contract'].isin(['One year', 'Two year'])]
    if len(monthly_df) > 0 and len(yearly_df) > 0:
        m_churn = monthly_df['Churn'].mean()
        y_churn = yearly_df['Churn'].mean()
        ratio = m_churn / y_churn if y_churn > 0 else float('inf')
        insights.append({'icon': '📝', 'title': 'Month-to-Month Contracts Drive Churn',
            'detail': f'Month-to-month customers churn at {m_churn:.1%} vs {y_churn:.1%} for long-term contracts — a {ratio:.1f}x difference. Launch aggressive contract conversion campaigns with discounts for annual plans.',
            'priority': 'Critical'})
    new_df = df[df['tenure'] <= 6]
    if len(new_df) > 0:
        n_churn = new_df['Churn'].mean()
        insights.append({'icon': '🆕', 'title': 'New Customers Are Extremely Vulnerable',
            'detail': f'Customers with ≤6 months tenure churn at {n_churn:.1%}. Implement a structured 90-day onboarding program with dedicated support, setup assistance, and early value demonstration.',
            'priority': 'High'})
    loyal_df = df[df['tenure'] > 48]
    if len(loyal_df) > 0:
        l_churn = loyal_df['Churn'].mean()
        insights.append({'icon': '🎯', 'title': 'Loyalty Threshold at ~48 Months',
            'detail': f'Customers beyond 48 months have only {l_churn:.1%} churn rate. Focus retention spend on months 1-24 where the return on investment is highest.',
            'priority': 'High'})
    fiber_df = df[df['InternetService'] == 'Fiber optic']
    dsl_df = df[df['InternetService'] == 'DSL']
    if len(fiber_df) > 0 and len(dsl_df) > 0:
        f_churn = fiber_df['Churn'].mean()
        d_churn = dsl_df['Churn'].mean()
        insights.append({'icon': '🌐', 'title': 'Fiber Optic Customers Churn More Than DSL',
            'detail': f'Fiber optic churn: {f_churn:.1%} vs DSL: {d_churn:.1%}. This is counterintuitive — likely a service quality or pricing issue. Investigate outages, speed delivery vs promise, and competitive pricing.',
            'priority': 'High'})
    no_support = df[(df['OnlineSecurity'] == 'No') & (df['TechSupport'] == 'No')]
    with_support = df[(df['OnlineSecurity'] == 'Yes') & (df['TechSupport'] == 'Yes')]
    if len(no_support) > 0 and len(with_support) > 0:
        ns_churn = no_support['Churn'].mean()
        ws_churn = with_support['Churn'].mean()
        insights.append({'icon': '🛡️', 'title': 'Support Services Are Powerful Retention Tools',
            'detail': f'Customers WITHOUT OnlineSecurity + TechSupport: {ns_churn:.1%} churn. Customers WITH both: {ws_churn:.1%} churn. Bundle these services at a discount — the retention value exceeds the service cost.',
            'priority': 'Medium'})
    seniors = df[df['SeniorCitizen'] == 1]
    non_seniors = df[df['SeniorCitizen'] == 0]
    if len(seniors) > 0 and len(non_seniors) > 0:
        s_churn = seniors['Churn'].mean()
        ns_churn = non_seniors['Churn'].mean()
        insights.append({'icon': '👴', 'title': 'Senior Citizens Need Special Attention',
            'detail': f'Senior churn rate: {s_churn:.1%} vs non-senior: {ns_churn:.1%}. Create senior-friendly support channels, simplified plans, and personalized check-in calls.',
            'priority': 'Medium'})
    ec_df = df[df['PaymentMethod'] == 'Electronic check']
    auto_df = df[df['PaymentMethod'].str.contains('automatic', case=False, na=False)]
    if len(ec_df) > 0 and len(auto_df) > 0:
        ec_churn = ec_df['Churn'].mean()
        auto_churn = auto_df['Churn'].mean()
        insights.append({'icon': '💳', 'title': 'Electronic Check Users Are High Risk',
            'detail': f'Electronic check churn: {ec_churn:.1%} vs auto-payment: {auto_churn:.1%}. Incentivize switching to automatic payments with small discounts.',
            'priority': 'Medium'})
    churned_rev = df[df['Churn'] == 1]['MonthlyCharges'].sum()
    total_rev = df['MonthlyCharges'].sum()
    rev_pct = churned_rev / total_rev * 100 if total_rev > 0 else 0
    insights.append({'icon': '💰', 'title': 'Revenue Impact Analysis',
        'detail': f'Churned customers represent ${churned_rev:,.0f}/month (${churned_rev*12:,.0f}/year) in lost revenue — {rev_pct:.1f}% of total monthly revenue. Even reducing churn by 10% would save ${churned_rev*0.1*12:,.0f}/year.',
        'priority': 'Critical'})
    insights.append({'icon': '📊', 'title': 'Recommended Retention Strategy Framework',
        'detail': '1) IMMEDIATE: Target month-to-month, fiber optic, electronic check customers with contract conversion + auto-pay incentives. 2) SHORT-TERM: Launch 90-day onboarding program for new customers. 3) MEDIUM-TERM: Bundle OnlineSecurity + TechSupport at discounted rates. 4) LONG-TERM: Build a churn prediction scoring system for real-time intervention.',
        'priority': 'Strategic'})
    priority_order = {'Critical': 0, 'High': 1, 'Medium': 2, 'Strategic': 3}
    insights.sort(key=lambda x: priority_order.get(x['priority'], 99))
    return insights


def profile_data(df):
    return {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'numeric_cols': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_cols': len(df.select_dtypes(include=['object']).columns),
        'missing_values': int(df.isnull().sum().sum()),
        'duplicate_rows': int(df.duplicated().sum()),
        'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB"
    }


def get_column_details(df):
    details = []
    for col in df.columns:
        details.append({
            'Column': col, 'Type': str(df[col].dtype),
            'Non-Null': df[col].count(), 'Null': df[col].isnull().sum(),
            'Unique': df[col].nunique(),
            'Sample': str(df[col].dropna().iloc[0]) if len(df[col].dropna()) > 0 else 'N/A'
        })
    return pd.DataFrame(details)