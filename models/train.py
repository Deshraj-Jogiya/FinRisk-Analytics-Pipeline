# Risk Model Training Script using XGBoost and Scikit-Learn
import os
import sqlite3
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score
import xgboost as xgb

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.db")
MODEL_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(MODEL_DIR, "risk_model.pkl")
CHART_PATH = os.path.join(MODEL_DIR, "feature_importance.png")

def load_data():
    print("Loading data from database...")
    conn = sqlite3.connect(DB_PATH)
    
    # Query features and targets
    query = """
        SELECT 
            c.credit_score, 
            c.annual_income, 
            c.employment_years, 
            c.debt_amount,
            l.loan_amount, 
            l.term_months, 
            l.loan_purpose, 
            l.interest_rate, 
            l.dti_ratio,
            l.decision_status
        FROM loan_applications l
        JOIN customers c ON l.customer_id = c.customer_id
        WHERE l.decision_status IN ('APPROVED', 'REJECTED')
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"Loaded {len(df)} historical applications for training.")
    return df

def train():
    df = load_data()
    
    # Preprocessing
    # Target variable: APPROVED = 1, REJECTED = 0
    df['target'] = (df['decision_status'] == 'APPROVED').astype(int)
    
    X = df.drop(columns=['decision_status', 'target'])
    y = df['target']
    
    # Identify numeric and categorical features
    numeric_features = ['credit_score', 'annual_income', 'employment_years', 'debt_amount', 'loan_amount', 'term_months', 'interest_rate', 'dti_ratio']
    categorical_features = ['loan_purpose']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Define preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    
    # 1. Random Forest Classifier
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, max_depth=6))
    ])
    
    print("Training Random Forest Classifier...")
    rf_pipeline.fit(X_train, y_train)
    rf_preds = rf_pipeline.predict(X_test)
    rf_probs = rf_pipeline.predict_proba(X_test)[:, 1]
    
    print("\n--- Random Forest Classifier Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, rf_preds):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, rf_probs):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, rf_preds))
    
    # 2. XGBoost Classifier (for comparison and actual use if preferred)
    # XGBoost requires preprocessed inputs, so we fit the preprocessor first
    preprocessor.fit(X_train)
    X_train_encoded = preprocessor.transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)
    
    # Get feature names after encoding
    cat_encoder = preprocessor.named_transformers_['cat']
    encoded_cat_features = list(cat_encoder.get_feature_names_out(categorical_features))
    feature_names = numeric_features + encoded_cat_features
    
    xgb_clf = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, eval_metric='logloss')
    print("Training XGBoost Classifier...")
    xgb_clf.fit(X_train_encoded, y_train)
    
    xgb_preds = xgb_clf.predict(X_test_encoded)
    xgb_probs = xgb_clf.predict_proba(X_test_encoded)[:, 1]
    
    print("\n--- XGBoost Classifier Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, xgb_preds):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, xgb_probs):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, xgb_preds))
    
    # Save the pipeline (preprocessing + Random Forest classifier is simpler for serving)
    print(f"Saving Random Forest model pipeline to {MODEL_PATH}...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(rf_pipeline, f)
        
    # Plot feature importance for Random Forest
    print("Generating feature importance chart...")
    classifier = rf_pipeline.named_steps['classifier']
    importances = classifier.feature_importances_
    
    # Create DataFrame for plotting
    feat_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # Apply modern style
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Premium color gradient
    colors = sns.color_palette("viridis", len(feat_imp_df))
    sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette=colors)
    
    plt.title('Credit Risk Prediction — Feature Importance (Random Forest)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Gini Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    plt.savefig(CHART_PATH, dpi=300)
    plt.close()
    print(f"Feature importance chart saved to {CHART_PATH}.")

if __name__ == '__main__':
    train()
