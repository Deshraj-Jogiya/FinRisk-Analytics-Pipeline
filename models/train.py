# Default-risk model training: Random Forest vs XGBoost, both tracked in MLflow
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
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import xgboost as xgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.db")
MODEL_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(MODEL_DIR, "risk_model.pkl")
CHART_PATH = os.path.join(MODEL_DIR, "feature_importance.png")

# Applications the model scores at or above this default probability are declined; precision is measured there
DECLINE_THRESHOLD = 0.80

NUMERIC_FEATURES = ['credit_score', 'annual_income', 'employment_years', 'debt_amount', 'prior_delinquencies',
                    'credit_utilization', 'loan_amount', 'term_months', 'interest_rate', 'dti_ratio']
CATEGORICAL_FEATURES = ['loan_purpose']

# Local SQLite-backed tracking store (./models/mlflow.db) -- no tracking
# server required. The plain filesystem store ("file:./mlruns") is in
# MLflow's maintenance mode as of 2.x/3.x; SQLite is the current
# recommended local backend. Set MLFLOW_TRACKING_URI before running this
# script to point at a real server/managed backend instead.
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{os.path.join(MODEL_DIR, 'mlflow.db')}"))
mlflow.set_experiment("finrisk-default-risk")


def load_data():
    """Every loan with its outcome. The simulator's true default probability (risk_score) and the underwriting
    decision are left out: the first is the answer key and the second is a rule over the same inputs."""
    print("Loading data from database...")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT c.credit_score, c.annual_income, c.employment_years, c.debt_amount, c.prior_delinquencies,
               c.credit_utilization, l.loan_amount, l.term_months, l.loan_purpose, l.interest_rate, l.dti_ratio,
               l.defaulted
        FROM loan_applications l
        JOIN customers c ON l.customer_id = c.customer_id
        """,
        conn,
    )
    conn.close()
    print(f"Loaded {len(df)} loans ({df['defaulted'].mean():.1%} defaulted).")
    return df


def decline_metrics(y_true, probs, threshold=DECLINE_THRESHOLD):
    """Precision and recall of declining every application scored at or above the threshold."""
    y_true, flagged = np.asarray(y_true), np.asarray(probs) >= threshold
    precision = float(y_true[flagged].mean()) if flagged.any() else 0.0
    recall = float(y_true[flagged].sum() / y_true.sum()) if y_true.sum() else 0.0
    return {"decline_precision": precision, "decline_recall": recall, "declined_share": float(flagged.mean())}


def evaluate(name, y_test, probs):
    preds = (probs >= 0.5).astype(int)
    m = {"accuracy": accuracy_score(y_test, preds), "roc_auc": roc_auc_score(y_test, probs), **decline_metrics(y_test, probs)}
    print(f"\n--- {name} ---")
    print(f"ROC AUC: {m['roc_auc']:.4f} | accuracy at 0.5: {m['accuracy']:.4f}")
    print(f"Declining scores >= {DECLINE_THRESHOLD:.0%}: precision {m['decline_precision']:.1%}, "
          f"recall {m['decline_recall']:.1%}, {m['declined_share']:.1%} of applications declined")
    print(classification_report(y_test, preds))
    return m


def train():
    df = load_data()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df['defaulted']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), NUMERIC_FEATURES),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES),
    ])

    # 1. Random Forest Classifier
    rf_params = {"n_estimators": 400, "min_samples_leaf": 20, "random_state": 42, "n_jobs": -1}
    rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(**rf_params))])
    print("Training Random Forest Classifier...")
    with mlflow.start_run(run_name="random_forest"):
        rf_pipeline.fit(X_train, y_train)
        rf_metrics = evaluate("Random Forest", y_test, rf_pipeline.predict_proba(X_test)[:, 1])
        mlflow.log_params(rf_params)
        mlflow.log_metrics(rf_metrics)
        # MLflow now saves scikit-learn models with skops, which refuses to load any type not explicitly trusted. The only one this
        # pipeline needs is the node storage of the trees in the Random Forest we just trained ourselves.
        mlflow.sklearn.log_model(rf_pipeline, name="model", skops_trusted_types=["sklearn.tree._tree.Tree"])

    # 2. XGBoost Classifier (for comparison); XGBoost takes the already-encoded matrix
    preprocessor.fit(X_train)
    X_train_encoded, X_test_encoded = preprocessor.transform(X_train), preprocessor.transform(X_test)
    cat_encoder = preprocessor.named_transformers_['cat']
    feature_names = NUMERIC_FEATURES + list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))

    xgb_params = {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05, "random_state": 42, "eval_metric": "logloss"}
    xgb_clf = xgb.XGBClassifier(**xgb_params)
    print("Training XGBoost Classifier...")
    with mlflow.start_run(run_name="xgboost"):
        xgb_clf.fit(X_train_encoded, y_train)
        xgb_metrics = evaluate("XGBoost", y_test, xgb_clf.predict_proba(X_test_encoded)[:, 1])
        mlflow.log_params(xgb_params)
        mlflow.log_metrics(xgb_metrics)
        mlflow.xgboost.log_model(xgb_clf, name="model")

    print(
        f"\nRandom Forest selected for production (ROC AUC {rf_metrics['roc_auc']:.4f}, decline precision "
        f"{rf_metrics['decline_precision']:.1%}) vs. XGBoost (ROC AUC {xgb_metrics['roc_auc']:.4f}, decline precision "
        f"{xgb_metrics['decline_precision']:.1%}) -- both runs logged to MLflow under 'finrisk-default-risk'."
    )
    pd.DataFrame([{"model": "Random Forest", **rf_metrics}, {"model": "XGBoost", **xgb_metrics}]).round(4).to_csv(
        os.path.join(MODEL_DIR, "metrics.csv"), index=False)

    # Save the pipeline (preprocessing + Random Forest classifier is simpler for serving)
    print(f"Saving Random Forest model pipeline to {MODEL_PATH}...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(rf_pipeline, f)

    # Plot feature importance for Random Forest
    print("Generating feature importance chart...")
    importances = rf_pipeline.named_steps['classifier'].feature_importances_
    feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    sns.barplot(x='Importance', y='Feature', data=feat_imp_df, hue='Feature', palette="viridis", legend=False)
    plt.title('Default Risk Model: Feature Importance (Random Forest)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Gini Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    os.makedirs(MODEL_DIR, exist_ok=True)
    plt.savefig(CHART_PATH, dpi=300)
    plt.close()
    print(f"Feature importance chart saved to {CHART_PATH}.")


if __name__ == '__main__':
    train()
