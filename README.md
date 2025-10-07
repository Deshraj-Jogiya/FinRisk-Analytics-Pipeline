# 📊 FinTech Risk & Fraud Command Center

An end-to-end real-time credit risk analytics and transaction fraud detection pipeline designed to simulate automated ingestion, machine learning inference, and analytical dashboard visualization.

---

## 🚀 Features

1. **End-to-End ETL Ingestion Pipeline:**
   * Ingests credit risk profiles and transaction feeds.
   * Centralizes SQL database with structured schemas for customers, applications, and logs.
   * Auto-calculates key indicators such as Debt-to-Income (DTI) ratio.
   
2. **Machine Learning Risk Engine:**
   * Simulates real-time credit decisioning via an ensemble classifier (Random Forest/XGBoost).
   * Generates probability scores for default risk and loan approval.
   * Features unsupervised anomaly scoring on transaction streams to detect potential credit card fraud.

3. **Inference Serving & Command Dashboard:**
   * Features a premium glassmorphic dark-mode dashboard built on FastAPI, Jinja2, and Chart.js.
   * Serves real-time risk predictions through a POST API endpoint.
   * Provides visual analytics tracking approval rates, credit bracket distribution, and transaction anomalies.

---

## 🛠️ Technology Stack

* **Core Language:** Python 3.11+
* **Database Layer:** SQLite3, SQL, SQLAlchemy, aiosqlite
* **Machine Learning:** Scikit-Learn, XGBoost, Pandas, NumPy
* **API Framework:** FastAPI, Uvicorn, Jinja2 Templates
* **Visualization Layer:** Chart.js, Matplotlib, Seaborn

---

## 📁 Repository Structure

```
├── api/
│   ├── app.py                # FastAPI web app and risk endpoints
│   └── templates/
│       ├── index.html        # Glassmorphic responsive dashboard UI
│       └── error.html        # Fallback database error template
├── db/
│   ├── schema.sql            # Database schema for customers, loans, transactions
│   └── queries.sql           # Complex analytical reporting SQL queries
├── etl/
│   └── pipeline.py           # Ingestion pipeline & synthetic data generator
├── models/
│   ├── train.py              # ML classifier training script
│   ├── risk_model.pkl        # Saved Random Forest classifier pipeline
│   └── feature_importance.png # Feature importance chart export
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

---

## 💻 Setup & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Ingestion Pipeline (ETL)
Generate the database schema and populate it with synthetic customer profiles, loan applications, and transaction streams:
```bash
python etl/pipeline.py
```
This generates the SQLite database file `app.db` in the root folder.

### 3. Train Machine Learning Models
Train the Random Forest and XGBoost classifiers, evaluate metrics, and save model binaries:
```bash
python models/train.py
```

### 4. Launch the Command Dashboard
Start the local FastAPI ASGI web server:
```bash
python -m uvicorn api.app:app --reload --host 127.0.0.1 --port 8000
```
Open [http://localhost:8000](http://localhost:8000) in your browser to view the Command Center.

---

## 📊 Analytical SQL Insights
You can find optimized queries in `db/queries.sql` to generate reports for:
* Approval rates across credit score brackets.
* Anomaly streams matching high fraud probabilities.
* Average Debt-to-Income (DTI) by decision status.
* Monthly approval volume and average interest rates.
