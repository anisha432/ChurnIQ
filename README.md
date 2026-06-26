# 🚀 **ChurnIQ**

## **AI-Powered Customer Churn Data Analytics Platform**

---

# 📌 **What is ChurnIQ?**

**ChurnIQ** is an AI-powered data analytics platform that analyzes telecom customer data to uncover churn patterns, predict which customers are likely to leave, and explain **why** they are at risk using Machine Learning.

Unlike traditional prediction systems, ChurnIQ combines **Data Analytics, Business Intelligence, Machine Learning, and Explainable AI (SHAP)** to generate actionable insights that help organizations reduce customer churn and improve retention strategies.

---

# 📊 **Data Analytics Capabilities**

## 🔍 **Data Profiling**

Automatically profiles uploaded datasets by displaying:

* Total rows & columns
* Data types
* Missing values
* Duplicate records
* Memory usage
* Dataset summary

---

## 🧹 **Data Cleaning Pipeline**

Automatically performs:

* Handles missing values in **TotalCharges**
* Removes duplicate records
* Strips whitespace from categorical columns
* Converts incorrect data types
* Cleans the dataset for model training

---

## ⚙️ **Feature Engineering**

Generates **8+ engineered analytical features**, including:

* Tenure Group
* Charges Per Tenure
* Total Services Count
* Premium Support Flag
* Contract Type Flag
* Auto Payment Flag
* Internet Service Flag
* Monthly-to-Total Charges Ratio

---

## 📈 **Exploratory Data Analysis (EDA)**

Includes **10+ interactive visualizations**, such as:

* Customer Churn Distribution
* Tenure vs Churn
* Monthly Charges Analysis
* Contract Type Impact
* Internet Service Comparison
* Payment Method Risk
* Demographic Analysis
* Service-Level Churn Breakdown
* Correlation Heatmap
* Scatter Plots with Trendlines

---

## 👥 **Segment Analysis**

Analyzes churn across customer segments:

* Contract Type
* Internet Service
* Payment Method
* Senior Citizens
* Partner Status
* Tenure Groups

---

## 📊 **KPI Analytics**

Automatically calculates executive-level KPIs:

* Overall Churn Rate
* Monthly Contract Churn
* New Customer Churn
* Senior Citizen Churn
* Fiber Optic Churn
* Revenue at Risk
* Annual Revenue Impact

---

# 🤖 **Machine Learning Module**

## 🏋️ **Model Training**

Trains and compares two Machine Learning models:

* Logistic Regression
* Random Forest

Both models use **balanced class weights** to improve churn prediction.

---

## 📉 **Model Evaluation**

Evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score
* 5-Fold Cross Validation

---

## 🏆 **Best Model Selection**

Automatically:

* Selects the highest ROC-AUC model
* Saves the trained model
* Stores preprocessing artifacts
* Enables instant predictions

---

## 🎯 **Achieved Performance**

* **ROC-AUC:** **84.7%**
* **Dataset:** Telco Customer Churn
* **Customers:** **7,043**
* **Features:** **28+ engineered features**

---

# 🔮 **Prediction & Explainability**

## 📝 **Customer Prediction**

Users can enter customer information including:

* Demographics
* Services
* Billing Details

and instantly receive:

* Churn Prediction
* Probability Score
* Confidence Score

---

## 🚨 **Risk Categorization**

Predictions are classified into:

* 🟢 Low (0–20%)
* 🟡 Medium (20–40%)
* 🟠 High (40–65%)
* 🔴 Critical (65–100%)

---

## 🎯 **Confidence Scoring**

Measures prediction confidence based on the distance from the **0.5 decision threshold**.

---

## 🔍 **Key Risk Factors**

Automatically identifies the major factors contributing to customer churn.

---

## 🧠 **SHAP Explainability**

Provides full model explainability using SHAP:

* SHAP Summary Beeswarm Plot
* Feature Importance Bar Chart
* Individual Waterfall Plot
* Dependence Plots

Understand **exactly why** a customer is predicted to churn.

---

# 💼 **Business Intelligence**

## 💡 **AI-Generated Insights**

Automatically generates **10 prioritized business insights**.

Priority Levels:

* 🔴 Critical
* 🟠 High
* 🟡 Medium
* 🔵 Strategic

Each insight includes:

* Data Finding
* Business Impact
* Recommended Action

---

## 📊 **Executive KPI Dashboard**

Displays:

* Total Customers
* Churned Customers
* Churn Rate
* Average Tenure
* Average Monthly Charges
* Revenue at Risk
* Annual Revenue Impact

---

## 🎯 **Segment-Level Recommendations**

Provides personalized retention strategies for:

* Month-to-Month Customers
* New Customers
* Fiber Optic Users
* Electronic Check Customers
* Senior Citizens

---

## 📄 **Downloadable Reports**

Export:

* Cleaned Dataset
* Text-Based Business Insight Report

Perfect for presentations and offline analysis.

---

# 🛠️ **Tech Stack**

| Technology   | Purpose                     |
| ------------ | --------------------------- |
| Streamlit    | Interactive Web Application |
| Scikit-Learn | Machine Learning Models     |
| Plotly       | Interactive Visualizations  |
| SHAP         | Explainable AI              |
| Pandas       | Data Processing             |
| NumPy        | Numerical Computing         |
| Matplotlib   | Visualization               |
| Seaborn      | Statistical Charts          |
| Joblib       | Model Serialization         |

---

# 🚀 **How to Run**

### **1️⃣ Clone the Repository**

```bash
git clone https://github.com/anisha432/ChurnIQ.git

cd ChurnIQ
```

---

### **2️⃣ Install Dependencies**

```bash
pip install -r requirements.txt

pip install statsmodels
```

---

### **3️⃣ Train the Model**

```bash
python train_model.py
```

---

### **4️⃣ Launch the Application**

```bash
streamlit run app.py
```

---

### **5️⃣ Open in Browser**

```
http://localhost:8501
```

---

# 📂 **Project Structure**

```text
ChurnIQ/

├── app.py
├── utils.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore

├── data/

└── models/
    ├── model.pkl
    ├── scaler.pkl
    ├── feature_columns.pkl
    ├── numerical_cols.pkl
    ├── training_results.pkl
    └── best_model_name.pkl
```

---

# 📜 **License**

This project is licensed under the **MIT License**.

You are free to use, modify, and distribute this project.
