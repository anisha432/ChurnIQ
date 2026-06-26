🔮 ChurnIQ
AI-Powered Customer Churn Data Analytics Platform

What is ChurnIQ?
ChurnIQ is a data analytics platform that analyzes telecom customer data to uncover churn patterns, predict which customers are likely to leave, and explain why — all powered by machine learning. It goes beyond simple prediction by providing deep data-driven insights that business teams can act on immediately.

Data Analytics Capabilities
Data ProfilingAutomatic dataset profiling — row counts, column types, missing values, duplicates, and memory usage at a glance.

Data Cleaning PipelineHandles missing values in TotalCharges, strips whitespace from categorical columns, removes duplicates, and converts data types automatically.

Feature EngineeringCreates 8 new analytical features from raw data — tenure grouping, charges per tenure, service count, premium support flag, contract type flag, auto-payment flag, internet service flag, and monthly-to-total charges ratio.

Exploratory Data Analysis10+ interactive visualizations that reveal hidden patterns — churn distribution, tenure vs churn, monthly charges analysis, contract type impact, internet service comparison, payment method risk, demographic analysis, service-level churn breakdown, correlation heatmap, and scatter plots with trendlines.

Segment AnalysisChurn rate breakdown by customer segments — contract type, internet service, payment method, senior citizen status, partner status, and tenure groups.

KPI AnalyticsExecutive-level KPIs calculated automatically — overall churn rate, monthly contract churn, new customer churn, senior citizen churn, fiber optic churn, revenue at risk, and annual revenue impact.

Machine Learning Module
Model TrainingTwo models trained and compared — Logistic Regression and Random Forest with balanced class weights.

Model EvaluationCompared on Accuracy, Precision, Recall, F1 Score, ROC-AUC, and 5-fold Cross-Validation.

Best Model SelectionAutomatically selects the best performing model based on ROC-AUC score and saves it for instant predictions.

Achieved Performance 84.7% ROC-AUC on the Telco Customer Churn dataset with 7,043 customer records and 28+ engineered features.

Prediction & Explainability
Customer Prediction FormEnter any customer's details — demographics, services, billing — and get an instant churn prediction with probability score.

Risk CategorizationEvery prediction is classified into four risk levels — Low (0-20%), Medium (20-40%), High (40-65%), and Critical (65-100%).

Confidence ScoringShows how confident the model is by measuring distance from the 0.5 decision threshold.

Key Risk FactorsAutomatically identifies which specific factors are driving that customer's churn risk.

SHAP ExplainabilityFull SHAP analysis — summary beeswarm plot showing global feature impact, bar chart of mean absolute SHAP values, waterfall plot for individual predictions, and dependence plots for top features. Understand exactly why each customer is predicted to churn.

Business Intelligence
AI-Generated Insights10 automatically generated business insights ranked by priority — Critical, High, Medium, and Strategic. Each insight includes the data finding and a specific recommended action.

Executive KPI DashboardKey business metrics at a glance — total customers, churned customers, churn rate, average tenure, average charges, revenue at risk, and annual revenue impact.

Segment-Level RecommendationsSpecific retention strategies for each customer segment — month-to-month customers, new customers, fiber optic users, electronic check payers, and senior citizens.

Downloadable ReportsExport cleaned datasets and text-based insight reports for offline analysis and presentations.

Tech Stack
Streamlit — Interactive web application
Scikit-Learn — Machine learning models and preprocessing
Plotly — Interactive data visualizations
SHAP — Model explainability
Pandas & NumPy — Data processing and analysis
Matplotlib & Seaborn — SHAP plot rendering
Joblib — Model serialization

How to Run
1. Clone the repository

git clone https://github.com/YOUR_USERNAME/ChurnIQ.git
cd ChurnIQ

**2. Install dependencies**
pip install -r requirements.txt
pip install statsmodels

**3. Train the model**
python train_model.py


**4. Launch the application**
streamlit run app.py

**5. Open in browser**
http://localhost:8501
