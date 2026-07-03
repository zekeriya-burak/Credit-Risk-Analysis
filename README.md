Credit Risk Analysis & Prediction
A concise Data Science and Machine Learning project focused on analyzing consumer loan data to predict repayment performance. The core objective is to classify loan applicants into 'Paid' (Non-Default) and 'Unpaid' (Default) categories to assist in financial risk management.

📊 Dataset Overview
The project utilizes the Credit Risk Dataset, a tabular dataset containing historical customer financial profiles.

Features: Demographics (age, income), employment history (length of employment, home ownership status), loan specifications (intent, grade, amount, interest rate), and historical credit records.
Target Variable: loan_status where 0 indicates the loan was successfully paid, and 1 indicates default/unpaid.
⚙️ Project Pipeline
The entire pipeline is implemented within a single, streamlined Jupyter Notebook (deneme.ipynb) divided into the following phases:

Data Preprocessing & Cleaning: Deduplication, handling missing data, and filtering out logical anomalies (e.g., age > 100 or employment length > 60 years).
Exploratory Data Analysis (EDA): Key relationship visualizations including loan-to-income distributions, default behaviors by home ownership and loan intent, and alignment checks on bank risk grades.
Feature Engineering: Categorical variable transformation using One-Hot Encoding (pd.get_dummies with drop_first=True) and numerical standardization using StandardScaler.
Model Training & Evaluation: Splitting the dataset into 80% training and 20% stratified testing sets to train and evaluate multiple classification algorithms.
🤖 Machine Learning Models & Results
Four distinct classification models were trained and evaluated side-by-side using high-resolution confusion matrices:

Random Forest: The top-performing model, demonstrating superior robustness and the highest balance across all key metrics (Accuracy, Precision, Recall, and F1-Score).
Decision Tree: Optimized with a max depth of 6 to capture non-linear decision boundaries effectively without overfitting.
K-Nearest Neighbors (KNN): Configured with 11 neighbors to smoothly map customer risk distributions.
Logistic Regression: Used as a baseline classifier, illustrating the limitations of linear boundaries on non-linear financial interaction data.
