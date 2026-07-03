# Credit Risk Analysis & Prediction

A concise Data Science and Machine Learning project focused on analyzing consumer loan data to predict repayment performance. The core objective is to classify loan applicants into 'Paid' (Non-Default) and 'Unpaid' (Default) categories to assist in financial risk management.

## 📊 Dataset Overview

The project utilizes the `credit_risk_dataset.csv` file, a tabular dataset containing historical customer financial profiles.

*   **Features:** Demographics (age, income), employment history (length of employment, home ownership status), loan specifications (intent, grade, amount, interest rate), and historical credit records.
*   **Target Variable:** `loan_status` where 0 indicates the loan was successfully paid, and 1 indicates default/unpaid.

## ⚙️ Project Pipeline

The entire pipeline is implemented within `project.ipynb` divided into the following phases:

*   **Data Preprocessing & Cleaning:** Deduplication, handling missing data, and filtering out logical anomalies (e.g., age > 100 or employment length > 60 years).
*   **Exploratory Data Analysis (EDA):** Key relationship visualizations including loan-to-income distributions, default behaviors by home ownership and loan intent, and alignment checks on bank risk grades.
*   **Feature Engineering:** Categorical variable transformation using One-Hot Encoding and numerical standardization using `StandardScaler`.
*   **Model Training & Evaluation:** Splitting the dataset into 80% training and 20% stratified testing sets to train and evaluate multiple classification algorithms.

## 🤖 Machine Learning Models & Results

Four distinct classification models were trained and evaluated side-by-side using high-resolution confusion matrices:

1.  **Random Forest:** The top-performing model, demonstrating superior robustness.
2.  **Decision Tree:** Optimized with a max depth of 6 to capture non-linear decision boundaries effectively.
3.  **K-Nearest Neighbors (KNN):** Configured with 11 neighbors to smoothly map customer risk distributions.
4.  **Logistic Regression:** Used as a baseline classifier, illustrating the limitations of linear boundaries.
