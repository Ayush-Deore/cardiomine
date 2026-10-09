# Project Brief: CardioMine

## 1. Project Overview
**CardioMine** is an end-to-end Data Warehousing and Data Mining (DWM) mini-project designed to analyze heart disease risk. Built around the **UCI Heart Disease (Cleveland) dataset**, the system demonstrates a complete data lifecycle: from extracting and cleaning raw clinical data to storing it in a dimensional data warehouse, and finally applying machine learning algorithms to extract insights and predict patient risk.

## 2. Objectives
- **Data Integration & Warehousing:** Build a robust ETL (Extract, Transform, Load) pipeline to clean raw data and populate a Star Schema data warehouse.
- **OLAP Analytics:** Demonstrate Online Analytical Processing capabilities (Roll-up, Drill-down, Slice, and Dice) to explore multidimensional patient data.
- **Predictive Data Mining:** Train and evaluate multiple classification models to predict the presence of heart disease.
- **Descriptive Data Mining:** Utilize clustering algorithms to identify inherent patient segments and profiles.
- **Interactive Visualization:** Provide a user-friendly, aesthetically pleasing web dashboard for stakeholders to interact with the models and data.

## 3. Key Components & Features

### 3.1. ETL Pipeline (`etl/`)
- **Extract:** Reads the raw dataset (`heart_disease.csv`).
- **Transform:** 
  - Cleans data by dropping duplicates and handling missing values (e.g., dynamically handling missing `thal` and `ca` values).
  - Enriches the dataset by creating categorical bins (e.g., Age Groups, BP Categories, Cholesterol Categories).
  - Maps numerical indicators to human-readable strings.
- **Load:** Saves the processed data to a clean CSV and loads it into a MySQL database formatted as a Star Schema.

### 3.2. Data Mining Models (`mining/`)
- **Classification (Supervised Learning):** 
  - Models used: **Random Forest**, **Logistic Regression**, **J48 (Decision Tree)**, and **Naive Bayes**.
  - Purpose: To predict binary heart disease risk (0 = No Disease, 1 = Disease Present).
- **Clustering (Unsupervised Learning):**
  - Algorithm used: **K-Means Clustering**.
  - Purpose: To group patients into similar profiles based on clinical features, evaluated using the Elbow Method and Silhouette Scores.
- **Regression Analysis:**
  - Algorithm used: **Linear / Polynomial Regression**.
  - Purpose: To analyze relationships between continuous variables (e.g., Age vs. Max Heart Rate).

### 3.3. Interactive Dashboard (`dashboard/app.py`)
Built with Streamlit, the dashboard features a modern **White and Teal** color scheme and includes:
- **Authentication:** A secure login gate (default credentials: `admin` / `12345678`).
- **Data Management:** A sidebar utility allowing users to dynamically upload and process a new raw UCI dataset, instantly updating the entire dashboard.
- **Pages:**
  1. **Overview:** High-level KPIs and demographic distributions (Pie charts, Bar charts).
  2. **OLAP Analytics:** Interactive slicing and dicing of the data warehouse.
  3. **Risk Prediction:** 
     - *Single Patient Entry:* Manual input of clinical parameters with an intuitive risk gauge.
     - *Batch Prediction:* CSV upload for scoring multiple patients at once.
  4. **Patient Clusters:** Visualization of K-Means profiles and feature distributions.
  5. **Model Comparison:** Side-by-side evaluation metrics of all trained classifiers.
  6. **Regression Analysis:** Trend visualizations.

## 4. Technology Stack
- **Language:** Python 3.9+
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn
- **Web Framework:** Streamlit
- **Visualizations:** Plotly
- **Database:** MySQL (SQLAlchemy + PyMySQL)
- **Model Serialization:** Joblib

## 5. Target Audience & Scope
This project is strictly for **academic demonstration** of Data Warehousing and Mining concepts. The risk predictions and cluster profiles are data-driven examples and do not constitute professional medical diagnoses.
