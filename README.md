# Customer Churn Prediction Using K-Nearest Neighbors (KNN)

## Overview

This project develops a **Customer Churn Prediction** model using the **K-Nearest Neighbors (KNN)** machine learning algorithm. The objective is to identify customers who are likely to discontinue a company's service based on historical customer information.

The project follows a complete end-to-end machine learning workflow, including:

* Exploratory Data Analysis (EDA)
* Data Cleaning
* Feature Engineering
* Feature Scaling
* Model Training
* Hyperparameter Tuning
* Model Evaluation
* Business Insights

The notebook is designed to be beginner-friendly, with detailed comments explaining each step of the implementation.

---

## Objectives

* Predict whether a customer will churn.
* Prepare raw customer data for machine learning.
* Train a KNN classification model.
* Determine the optimal value of **K** through hyperparameter tuning.
* Evaluate model performance using standard classification metrics.
* Generate business insights from the prediction results.

---

## Dataset

This project uses the **Customer Churn Dataset**, which contains customer demographic information, account details, and service usage attributes.

The target variable is:

* **Churn**

  * Yes = Customer left the service
  * No = Customer remained with the service

Typical features include:

* Customer demographics
* Account information
* Service subscriptions
* Billing information
* Contract type
* Tenure
* Monthly charges
* Total charges

---

## Technologies Used

* Python
* Google Colab
* Pandas
* NumPy
* Matplotlib
* Scikit-learn

---

## Machine Learning Workflow

### 1. Exploratory Data Analysis (EDA)

* Dataset overview
* Missing value analysis
* Data type inspection
* Summary statistics

### 2. Data Cleaning

* Handle missing values
* Remove inconsistencies
* Prepare data for modeling

### 3. Feature Engineering

* Separate features and target variable
* Encode categorical variables
* Prepare model-ready data

### 4. Feature Scaling

Since KNN is a distance-based algorithm, feature scaling is performed using **StandardScaler** to ensure all variables contribute equally during distance calculations.

### 5. Model Training

A baseline **K-Nearest Neighbors Classifier** is trained using an initial value of **K = 5**.

### 6. Model Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* Classification Report

### 7. Hyperparameter Tuning

Different values of **K** are tested to determine the optimal number of nearest neighbors that produces the highest prediction accuracy.

### 8. Business Insights

The final section translates machine learning results into practical business insights, including customer churn trends and potential retention strategies.

---

## Project Structure

```
customer-churn-prediction-knn/
│
├── Level_1_Task3_Customer_Churn_Prediction_Using_KNN.ipynb
├── dataset/
├── outputs/
│   ├── charts/
│   ├── tables/
│   └── reports/
├── README.md
└── LICENSE
```

---

## Results

The project successfully:

* Built a customer churn prediction model using KNN.
* Identified the optimal value of **K** through hyperparameter tuning.
* Evaluated classification performance using multiple metrics.
* Produced visualizations that aid in understanding model performance.
* Generated business-oriented insights that can support customer retention strategies.

---

## Key Learning Outcomes

Through this project, I gained practical experience in:

* Data preprocessing
* Exploratory Data Analysis
* Feature engineering
* Feature scaling
* Classification using KNN
* Hyperparameter tuning
* Model evaluation
* Translating machine learning results into actionable business insights

---

## Future Improvements

Possible enhancements include:

* Testing additional machine learning algorithms such as Logistic Regression, Decision Trees, Random Forest, Support Vector Machine (SVM), and XGBoost.
* Applying cross-validation for more robust model evaluation.
* Addressing class imbalance using techniques such as SMOTE.
* Performing feature selection to improve model performance.
* Deploying the trained model as a web application using Streamlit or Flask.

---

## Author

**Lady Jean Y. Geverola**

BS Data Science Student
University of the Philippines Mindanao

This repository was developed as part of a machine learning project to demonstrate practical skills in customer churn prediction using the K-Nearest Neighbors (KNN) algorithm.
