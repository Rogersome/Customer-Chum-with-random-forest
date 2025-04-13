Customer Churn Prediction Project
Project Overview
This project analyzes and predicts customer churn using machine learning techniques. By identifying key factors contributing to customer churn, businesses can develop targeted retention strategies to reduce attrition and improve customer loyalty. The implementation includes both comprehensive data analysis and predictive modeling components.
Dataset
The project uses the Customer Churn Dataset from Kaggle, which contains:

Training Dataset: 440,833 customer records
Testing Dataset: 64,374 customer records

Features include:

CustomerID: Unique identifier for each customer
Age: Customer's age
Gender: Customer's gender (Male/Female)
Tenure: Duration of customer relationship (in months)
Usage Frequency: Service usage frequency
Support Calls: Number of support interactions
Payment Delay: Average payment delay in days
Subscription Type: Service tier (Basic, Standard, Premium)
Contract Length: Contract duration (Monthly, Quarterly, Annual)
Total Spend: Customer spending amount
Last Interaction: Days since last customer interaction
Churn: Target variable (1 = churned, 0 = not churned)

Project Structure
The project is organized into two main components:
1. Data Analysis (DA)
File: customer_churn_da.py
This component focuses on exploratory data analysis, generating visualization outputs stored in the da_outputs folder:

Feature-specific analyses (age, gender, tenure, etc.)
Correlation studies
Customer segmentation analysis
Risk factor visualization
Revenue impact assessment
Comprehensive heat maps and distribution plots

2. Predictive Modeling
File: random_forest_model.py
This component builds and evaluates a Random Forest classifier:

Data preprocessing and feature engineering
Model training and evaluation
Feature importance analysis
Output files stored in model_outputs folder

Results Summary
Model Performance

Accuracy: 58%
Precision (Churn): 98%
Recall (Churn): 27%
F1 Score (Churn): 0.42
AUC: 0.77

Key Findings

Payment behavior is the strongest predictor of churn, with Payment Delay being the most important feature
Support calls strongly correlate with churn risk
Usage frequency shows inverse relationship with churn probability
Contract type significantly impacts churn rates (monthly contracts show higher churn)
Tenure affects churn probability (newer customers are more likely to churn)

Installation & Usage
Prerequisites

Python 3.8+
Required libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, scipy

Setup
bash# Clone or download this repository
# Install required packages
pip install pandas numpy matplotlib seaborn scikit-learn scipy
Running the Analysis
bash# Run data analysis component
python customer_churn_da.py

# Run Random Forest model
python random_forest_model.py
Business Recommendations
Based on the analysis, we recommend:

Improve payment experience:

Implement proactive payment reminders
Offer flexible payment options
Create early warning systems for payment delays


Enhance customer support:

Review support processes for customers with multiple calls
Implement post-interaction satisfaction surveys
Consider dedicated support for high-value customers


Optimize contract strategies:

Review monthly contract offerings
Create incentives for longer-term commitments
Implement special retention programs for first-year customers


Develop targeted interventions:

Implement risk scoring based on identified factors
Create proactive outreach for high-risk customers
Design segment-specific retention offers



Future Improvements

Address class imbalance with techniques like SMOTE
Optimize decision threshold for better business outcomes
Explore ensemble methods and alternative algorithms
Create segment-specific prediction models
Incorporate additional customer behavioral data

Acknowledgments

Customer Churn Dataset from Kaggle by Muhammad Shahid Azeem
RetryClaude does not have internet access. Links provided may not be accurate or up to date.Claude can make mistakes. Please double-check responses.
