# Customer Churn Prediction Project

## Project Overview
This project aims to analyze and predict customer churn using machine learning techniques. By identifying the key factors that contribute to customer churn, businesses can develop targeted retention strategies to reduce customer attrition and improve customer loyalty.

## Dataset Description
The project utilizes two main datasets:
- **Training Dataset**: 440,833 customer records with 12 features
- **Testing Dataset**: 64,374 customer records with the same features

### Features include:
- **CustomerID**: Unique identifier for each customer
- **Age**: Customer's age
- **Gender**: Customer's gender (Male/Female)
- **Tenure**: How long the customer has been with the company (in months)
- **Usage Frequency**: How often the customer uses the service
- **Support Calls**: Number of times the customer has contacted support
- **Payment Delay**: Average number of days the customer delays payment
- **Subscription Type**: Type of subscription (Basic, Standard, Premium)
- **Contract Length**: Length of the customer's contract (Monthly, Quarterly, Annual)
- **Total Spend**: Amount spent by the customer
- **Last Interaction**: Days since the customer's last interaction
- **Churn**: Target variable (1 = churned, 0 = not churned)

## Project Structure
The project is organized into two main components:

### 1. Data Analysis (DA)
The data analysis component focuses on exploring and understanding the dataset to identify patterns and key factors related to customer churn.

**File**: `customer_churn_da.py`

**Key functionalities**:
- Basic data exploration and statistics
- Analysis of numerical and categorical features
- Correlation analysis with churn
- Customer segmentation analysis
- Risk factor analysis
- Customer value analysis
- Generation of key insights and recommendations

### 2. Predictive Modeling
The predictive modeling component builds a Random Forest classifier to predict customer churn based on the identified factors.

**File**: `random_forest_model.py`

**Key functionalities**:
- Data preprocessing and feature engineering
- Random Forest model training
- Model evaluation
- Feature importance analysis
- Optional hyperparameter tuning
- Model saving for future use

## Results Summary

### Model Performance
- **Accuracy**: 58%
- **Precision (Churn)**: 98%
- **Recall (Churn)**: 27%
- **F1 Score (Churn)**: 0.42
- **AUC**: 0.77

### Key Findings
1. **Payment behavior** is the most significant predictor of churn, with Payment Delay being the top feature
2. **Customer support interactions** strongly correlate with churn (higher support calls indicate higher churn risk)
3. **Usage frequency** is inversely related to churn risk
4. **Contract type** impacts churn rates significantly, with monthly contracts showing higher churn than annual contracts
5. **Customer tenure** is an important factor, with newer customers more likely to churn

## Getting Started

### Prerequisites
- Python 3.8+
- Required libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - scipy

### Installation
```bash
# Clone this repository
git clone https://github.com/Rogersome/Customer-Chum-with-random-forest.git

# Navigate to the project directory
cd customer-churn-prediction

# Install required packages
pip install -r requirements.txt
```

### Usage

#### Data Analysis
```bash
python customer_churn_da.py
```
This will generate visualizations and insights in the `da_outputs` directory.

#### Random Forest Model
```bash
python random_forest_model.py
```
This will train the model, evaluate its performance, and save the model in the `model_outputs` directory.

## Model Interpretation
The Random Forest model shows:
- High precision but low recall for churn prediction
- Strong ability to identify key churn factors
- Reasonable discriminative ability (AUC = 0.77)
- Class imbalance issues that could be addressed with resampling techniques

## Business Recommendations
Based on the analysis, we recommend:

1. **Improve payment experience**:
   - Implement proactive payment reminders
   - Offer flexible payment options for customers with payment delays
   - Create early warning systems for increasing payment delays

2. **Enhance customer support**:
   - Review support processes for customers with multiple calls
   - Implement satisfaction surveys after support interactions
   - Consider dedicated account managers for high-value customers with support issues

3. **Adjust contract strategies**:
   - Review monthly contract offerings and incentives
   - Create targeted offers to move monthly customers to longer contracts
   - Implement special retention programs for first-year customers

4. **Implement value-based retention**:
   - Prioritize retention efforts for high-value segments
   - Develop targeted upgrade paths for medium-value customers
   - Consider special loyalty programs for premium customers

5. **Develop early intervention programs**:
   - Implement risk scoring based on identified factors
   - Create proactive outreach for customers with increasing risk scores
   - Develop targeted offers based on specific risk factors

## Future Improvements
- Explore additional models (XGBoost, LightGBM) for potentially better performance
- Address class imbalance with techniques like SMOTE
- Optimize prediction thresholds for better business outcomes
- Create segment-specific models for more targeted predictions
- Incorporate additional data sources if available

## License

## Acknowledgments
- Customer Churn Dataset from Kaggle by Muhammad Shahid Azeem
