import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle
import os

# Create output directory for model results
if not os.path.exists('model_outputs'):
    os.makedirs('model_outputs')

print("====== Customer Churn Random Forest Model ======")

# 1. Data Loading and Preprocessing
def load_and_preprocess_data():
    print("\n1. Loading and preprocessing data...")
    
    # Load datasets
    train_data = pd.read_csv('dataset\customer_churn_dataset-testing-master.csv')
    test_data = pd.read_csv('dataset\customer_churn_dataset-training-master.csv')
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Testing data shape: {test_data.shape}")
    
    # Check for missing values
    train_missing = train_data.isnull().sum().sum()
    test_missing = test_data.isnull().sum().sum()
    print(f"Missing values in training data: {train_missing}")
    print(f"Missing values in testing data: {test_missing}")
    
    # Handle missing values if any
    if train_missing > 0:
        train_data = train_data.dropna()
        print(f"Training data shape after removing missing values: {train_data.shape}")
    
    if test_missing > 0:
        test_data = test_data.dropna()
        print(f"Testing data shape after removing missing values: {test_data.shape}")
    
    # Data preprocessing
    def preprocess_dataset(data):
        # Create a copy to avoid modifying the original data
        df = data.copy()
        
        # Drop non-predictive features
        if 'CustomerID' in df.columns:
            df = df.drop('CustomerID', axis=1)
        
        # One-hot encode categorical variables
        categorical_cols = ['Gender', 'Subscription Type', 'Contract Length']
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
        
        return df
    
    # Preprocess both datasets
    train_processed = preprocess_dataset(train_data)
    test_processed = preprocess_dataset(test_data)
    
    # Feature engineering (based on insights from the data analysis)
    def engineer_features(data):
        df = data.copy()
        
        # Recent activity ratio
        if 'Last Interaction' in df.columns and 'Tenure' in df.columns:
            df['Recent_Activity_Ratio'] = df['Last Interaction'] / (df['Tenure'] + 1)
        
        # Support calls per tenure
        if 'Support Calls' in df.columns and 'Tenure' in df.columns:
            df['Support_Calls_Per_Tenure'] = df['Support Calls'] / (df['Tenure'] + 1)
        
        # Payment delay ratio
        if 'Payment Delay' in df.columns and 'Tenure' in df.columns:
            df['Payment_Delay_Ratio'] = df['Payment Delay'] / (df['Tenure'] + 1)
        
        # Spending per tenure
        if 'Total Spend' in df.columns and 'Tenure' in df.columns:
            df['Spend_Per_Tenure'] = df['Total Spend'] / (df['Tenure'] + 1)
        
        return df
    
    # Apply feature engineering
    train_engineered = engineer_features(train_processed)
    test_engineered = engineer_features(test_processed)
    
    # Ensure both datasets have the same columns
    # This handles cases where test data might be missing some categories present in training
    train_cols = set(train_engineered.columns)
    test_cols = set(test_engineered.columns)
    
    # Add missing columns to test data
    for col in train_cols - test_cols:
        if col != 'Churn':  # Don't add the target column
            test_engineered[col] = 0
    
    # Add missing columns to train data
    for col in test_cols - train_cols:
        if col != 'Churn':  # Don't add the target column
            train_engineered[col] = 0
    
    # Ensure column order is the same
    common_cols = [col for col in train_engineered.columns if col != 'Churn']
    train_features = train_engineered[common_cols]
    test_features = test_engineered[common_cols]
    
    # Separate features and target
    X_train = train_features
    y_train = train_engineered['Churn']
    X_test = test_features
    y_test = test_engineered['Churn']
    
    print(f"Number of features: {X_train.shape[1]}")
    
    return X_train, y_train, X_test, y_test, common_cols

# 2. Build and train the Random Forest model
def build_random_forest_model(X_train, y_train):
    print("\n2. Building and training Random Forest model...")
    
    # Initialize the base Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=100,        # Number of trees
        max_depth=None,          # Maximum depth of trees
        min_samples_split=2,     # Minimum samples required to split
        min_samples_leaf=1,      # Minimum samples required at leaf node
        max_features='sqrt',     # Number of features to consider for best split
        bootstrap=True,          # Whether to use bootstrap samples
        random_state=42,         # Random seed for reproducibility
        n_jobs=-1                # Use all available cores
    )
    
    # Train the model
    rf_model.fit(X_train, y_train)
    
    print("Base Random Forest model trained")
    
    return rf_model

# 3. Evaluate the model
def evaluate_model(model, X_test, y_test, X_train, y_train, feature_names):
    print("\n3. Evaluating Random Forest model...")
    
    # Make predictions on train and test data
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate probabilities for ROC curve
    y_train_prob = model.predict_proba(X_train)[:, 1]
    y_test_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate evaluation metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    train_precision = precision_score(y_train, y_train_pred)
    test_precision = precision_score(y_test, y_test_pred)
    
    train_recall = recall_score(y_train, y_train_pred)
    test_recall = recall_score(y_test, y_test_pred)
    
    train_f1 = f1_score(y_train, y_train_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    
    # Print evaluation results
    print("\nModel Performance:")
    print(f"                   Training   Testing")
    print(f"Accuracy:          {train_accuracy:.4f}     {test_accuracy:.4f}")
    print(f"Precision:         {train_precision:.4f}     {test_precision:.4f}")
    print(f"Recall:            {train_recall:.4f}     {test_recall:.4f}")
    print(f"F1 Score:          {train_f1:.4f}     {test_f1:.4f}")
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('model_outputs/confusion_matrix.png')
    
    # Generate classification report
    class_report = classification_report(y_test, y_test_pred)
    print("\nClassification Report:")
    print(class_report)
    
    # Save classification report to file
    with open('model_outputs/classification_report.txt', 'w') as f:
        f.write("Classification Report:\n")
        f.write(class_report)
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_test_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('model_outputs/roc_curve.png')
    
    # Plot feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
    plt.title('Top 15 Feature Importance')
    plt.tight_layout()
    plt.savefig('model_outputs/feature_importance.png')
    
    # Save feature importance to file
    feature_importance.to_csv('model_outputs/feature_importance.csv', index=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    return feature_importance, test_accuracy, test_f1

# 4. Hyperparameter tuning
def tune_random_forest(X_train, y_train, X_test, y_test):
    print("\n4. Tuning Random Forest hyperparameters...")
    
    # Define a simple parameter grid for grid search
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 20],
        'min_samples_split': [2, 5]
    }
    
    # Initialize the base Random Forest model
    rf = RandomForestClassifier(random_state=42)
    
    # Initialize grid search
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,                 # 3-fold cross-validation
        scoring='f1',         # Optimize for F1 score
        n_jobs=-1,            # Use all available cores
        verbose=1
    )
    
    # Fit the grid search
    print("Running grid search (this may take a while)...")
    grid_search.fit(X_train, y_train)
    
    # Get the best parameters
    best_params = grid_search.best_params_
    print("\nBest Parameters:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    # Evaluate the best model
    y_pred = best_model.predict(X_test)
    best_accuracy = accuracy_score(y_test, y_pred)
    best_f1 = f1_score(y_test, y_pred)
    
    print(f"\nBest model accuracy: {best_accuracy:.4f}")
    print(f"Best model F1 score: {best_f1:.4f}")
    
    return best_model

# 5. Main function
def main():
    print("\n====== Starting Customer Churn Prediction with Random Forest ======")
    
    # 1. Load and preprocess data
    X_train, y_train, X_test, y_test, feature_names = load_and_preprocess_data()
    
    # 2. Build and train the base model
    base_model = build_random_forest_model(X_train, y_train)
    
    # 3. Evaluate the base model
    base_importance, base_accuracy, base_f1 = evaluate_model(
        base_model, X_test, y_test, X_train, y_train, feature_names
    )
    
    # 4. Optional: Tune the model (comment out if not needed)
    print("\nWould you like to tune the model? (y/n)")
    tune_option = input().lower()
    
    if tune_option == 'y':
        tuned_model = tune_random_forest(X_train, y_train, X_test, y_test)
        print("\nEvaluating tuned model...")
        tuned_importance, tuned_accuracy, tuned_f1 = evaluate_model(
            tuned_model, X_test, y_test, X_train, y_train, feature_names
        )
        
        # Compare base and tuned models
        print("\nModel Comparison:")
        print(f"Base model - Accuracy: {base_accuracy:.4f}, F1: {base_f1:.4f}")
        print(f"Tuned model - Accuracy: {tuned_accuracy:.4f}, F1: {tuned_f1:.4f}")
        
        # Save the best model
        best_model = tuned_model if tuned_f1 > base_f1 else base_model
    else:
        best_model = base_model
    
    # 5. Save the best model
    with open('model_outputs/random_forest_churn_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    print("\nBest model saved as 'model_outputs/random_forest_churn_model.pkl'")
    
    # 6. Extract the most important features for future reference
    top_features = pd.DataFrame({
        'Feature': feature_names,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    top_features.head(10).to_csv('model_outputs/top_features.csv', index=False)
    print("\nTop features saved to 'model_outputs/top_features.csv'")
    
    print("\n====== Customer Churn Prediction Completed ======")

# Run the main function if this script is executed
if __name__ == "__main__":
    main()