import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette('viridis')

# 1. Load and examine data
print("Loading datasets...")
train_data = pd.read_csv('dataset\customer_churn_dataset-testing-master.csv')
test_data = pd.read_csv('dataset\customer_churn_dataset-training-master.csv')

# Create output directory
import os
if not os.path.exists('da_outputs'):
    os.makedirs('da_outputs')

# 2. Basic dataset statistics
def basic_stats(data, name):
    print(f"\n==== {name} Dataset Stats ====")
    print(f"Shape: {data.shape}")
    print(f"Churn rate: {data['Churn'].mean():.2%}")
    
    # Missing values
    missing = data.isnull().sum()
    if missing.sum() > 0:
        print("\nMissing values:")
        print(missing[missing > 0])
    else:
        print("\nNo missing values found")
    
    # Numerical feature stats
    print("\nNumerical feature statistics:")
    num_stats = data.describe()
    print(num_stats)
    
    # Categorical feature counts
    print("\nCategorical feature counts:")
    cat_cols = ['Gender', 'Subscription Type', 'Contract Length']
    for col in cat_cols:
        print(f"\n{col}:")
        print(data[col].value_counts())
        print(f"% distribution:")
        print(data[col].value_counts(normalize=True).map(lambda x: f"{x:.2%}"))

# Run basic stats on both datasets
basic_stats(train_data, "Training")
basic_stats(test_data, "Testing")

# 3. Churn Analysis by Feature
print("\n==== Churn Analysis by Feature ====")

# Function to analyze numerical features
def analyze_numerical_features(data):
    num_features = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 
                   'Payment Delay', 'Total Spend', 'Last Interaction']
    
    # Mean by churn status
    churn_means = data.groupby('Churn')[num_features].mean()
    print("\nMean values by churn status:")
    print(churn_means)
    
    # Percent difference
    percent_diff = (churn_means.loc[1] - churn_means.loc[0]) / churn_means.loc[0] * 100
    print("\nPercent difference (Churned vs Non-churned):")
    print(percent_diff.map(lambda x: f"{x:.2f}%"))
    
    # Statistical significance (t-test)
    print("\nStatistical significance (p-values):")
    p_values = {}
    for feature in num_features:
        churned = data[data['Churn'] == 1][feature].dropna()
        non_churned = data[data['Churn'] == 0][feature].dropna()
        t_stat, p_val = stats.ttest_ind(churned, non_churned, equal_var=False)
        p_values[feature] = p_val
        significance = "Significant" if p_val < 0.05 else "Not significant"
        print(f"{feature}: {p_val:.4f} ({significance})")
    
    # Visualize distributions
    for feature in num_features:
        plt.figure(figsize=(10, 6))
        
        # Histogram with KDE
        plt.subplot(2, 1, 1)
        sns.histplot(data=data, x=feature, hue='Churn', kde=True, common_norm=False)
        plt.title(f'Distribution of {feature} by Churn Status')
        
        # Box plot
        plt.subplot(2, 1, 2)
        sns.boxplot(x='Churn', y=feature, data=data)
        plt.title(f'Box Plot of {feature} by Churn Status')
        
        plt.tight_layout()
        plt.savefig(f'da_outputs/{feature.replace(" ", "_")}_analysis.png')
        plt.close()
    
    # Correlation matrix
    plt.figure(figsize=(10, 8))
    corr_matrix = data[num_features + ['Churn']].corr()
    mask = np.triu(corr_matrix)
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', mask=mask, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('da_outputs/correlation_matrix.png')
    plt.close()
    
    # Feature correlation with churn sorted
    churn_corr = corr_matrix['Churn'].drop('Churn').sort_values(ascending=False)
    print("\nFeature correlation with churn:")
    print(churn_corr)
    
    # Plot correlations with churn
    plt.figure(figsize=(10, 6))
    churn_corr.plot(kind='bar')
    plt.title('Correlation with Churn')
    plt.tight_layout()
    plt.savefig('da_outputs/churn_correlation.png')
    plt.close()
    
    return churn_corr

# Function to analyze categorical features
def analyze_categorical_features(data):
    cat_features = ['Gender', 'Subscription Type', 'Contract Length']
    
    print("\nChurn rate by categorical features:")
    for feature in cat_features:
        # Churn rate by category
        churn_rate = data.groupby(feature)['Churn'].mean().sort_values(ascending=False)
        
        print(f"\n{feature}:")
        for category, rate in churn_rate.items():
            print(f"  {category}: {rate:.2%}")
        
        # Chi-square test for independence
        cross_tab = pd.crosstab(data[feature], data['Churn'])
        chi2, p, dof, expected = stats.chi2_contingency(cross_tab)
        significance = "Significant" if p < 0.05 else "Not significant"
        print(f"  Chi-square p-value: {p:.4f} ({significance})")
        
        # Visualization
        plt.figure(figsize=(10, 6))
        
        # Count plot
        plt.subplot(2, 1, 1)
        sns.countplot(x=feature, hue='Churn', data=data)
        plt.title(f'Count of Customers by {feature} and Churn Status')
        
        # Churn rate plot
        plt.subplot(2, 1, 2)
        churn_rate.plot(kind='bar')
        plt.title(f'Churn Rate by {feature}')
        plt.ylabel('Churn Rate')
        
        plt.tight_layout()
        plt.savefig(f'da_outputs/{feature.replace(" ", "_")}_analysis.png')
        plt.close()
    
    # Combined analysis (Contract Length and Subscription Type)
    combined = data.groupby(['Contract Length', 'Subscription Type'])['Churn'].mean()
    combined_df = combined.reset_index()
    combined_df['Churn'] = combined_df['Churn'] * 100
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Contract Length', y='Churn', hue='Subscription Type', data=combined_df)
    plt.title('Churn Rate by Contract Length and Subscription Type')
    plt.ylabel('Churn Rate (%)')
    plt.tight_layout()
    plt.savefig('da_outputs/combined_categorical_analysis.png')
    plt.close()
    
    print("\nChurn rate by Contract Length and Subscription Type:")
    print(combined)

# 4. Customer Segmentation Analysis
def segment_analysis(data):
    print("\n==== Customer Segmentation Analysis ====")
    
    # Create customer segments
    data = data.copy()
    
    # Age segments
    data['Age_Group'] = pd.cut(data['Age'], bins=[0, 25, 35, 45, 55, 100], 
                              labels=['18-25', '26-35', '36-45', '46-55', '56+'])
    
    # Tenure segments
    data['Tenure_Group'] = pd.cut(data['Tenure'], bins=[0, 12, 24, 36, 48, 100], 
                                 labels=['0-12 months', '1-2 years', '2-3 years', '3-4 years', '4+ years'])
    
    # Usage segments
    data['Usage_Group'] = pd.qcut(data['Usage Frequency'], q=3, 
                                labels=['Low', 'Medium', 'High'])
    
    # Spending segments
    data['Spend_Group'] = pd.qcut(data['Total Spend'], q=4, 
                                labels=['Low', 'Medium', 'High', 'Very High'])
    
    # Analyze churn by segment
    segment_columns = ['Age_Group', 'Tenure_Group', 'Usage_Group', 'Spend_Group']
    
    for segment in segment_columns:
        # Churn rate by segment
        churn_by_segment = data.groupby(segment)['Churn'].mean().sort_values(ascending=False)
        
        print(f"\nChurn rate by {segment}:")
        for category, rate in churn_by_segment.items():
            print(f"  {category}: {rate:.2%}")
        
        # Visualization
        plt.figure(figsize=(10, 6))
        churn_by_segment.plot(kind='bar')
        plt.title(f'Churn Rate by {segment}')
        plt.ylabel('Churn Rate')
        plt.tight_layout()
        plt.savefig(f'da_outputs/{segment}_analysis.png')
        plt.close()
    
    # Multi-dimensional analysis
    # Tenure and Usage
    tenure_usage = data.groupby(['Tenure_Group', 'Usage_Group'])['Churn'].mean().unstack()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(tenure_usage, annot=True, fmt='.2%', cmap='YlOrRd')
    plt.title('Churn Rate by Tenure and Usage')
    plt.tight_layout()
    plt.savefig('da_outputs/tenure_usage_analysis.png')
    plt.close()
    
    # Contract and Spending
    contract_spend = data.groupby(['Contract Length', 'Spend_Group'])['Churn'].mean().unstack()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(contract_spend, annot=True, fmt='.2%', cmap='YlOrRd')
    plt.title('Churn Rate by Contract Length and Spending Level')
    plt.tight_layout()
    plt.savefig('da_outputs/contract_spend_analysis.png')
    plt.close()
    
    return data

# 5. Risk Factor Analysis
def risk_factor_analysis(data):
    print("\n==== Customer Risk Factor Analysis ====")
    
    # Create engineered features for analysis
    data = data.copy()
    
    # Key ratios
    data['Payment_Delay_Ratio'] = data['Payment Delay'] / (data['Tenure'] + 1)
    data['Support_Calls_Per_Tenure'] = data['Support Calls'] / (data['Tenure'] + 1)
    data['Recent_Activity_Ratio'] = data['Last Interaction'] / (data['Tenure'] + 1)
    data['Spend_Per_Tenure'] = data['Total Spend'] / (data['Tenure'] + 1)
    
    # Analyze engineered features
    engineered_features = ['Payment_Delay_Ratio', 'Support_Calls_Per_Tenure', 
                          'Recent_Activity_Ratio', 'Spend_Per_Tenure']
    
    # Mean by churn status
    eng_means = data.groupby('Churn')[engineered_features].mean()
    print("\nMean values of risk factors by churn status:")
    print(eng_means)
    
    # Percent difference
    eng_percent_diff = (eng_means.loc[1] - eng_means.loc[0]) / eng_means.loc[0] * 100
    print("\nPercent difference in risk factors (Churned vs Non-churned):")
    print(eng_percent_diff.map(lambda x: f"{x:.2f}%"))
    
    # Correlation with churn
    eng_corr = data[engineered_features + ['Churn']].corr()['Churn'].drop('Churn')
    print("\nRisk factor correlation with churn:")
    print(eng_corr)
    
    # Visualize engineered features
    for feature in engineered_features:
        plt.figure(figsize=(10, 6))
        
        # Box plot
        plt.subplot(2, 1, 1)
        sns.boxplot(x='Churn', y=feature, data=data)
        plt.title(f'{feature} by Churn Status')
        
        # Distribution plot
        plt.subplot(2, 1, 2)
        sns.histplot(data=data, x=feature, hue='Churn', kde=True)
        plt.title(f'Distribution of {feature} by Churn Status')
        
        plt.tight_layout()
        plt.savefig(f'da_outputs/{feature}_analysis.png')
        plt.close()
    
    # Create a risk score (simple weighted sum of normalized risk factors)
    # First normalize the risk factors
    from sklearn.preprocessing import MinMaxScaler
    
    scaler = MinMaxScaler()
    risk_features = data[engineered_features].copy()
    risk_features_scaled = pd.DataFrame(
        scaler.fit_transform(risk_features),
        columns=engineered_features,
        index=risk_features.index
    )
    
    # Create weighted risk score based on correlation with churn
    weights = eng_corr.abs() / eng_corr.abs().sum()
    data['Risk_Score'] = 0
    
    for feature in engineered_features:
        data['Risk_Score'] += risk_features_scaled[feature] * weights[feature]
    
    # Analyze risk score
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x='Risk_Score', hue='Churn', kde=True)
    plt.title('Distribution of Risk Score by Churn Status')
    plt.tight_layout()
    plt.savefig('da_outputs/risk_score_distribution.png')
    plt.close()
    
    # Risk score thresholds
    risk_thresholds = [0.3, 0.5, 0.7]
    
    print("\nChurn rate by risk score threshold:")
    for threshold in risk_thresholds:
        high_risk = data[data['Risk_Score'] >= threshold]
        churn_rate = high_risk['Churn'].mean()
        customer_count = len(high_risk)
        customer_percent = customer_count / len(data) * 100
        
        print(f"Risk score >= {threshold}: {churn_rate:.2%} churn rate " 
              f"({customer_count} customers, {customer_percent:.1f}% of total)")
    
    return data

# 6. Customer Value Analysis
def value_analysis(data):
    print("\n==== Customer Value Analysis ====")
    
    # Create value segments
    data = data.copy()
    data['Value_Segment'] = pd.qcut(data['Total Spend'], q=4, 
                                   labels=['Low', 'Medium', 'High', 'Premium'])
    
    # Analyze churn by value segment
    value_churn = data.groupby('Value_Segment')['Churn'].mean().sort_values(ascending=False)
    
    print("\nChurn rate by customer value segment:")
    for segment, rate in value_churn.items():
        segment_count = len(data[data['Value_Segment'] == segment])
        print(f"  {segment}: {rate:.2%} ({segment_count} customers)")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    value_churn.plot(kind='bar')
    plt.title('Churn Rate by Customer Value Segment')
    plt.ylabel('Churn Rate')
    plt.tight_layout()
    plt.savefig('da_outputs/value_segment_analysis.png')
    plt.close()
    
    # Value and contract analysis
    value_contract = data.groupby(['Value_Segment', 'Contract Length'])['Churn'].mean().unstack()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(value_contract, annot=True, fmt='.2%', cmap='YlOrRd')
    plt.title('Churn Rate by Value Segment and Contract Length')
    plt.tight_layout()
    plt.savefig('da_outputs/value_contract_analysis.png')
    plt.close()
    
    # Calculate potential revenue impact of churn
    value_impact = data.groupby('Value_Segment').agg({
        'Churn': 'mean',
        'Total Spend': 'sum'
    })
    
    value_impact['Churned_Revenue'] = value_impact['Churn'] * value_impact['Total Spend']
    value_impact['Revenue_Percent'] = value_impact['Churned_Revenue'] / value_impact['Total Spend'].sum() * 100
    
    print("\nRevenue impact of churn by value segment:")
    print(value_impact)
    
    # Visualization of revenue impact
    plt.figure(figsize=(10, 6))
    plt.bar(value_impact.index, value_impact['Churned_Revenue'])
    plt.title('Revenue Lost to Churn by Value Segment')
    plt.ylabel('Revenue Lost')
    plt.tight_layout()
    plt.savefig('da_outputs/revenue_impact_analysis.png')
    plt.close()
    
    return data

# 7. Key Findings and Insights
def generate_insights(data, churn_corr):
    print("\n==== Key Findings and Recommendations ====")
    
    # Top churning segments
    segment_analysis_data = segment_analysis(data)
    
    # Top correlations with churn
    print("\nTop factors correlated with churn:")
    for feature, corr in churn_corr.head(5).items():
        print(f"  {feature}: {corr:.4f}")
    
    # Risk factor analysis
    risk_data = risk_factor_analysis(data)
    
    # Value segment analysis
    value_data = value_analysis(data)
    
    # Generate insights file
    with open('da_outputs/key_insights.txt', 'w') as f:
        f.write("==== CUSTOMER CHURN ANALYSIS: KEY INSIGHTS ====\n\n")
        
        f.write("1. CHURN RISK FACTORS\n")
        f.write("   Top factors correlated with churn:\n")
        for feature, corr in churn_corr.head(5).items():
            f.write(f"   - {feature}: {corr:.4f}\n")
        
        f.write("\n2. HIGH-RISK CUSTOMER SEGMENTS\n")
        
        # Contract analysis
        contract_churn = data.groupby('Contract Length')['Churn'].mean()
        f.write(f"   Contract types by churn rate:\n")
        for contract, rate in contract_churn.sort_values(ascending=False).items():
            f.write(f"   - {contract}: {rate:.2%}\n")
        
        # Tenure analysis
        tenure_groups = segment_analysis_data.groupby('Tenure_Group')['Churn'].mean()
        f.write(f"\n   Tenure groups by churn rate:\n")
        for tenure, rate in tenure_groups.sort_values(ascending=False).items():
            f.write(f"   - {tenure}: {rate:.2%}\n")
        
        # Payment delay analysis
        f.write("\n3. PAYMENT BEHAVIOR INSIGHTS\n")
        payment_stats = data.groupby('Churn')['Payment Delay'].mean()
        f.write(f"   Average payment delays:\n")
        f.write(f"   - Churned customers: {payment_stats[1]:.2f} days\n")
        f.write(f"   - Retained customers: {payment_stats[0]:.2f} days\n")
        f.write(f"   - Difference: {payment_stats[1] - payment_stats[0]:.2f} days\n")
        
        # Support calls analysis
        f.write("\n4. CUSTOMER SERVICE INSIGHTS\n")
        support_stats = data.groupby('Churn')['Support Calls'].mean()
        f.write(f"   Average support calls:\n")
        f.write(f"   - Churned customers: {support_stats[1]:.2f} calls\n")
        f.write(f"   - Retained customers: {support_stats[0]:.2f} calls\n")
        f.write(f"   - Difference: {support_stats[1] - support_stats[0]:.2f} calls\n")
        
        # Value segment insights
        f.write("\n5. CUSTOMER VALUE INSIGHTS\n")
        value_churn = value_data.groupby('Value_Segment')['Churn'].mean()
        f.write(f"   Value segments by churn rate:\n")
        for segment, rate in value_churn.sort_values(ascending=False).items():
            f.write(f"   - {segment}: {rate:.2%}\n")
        
        # Risk segments
        f.write("\n6. RISK ASSESSMENT\n")
        risk_threshold = 0.7
        high_risk = risk_data[risk_data['Risk_Score'] >= risk_threshold]
        high_risk_rate = high_risk['Churn'].mean()
        high_risk_count = len(high_risk)
        high_risk_percent = high_risk_count / len(risk_data) * 100
        
        f.write(f"   High-risk customers (score >= {risk_threshold}):\n")
        f.write(f"   - Count: {high_risk_count} ({high_risk_percent:.1f}% of customer base)\n")
        f.write(f"   - Churn rate: {high_risk_rate:.2%}\n")
        
        # Business recommendations
        f.write("\n==== BUSINESS RECOMMENDATIONS ====\n\n")
        f.write("1. PAYMENT ISSUES\n")
        f.write("   - Implement proactive payment reminders\n")
        f.write("   - Offer flexible payment options for customers with history of delays\n")
        f.write("   - Create early warning system for customers with increasing payment delays\n")
        
        f.write("\n2. CUSTOMER SUPPORT\n")
        f.write("   - Review support processes for customers with multiple support calls\n")
        f.write("   - Implement follow-up satisfaction surveys after support interactions\n")
        f.write("   - Consider dedicated account managers for high-value customers with support issues\n")
        
        f.write("\n3. CONTRACT STRATEGIES\n")
        f.write("   - Review monthly contract offerings and incentives\n")
        f.write("   - Create targeted offers to move monthly customers to longer contracts\n")
        f.write("   - Implement special retention programs for customers in their first year\n")
        
        f.write("\n4. VALUE-BASED RETENTION\n")
        f.write("   - Prioritize retention efforts for high-value segments\n")
        f.write("   - Develop targeted upgrade paths for medium-value customers\n")
        f.write("   - Consider special loyalty programs for premium customers\n")
        
        f.write("\n5. EARLY INTERVENTION\n")
        f.write("   - Implement risk scoring system based on this analysis\n")
        f.write("   - Create proactive outreach for customers with increasing risk scores\n")
        f.write("   - Develop targeted offers based on specific risk factors\n")

# Run the analysis pipeline
print("\n==== Starting Customer Churn Data Analysis ====")

# Run numerical feature analysis on training data
churn_correlations = analyze_numerical_features(train_data)

# Run categorical feature analysis
analyze_categorical_features(train_data)

# Generate comprehensive insights
generate_insights(train_data, churn_correlations)

print("\n==== Data Analysis Complete ====")
print("Results saved to 'da_outputs' directory")