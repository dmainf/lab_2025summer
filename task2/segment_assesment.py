import pandas as pd
import numpy as np
from lib.prepro import clean_data, one_hot_encoding, merge_df
from lib.feature_eng import feature_engineering, feature_group_diff
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import lightgbm as lgb

# Load and prepare data (same as predict.py)
df_raw = pd.read_csv('../WA_Fn-UseC_-Telco-Customer-Churn.csv')
clean_df = shuffle(clean_data(df_raw.copy()), random_state=42).reset_index()
clean_df = feature_engineering(clean_df)
clean_df = feature_group_diff(clean_df)
clean_df = clean_df.drop(columns=['CustomerID'])

feature_df = clean_df.drop(columns=['Churn'])
target_df = clean_df['Churn']

X_train, X_test, y_train, y_test = train_test_split(feature_df, target_df, test_size=0.2, random_state=42, stratify=target_df)

# Train simple LightGBM model
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 24,
    'learning_rate': 0.067,
    'feature_fraction': 0.48,
    'verbose': -1
}

train_data = lgb.Dataset(X_train, label=y_train)
model = lgb.train(params, train_data, num_boost_round=50)
y_prob = model.predict(X_test, num_iteration=model.best_iteration)
y_pred = (y_prob >= 0.5).astype(int)

# Add original columns for segmentation
test_indices = X_test.index
original_test_data = df_raw.iloc[test_indices].copy()

def create_segments(df):
    """Create contract/charges segments"""
    df = df.copy()
    
    # Contract segments
    df['ContractSeg'] = df['Contract'].map({
        'Month-to-month': 'Short',
        'One year': 'Medium', 
        'Two year': 'Long'
    })
    
    # Monthly charges segments (quartiles)
    monthly_q25 = df['MonthlyCharges'].quantile(0.25)
    monthly_q75 = df['MonthlyCharges'].quantile(0.75)
    
    df['ChargesSeg'] = 'Medium'
    df.loc[df['MonthlyCharges'] <= monthly_q25, 'ChargesSeg'] = 'Low'
    df.loc[df['MonthlyCharges'] >= monthly_q75, 'ChargesSeg'] = 'High'
    
    # Combined segments
    df['Combined_Segment'] = df['ContractSeg'] + '_' + df['ChargesSeg']
    
    print(f"Monthly charges quartiles: Q1={monthly_q25:.1f}, Q3={monthly_q75:.1f}")
    
    return df

original_test_data = create_segments(original_test_data)

def detailed_segment_analysis(y_true, y_pred, segments, segment_col):
    """Detailed analysis with clear percentages"""
    results = []
    
    total_test_cases = len(y_true)
    
    for segment in segments[segment_col].unique():
        mask = segments[segment_col] == segment
        if mask.sum() == 0:
            continue
            
        seg_y_true = y_true[mask]
        seg_y_pred = y_pred[mask]
        
        total_cases = len(seg_y_true)
        
        # Basic counts
        fp = ((seg_y_true == 0) & (seg_y_pred == 1)).sum()
        fn = ((seg_y_true == 1) & (seg_y_pred == 0)).sum()
        tp = ((seg_y_true == 1) & (seg_y_pred == 1)).sum()
        tn = ((seg_y_true == 0) & (seg_y_pred == 0)).sum()
        
        # Actual distribution
        actual_churn = (seg_y_true == 1).sum()
        actual_no_churn = (seg_y_true == 0).sum()
        
        # Calculate rates
        total_errors = fp + fn
        misclassification_rate = total_errors / total_cases * 100 if total_cases > 0 else 0
        
        # Specific error rates
        fp_rate_of_nochurn = fp / actual_no_churn * 100 if actual_no_churn > 0 else 0
        fn_rate_of_churn = fn / actual_churn * 100 if actual_churn > 0 else 0
        
        # Population percentages
        segment_pct_of_total = total_cases / total_test_cases * 100
        
        results.append({
            'Segment': segment,
            'Cases': total_cases,
            'Segment_Pct': segment_pct_of_total,
            'Actual_Churn': actual_churn,
            'Actual_NoChurn': actual_no_churn,
            'Churn_Rate': actual_churn / total_cases * 100,
            'FP': fp, 'FN': fn,
            'Total_Errors': total_errors,
            'Misclass_Rate': misclassification_rate,
            'FP_Rate': fp_rate_of_nochurn,
            'FN_Rate': fn_rate_of_churn
        })
    
    return pd.DataFrame(results)

print("=== DETAILED SEGMENT MISCLASSIFICATION ANALYSIS ===")
print(f"Total test cases: {len(y_test)}")
print(f"Overall misclassification rate: {((y_test != y_pred).sum() / len(y_test) * 100):.1f}%")
print(f"Overall false positives: {((y_test == 0) & (y_pred == 1)).sum()}")
print(f"Overall false negatives: {((y_test == 1) & (y_pred == 0)).sum()}")
print()

# Combined segment analysis
print("=== CONTRACT + CHARGES SEGMENTS ===")
combined_results = detailed_segment_analysis(y_test.values, y_pred, original_test_data, 'Combined_Segment')

# Sort by misclassification rate
combined_sorted = combined_results.sort_values('Misclass_Rate', ascending=False)

print("\nSegments ordered by misclassification rate:")
print("-" * 90)
print(f"{'Segment':<15} {'Cases':<6} {'%Pop':<5} {'ChurnRate':<9} {'Errors':<7} {'Miss%':<6} {'FP_Rate%':<8} {'FN_Rate%':<8}")
print("-" * 90)

for _, row in combined_sorted.iterrows():
    print(f"{row['Segment']:<15} {row['Cases']:<6.0f} {row['Segment_Pct']:<5.1f} {row['Churn_Rate']:<9.1f} {row['Total_Errors']:<7.0f} {row['Misclass_Rate']:<6.1f} {row['FP_Rate']:<8.1f} {row['FN_Rate']:<8.1f}")