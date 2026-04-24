import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

def remove_highly_correlated_features(df, threshold=0.8):
    """Remove features that are highly correlated with others."""
    # Ensure only numeric columns are used for correlation
    df_numeric = df.select_dtypes(include=[np.number])
    corr_matrix = df_numeric.corr().abs()
    
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    print(f"Highly correlated features to drop (>{threshold}): {to_drop}")
    df_reduced = df.drop(columns=to_drop)
    return df_reduced, to_drop

def get_feature_importances(X, y):
    """Rank features using Random Forest."""
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    importances = rf.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    return feature_importance_df

def select_features(X, y, n_features=10):
    """Use Recursive Feature Elimination to select top features."""
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rfe = RFE(estimator=rf, n_features_to_select=n_features, step=1)
    rfe.fit(X, y)
    
    selected_features = X.columns[rfe.support_].tolist()
    dropped_features = X.columns[~rfe.support_].tolist()
    return selected_features, dropped_features

def perform_feature_selection(df, target_col='Survived'):
    """Run all feature selection steps."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in the dataset.")
        
    # Drop irrelevant identifier
    if 'PassengerId' in df.columns:
        df = df.drop('PassengerId', axis=1)
        
    # Step 1: Correlation Analysis
    df_reduced, dropped_corr = remove_highly_correlated_features(df)
    
    # Prepare X and y
    y = df_reduced[target_col]
    X = df_reduced.drop(target_col, axis=1)
    
    # Step 2: Feature Importance via Random Forest
    importance_df = get_feature_importances(X, y)
    print("\nTop 10 Feature Importances:")
    print(importance_df.head(10))
    
    # Step 3: Recursive Feature Elimination (RFE)
    n_features = min(12, X.shape[1])
    rfe_selected, rfe_dropped = select_features(X, y, n_features=n_features)
    
    print("\n--- Justification for Feature Selection ---")
    print("\nFeatures Dropped during Correlation Analysis (>0.8 threshold):")
    for feat in dropped_corr:
        print(f" - {feat}: Dropped due to high redundancy/correlation with other features.")
        
    print(f"\nFeatures Dropped by RFE (Low Importance):")
    for feat in rfe_dropped:
        print(f" - {feat}: Dropped because RFE deemed it less predictive compared to the top {n_features} features.")
        
    print(f"\nFeatures Kept by RFE (Top Predictors):")
    for feat in rfe_selected:
        print(f" - {feat}: Kept because Random Forest importance and RFE selected it as a strong predictor.")
    
    return rfe_selected, importance_df

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, 'data', 'train_engineered.csv')
    
    if os.path.exists(input_path):
        print(f"Loading engineered data from {input_path}...")
        df = pd.read_csv(input_path)
        
        print("Performing feature selection...")
        selected_features, importance_df = perform_feature_selection(df)
        
        print("\nFinal Selected Features for Modeling:")
        for feature in selected_features:
            print(f"- {feature}")
            
        print("\nFeature selection completed successfully.")
    else:
        print(f"Error: Could not find input file at {input_path}. Please run feature_engineering.py first.")
