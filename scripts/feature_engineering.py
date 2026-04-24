import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

def create_derived_features(df):
    """Create new derived features from existing ones."""
    df = df.copy()
    
    # 1. FamilySize
    if 'SibSp' in df.columns and 'Parch' in df.columns:
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        # 2. IsAlone
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        
        # 6. Fare per person
        if 'Fare' in df.columns:
            df['FarePerPerson'] = df['Fare'] / df['FamilySize']
            
    # 3. Title extraction from Name
    if 'Name' in df.columns:
        df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        # Consolidate rare titles
        rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 
                       'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
        df['Title'] = df['Title'].replace(rare_titles, 'Rare')
        df['Title'] = df['Title'].replace('Mlle', 'Miss')
        df['Title'] = df['Title'].replace('Ms', 'Miss')
        df['Title'] = df['Title'].replace('Mme', 'Mrs')
        df = df.drop('Name', axis=1)
        
    # 4. Deck extraction from Cabin
    if 'Cabin' in df.columns:
        df['Deck'] = df['Cabin'].str[0]
        # Replace 'U' (from our 'Unknown' imputation) with 'U'
        df = df.drop('Cabin', axis=1)
        
    # 5. Age groups
    if 'Age' in df.columns:
        bins = [0, 12, 18, 60, 120]
        labels = ['Child', 'Teen', 'Adult', 'Senior']
        df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
        
    return df

def apply_categorical_encoding(df):
    """Encode categorical features."""
    df = df.copy()
    
    # One-hot encoding for nominal features
    nominal_cols = ['Sex', 'Embarked', 'Title', 'Deck', 'AgeGroup']
    cols_to_encode = [col for col in nominal_cols if col in df.columns]
    
    df = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)
    
    # Ordinal encoding for Pclass (already numeric, but can leave as is or transform if needed)
    # Pclass is inherently ordinal (1st, 2nd, 3rd) so its numeric representation is fine.
    
    return df

def create_interaction_features(df):
    """Create interaction features."""
    df = df.copy()
    
    if 'Pclass' in df.columns and 'Fare' in df.columns:
        df['Pclass_Fare_Interaction'] = df['Pclass'] * df['Fare']
        
    if 'Age' in df.columns and 'Pclass' in df.columns:
        df['Age_Pclass_Interaction'] = df['Age'] * df['Pclass']
        
    if 'Sex_male' in df.columns and 'Pclass' in df.columns:
        df['Sex_Pclass_Interaction'] = df['Sex_male'] * df['Pclass']
        
    # Age x Title interactions
    title_cols = [c for c in df.columns if c.startswith('Title_')]
    if 'Age' in df.columns:
        for t_col in title_cols:
            df[f'Age_{t_col}_Interaction'] = df['Age'] * df[t_col]
        
    return df

def apply_feature_transformations(df):
    """Log transform and scale features."""
    df = df.copy()
    
    # Log transform
    for col in ['Fare', 'Age', 'FarePerPerson']:
        if col in df.columns:
            # Add small constant to avoid log(0)
            df[col + '_Log'] = np.log1p(df[col])
            # We will scale the log versions and drop the original below or scale original based on preference.
            # Usually, we replace the original with log, but let's keep log version.
            
    # Select numerical columns for scaling
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    # Exclude target 'Survived' and ID 'PassengerId'
    cols_to_exclude = ['PassengerId', 'Survived']
    cols_to_scale = [col for col in numerical_cols if col not in cols_to_exclude]
    
    scaler = StandardScaler()
    if len(cols_to_scale) > 0:
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        
    return df

def engineer_features(df):
    """Run all feature engineering steps."""
    df = create_derived_features(df)
    df = apply_categorical_encoding(df)
    df = create_interaction_features(df)
    df = apply_feature_transformations(df)
    
    # Drop irrelevant columns if they exist
    if 'Ticket' in df.columns:
        df = df.drop('Ticket', axis=1)
        
    return df

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, 'data', 'train_cleaned.csv')
    output_path = os.path.join(base_dir, 'data', 'train_engineered.csv')
    
    if os.path.exists(input_path):
        print(f"Loading cleaned data from {input_path}...")
        df = pd.read_csv(input_path)
        
        print("Engineering features...")
        df_engineered = engineer_features(df)
        
        print(f"Saving engineered data to {output_path}...")
        df_engineered.to_csv(output_path, index=False)
        print("Feature engineering completed successfully.")
    else:
        print(f"Error: Could not find input file at {input_path}. Please run data_cleaning.py first.")
