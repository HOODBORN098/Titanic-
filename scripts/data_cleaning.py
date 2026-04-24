import pandas as pd
import numpy as np
import os

def load_data(filepath):
    """Load the dataset."""
    return pd.read_csv(filepath)

def handle_missing_values(df):
    """Handle missing values in the dataset."""
    df = df.copy()
    
    # 1. Age: Impute with median (by Pclass and Title groups)
    if 'Name' in df.columns and 'Age' in df.columns:
        temp_title = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        df['Age'] = df.groupby(['Pclass', temp_title])['Age'].transform(lambda x: x.fillna(x.median()))
        # Fill any remaining NaNs with overall median (if a group was entirely NaN)
        df['Age'] = df['Age'].fillna(df['Age'].median())
    elif 'Age' in df.columns:
        df['Age'] = df['Age'].fillna(df['Age'].median())
    
    # 2. Cabin: Create indicator and fill 'Unknown' for Deck extraction
    if 'Cabin' in df.columns:
        df['Cabin_Missing'] = df['Cabin'].isnull().astype(int)
        df['Cabin'] = df['Cabin'].fillna('Unknown')
    
    # 3. Embarked: Impute with mode
    if 'Embarked' in df.columns:
        mode_embarked = df['Embarked'].mode()[0]
        df['Embarked'] = df['Embarked'].fillna(mode_embarked)
        
    # 4. Fare: Impute with median
    if 'Fare' in df.columns:
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    
    return df

def handle_outliers(df):
    """Handle outliers in Age and Fare."""
    df = df.copy()
    
    # Age outliers: Detect using IQR method, cap at 1.5*IQR
    if 'Age' in df.columns:
        Q1 = df['Age'].quantile(0.25)
        Q3 = df['Age'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df['Age'] = np.where(df['Age'] < lower_bound, lower_bound, df['Age'])
        df['Age'] = np.where(df['Age'] > upper_bound, upper_bound, df['Age'])
            
    # Fare outliers: cap extreme values at 99th percentile
    if 'Fare' in df.columns:
        p99 = df['Fare'].quantile(0.99)
        df['Fare'] = np.where(df['Fare'] > p99, p99, df['Fare'])
        
    return df

def ensure_data_consistency(df):
    """Fix inconsistencies and remove duplicates."""
    df = df.copy()
    
    # Ensure Sex values are strictly lowercase 'male'/'female'
    if 'Sex' in df.columns:
        df['Sex'] = df['Sex'].str.lower().str.strip()
        
    # Fix Pclass: ensure integer
    if 'Pclass' in df.columns:
        df['Pclass'] = df['Pclass'].astype(int)
        
    # Remove duplicates
    df = df.drop_duplicates()
    
    return df

def clean_data(input_path, output_path):
    """Run all cleaning steps."""
    df = load_data(input_path)
    df = handle_missing_values(df)
    df = handle_outliers(df)
    df = ensure_data_consistency(df)
    if output_path:
        df.to_csv(output_path, index=False)
    return df

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, 'data', 'train.csv')
    output_path = os.path.join(base_dir, 'data', 'train_cleaned.csv')
    
    if os.path.exists(input_path):
        print("Cleaning data...")
        df_cleaned = clean_data(input_path, output_path)
        print("Data cleaning completed successfully.")
    else:
        print(f"Error: Could not find input file at {input_path}")
