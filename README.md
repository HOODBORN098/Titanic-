# Titanic Survival Analysis

This project aims to build a predictive model for Titanic survival by performing data cleaning, feature engineering, and feature selection. The final deliverables include clean modular Python scripts and an interactive Jupyter Notebook detailing the analysis, visualizations, and automated feature selection to pick the strongest predictors for the target variable 'Survived'.

## Approach
1. **Data Cleaning**: Identify and impute missing values (utilizing Title and Pclass for accurate Age imputation). Detect outliers using IQR and handle extreme Fare cases using 99th percentile capping.
2. **Feature Engineering**: Create robust derived features (`FamilySize`, `IsAlone`, `Title`, `Deck`, `AgeGroup`), perform categorical encoding, and generate interaction factors. Fix skewness using log transforms and apply standard scaling for numerical columns.
3. **Feature Selection**: Drop highly correlated items to prevent multicollinearity, extract feature importances using a Random Forest Classifier, and perform Recursive Feature Elimination with Cross-Validation (RFECV) to automatically isolate the optimal number of final features.

## Features Engineered
- **FamilySize**: `SibSp + Parch + 1` (Total family members onboard).
- **IsAlone**: Binary indicator for solo passengers (1 if FamilySize == 1 else 0).
- **Title**: Extracted from Name (Mr, Mrs, Miss, Master, plus grouped 'Rare' titles) to capture social status.
- **Deck**: Extracted first letter from Cabin ('U' for Unknown) indicating location on ship.
- **AgeGroup**: Categorical binning of Age (Child, Teen, Adult, Senior) for survival patterns.
- **FarePerPerson**: `Fare / FamilySize` to calculate true per-person ticket cost.
- **Interactions**: `Pclass × Fare`, `Age × Pclass`, and `Sex × Pclass` (capturing gender-class survival biases).

## Data Cleaning Decisions
- **Age**: Imputed using the median grouped by `Pclass` and `Title` because passenger status heavily correlates with age. Leftovers were filled with the overall median.
- **Cabin**: Missing values were filled with 'Unknown' instead of dropping the column entirely. This allowed us to extract the 'Deck' level during feature engineering.
- **Embarked**: Imputed with the mode ('S').
- **Fare**: Missing values were filled with median. Outliers were capped at the 99th percentile to prevent extreme values from distorting distance-based algorithms.
- **Age Outliers**: Capped at 1.5*IQR bounds.
- **Consistency**: The `Sex` column was standardized to strict lowercase to avoid duplicate categories, and `Pclass` was enforced as an integer.

## Key Findings & Feature Selection Justification
- **Features Kept**: Encoded `Sex` (e.g. `Sex_male`), `Pclass`, `Fare`, `Age`, and `Title` groups. *Justification*: Random Forest deemed these the most important variables, with Gender and Pclass heavily influencing survival. RFE selected them as the top predictors.
- **Features Dropped**: `SibSp` and `Parch` (highly correlated with `FamilySize`), `Ticket` (too many unique values, not predictive), and `PassengerId`. *Justification*: Dropped due to high redundancy (correlation > 0.8) or low predictive power during Recursive Feature Elimination (RFE).
- **Interaction effects**: `Sex × Pclass` and `Age × Title` interactions provided deeper insight into specific demographic groups (e.g., 1st Class females had a significantly higher survival rate than 3rd class males).

## How to Run
First, install the required dependencies:
```bash
pip install -r requirements.txt
```

To run the full pipeline sequentially, execute the scripts from the project root:
```bash
python scripts/data_cleaning.py
python scripts/feature_engineering.py
python scripts/feature_selection.py
```

To explore the visualizations and interactive analysis:
```bash
jupyter notebook notebooks/Titanic_Feature_Engineering.ipynb
```

## Results
The structured feature engineering and subsequent automated selection using RFECV yielded a robust feature set ready for downstream predictive modeling, demonstrating that a curated subset of features provides better validation scores than raw attributes.
