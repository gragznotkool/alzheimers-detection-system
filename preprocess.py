# src/preprocess.py
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

# 1. Load data
df = pd.read_csv('alzheimers_disease_data.csv')

# 2. Inspect
print(df.info())
print(df.isnull().sum())
print(df.describe())

# 3. Handle missing values
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

# Numeric: fill with median
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Categorical: fill with mode
for c in cat_cols:
    df[c].fillna(df[c].mode()[0], inplace=True)

# 4. Encode categorical features
encoder_dict = {}
for c in cat_cols:
    le = LabelEncoder()
    df[c] = le.fit_transform(df[c])
    encoder_dict[c] = le

# 5. Feature creation (if needed)
# Example: Binning age
df['AgeGroup'] = pd.qcut(df['Age'], q=4, labels=['Young', 'MidAge', 'Old', 'Elder'])
age_encoder = LabelEncoder()
df['AgeGroup'] = age_encoder.fit_transform(df['AgeGroup'])
encoder_dict['AgeGroup'] = age_encoder

# 6. Scale numeric columns
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# 7. Save processed data
os.makedirs('data/processed', exist_ok=True)
os.makedirs('artifacts', exist_ok=True)

df.to_csv('data/processed/alzheimers_cleaned.csv', index=False)

# Save encoders and scaler
with open('artifacts/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('artifacts/encoder.pkl', 'wb') as f:
    pickle.dump(encoder_dict, f)

print("✅ Preprocessing complete.")
print("Saved cleaned dataset → data/processed/alzheimers_cleaned.csv")
