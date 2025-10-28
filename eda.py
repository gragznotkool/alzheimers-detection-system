# eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Load processed data
df = pd.read_csv('data/processed/alzheimers_cleaned.csv')

# 2. Basic Info
print("🔹 Shape of dataset:", df.shape)
print("🔹 Columns:", df.columns.tolist())
print("\n--- Basic Info ---")
print(df.info())
print("\n--- Summary Stats ---")
print(df.describe())

# 3. Check duplicates & missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())
print("\n--- Duplicate Rows ---")
print(df.duplicated().sum())

# 4. Distribution of Target Variable
if 'Diagnosis' in df.columns:
    plt.figure(figsize=(6,4))
    sns.countplot(x='Diagnosis', data=df, palette='Set2')
    plt.title("Distribution of Alzheimer’s Diagnosis")
    plt.xlabel("Diagnosis Category")
    plt.ylabel("Count")
    plt.show()

# 5. Age Distribution
if 'Age' in df.columns:
    plt.figure(figsize=(6,4))
    sns.histplot(df['Age'], kde=True, color='purple')
    plt.title("Age Distribution of Patients")
    plt.show()

# 6. Gender vs Diagnosis
if {'Gender', 'Diagnosis'}.issubset(df.columns):
    plt.figure(figsize=(6,4))
    sns.countplot(x='Gender', hue='Diagnosis', data=df, palette='pastel')
    plt.title("Gender vs Diagnosis Distribution")
    plt.show()

# 7. Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), cmap='coolwarm', annot=False)
plt.title("Correlation Heatmap")
plt.show()

# 8. Pairplot for top correlated features
top_corr = df.corr()['Diagnosis'].abs().sort_values(ascending=False)[1:5].index.tolist()
sns.pairplot(df, vars=top_corr, hue='Diagnosis', palette='husl')
plt.suptitle("Pairplot of Top Correlated Features", y=1.02)
plt.show()

# 9. Boxplots for numeric features
num_cols = df.select_dtypes(include=['int64','float64']).columns
for col in num_cols[:6]:  # limit to first few for readability
    plt.figure(figsize=(6,4))
    sns.boxplot(x='Diagnosis', y=col, data=df, palette='cool')
    plt.title(f"{col} vs Diagnosis")
    plt.show()

# 10. Correlation Summary
corr_target = df.corr()['Diagnosis'].sort_values(ascending=False)
print("\n--- Correlation with Diagnosis ---")
print(corr_target)

print("\n✅ EDA completed successfully.")
