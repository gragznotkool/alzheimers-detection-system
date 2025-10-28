# apriori_rules.py (fixed)
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# 1. Load preprocessed data
df = pd.read_csv('data/processed/alzheimers_cleaned.csv')

# 2. Select key columns (lifestyle + diagnosis)
cols = ['Age', 'BMI', 'Smoking', 'AlcoholConsumption', 'PhysicalActivity',
        'DietQuality', 'SleepQuality', 'FamilyHistoryAlzheimers', 'Diagnosis']
data = df[cols].copy()

# 3. Discretize continuous variables into categorical bins
data['AgeGroup'] = pd.cut(data['Age'], bins=[40,60,80,100], labels=['40-60','60-80','80+'])
data['BMIGroup'] = pd.cut(data['BMI'], bins=[0,18.5,25,30,50],
                          labels=['Underweight','Normal','Overweight','Obese'])

# Drop continuous versions
data.drop(['Age','BMI'], axis=1, inplace=True)

# 4. Convert numeric 0/1 columns into Yes/No for clarity
data = data.replace({0: 'No', 1: 'Yes'})

# 5. One-hot encode (this makes all values 0/1)
encoded_df = pd.get_dummies(data)

# Convert to bool explicitly (important for mlxtend)
encoded_df = encoded_df.astype(bool)

# 6. Apply Apriori
frequent_items = apriori(encoded_df, min_support=0.1, use_colnames=True)
print("Top Frequent Itemsets:")
print(frequent_items.sort_values(by='support', ascending=False).head())

# 7. Generate association rules
rules = association_rules(frequent_items, metric='confidence', min_threshold=0.6)
rules = rules.sort_values(by='lift', ascending=False)
print("\nTop 10 Association Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

# 8. Save rules
rules.to_csv('data/processed/alzheimers_association_rules.csv', index=False)
print("\n✅ Association Rule Mining completed successfully!")
