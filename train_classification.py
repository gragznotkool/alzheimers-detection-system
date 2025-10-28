# train_classification.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load preprocessed data
df = pd.read_csv('data/processed/alzheimers_cleaned.csv')

# 2. Separate features and target
X = df.drop(columns=['Diagnosis'])

# ✅ Force to integer and categorical type
y = df['Diagnosis'].astype(int)
print("Unique labels in target:", y.unique())
print("Datatype of target:", y.dtype)
# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 4. --- Decision Tree Classifier ---
dt = DecisionTreeClassifier(max_depth=5, criterion='entropy', random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

print("\n🌳 Decision Tree Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

# Confusion Matrix - Decision Tree
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_pred_dt), annot=True, fmt='d', cmap='Blues')
plt.title("Decision Tree - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Visualize tree
plt.figure(figsize=(15,8))
plot_tree(dt, filled=True, feature_names=X.columns, class_names=[str(c) for c in y.unique()])
plt.title("Decision Tree Visualization")
plt.show()


# 5. --- Naïve Bayes Classifier ---
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

print("\n🤖 Naïve Bayes Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))

# Confusion Matrix - Naive Bayes
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_pred_nb), annot=True, fmt='d', cmap='Greens')
plt.title("Naive Bayes - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("\n✅ Classification complete. Models trained and evaluated.")