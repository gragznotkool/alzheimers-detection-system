# clustering.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 1. Load processed data
df = pd.read_csv('data/processed/alzheimers_cleaned.csv')

# 2. Select relevant numeric features for clustering
features = ['Age', 'BMI', 'SleepQuality', 'PhysicalActivity',
            'DietQuality', 'CholesterolTotal', 'MMSE']
X = df[features]

# 3. Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. --- K-MEANS ---
kmeans = KMeans(n_clusters=3, random_state=42)
labels_km = kmeans.fit_predict(X_scaled)
df['Cluster_KMeans'] = labels_km
sil_km = silhouette_score(X_scaled, labels_km)
print(f"K-Means Silhouette Score: {sil_km:.3f}")

# Visualization
plt.figure(figsize=(6,5))
sns.scatterplot(x='Age', y='MMSE', hue='Cluster_KMeans', data=df, palette='Set2')
plt.title("K-Means Clustering: Age vs MMSE")
plt.show()


# 5. --- AGGLOMERATIVE CLUSTERING ---
agg = AgglomerativeClustering(n_clusters=3)
labels_ag = agg.fit_predict(X_scaled)
df['Cluster_Agglo'] = labels_ag
sil_ag = silhouette_score(X_scaled, labels_ag)
print(f"Agglomerative Clustering Silhouette Score: {sil_ag:.3f}")

plt.figure(figsize=(6,5))
sns.scatterplot(x='Age', y='BMI', hue='Cluster_Agglo', data=df, palette='pastel')
plt.title("Agglomerative Clustering: Age vs BMI")
plt.show()


# 6. --- DBSCAN ---
db = DBSCAN(eps=1.2, min_samples=5)
labels_db = db.fit_predict(X_scaled)
df['Cluster_DBSCAN'] = labels_db

# Filter out noise (-1)
valid = df[df['Cluster_DBSCAN'] != -1]
if len(valid) > 0:
    sil_db = silhouette_score(X_scaled[labels_db != -1], labels_db[labels_db != -1])
    print(f"DBSCAN Silhouette Score: {sil_db:.3f}")
else:
    print("DBSCAN: Too much noise, adjust eps/min_samples.")

plt.figure(figsize=(6,5))
sns.scatterplot(x='Age', y='CholesterolTotal', hue='Cluster_DBSCAN', data=df, palette='cool')
plt.title("DBSCAN Clustering: Age vs Cholesterol")
plt.show()


# 7. Save results
df.to_csv('data/processed/alzheimers_clustered.csv', index=False)
print("\n✅ Clustering completed. Saved → data/processed/alzheimers_clustered.csv")
