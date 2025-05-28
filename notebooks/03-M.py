import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns

X = pd.read_csv(
    "/Users/mamduhhalawa/Desktop/Mlrepos/kaggle-project-3/data/processed/df_scaled.csv"
)

# Choosing clusters

inertias = []
K_range = range(1, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X)
    inertias.append(km.inertia_)

plt.plot(K_range, inertias, "bo-")
plt.xlabel("Number of clusters K")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")
plt.show()

# Run KMeans on K=5 clusters
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X)

# Add cluster labels to your DataFrame
X["cluster"] = clusters

cluster_summary = X.groupby("cluster").mean()
print(cluster_summary)
print(X["cluster"].value_counts().sort_index())

plt.figure(figsize=(12, 6))
sns.heatmap(cluster_summary, annot=True, cmap="coolwarm")
plt.title("Cluster Feature Means")
plt.show()

# rescale redo EDA - overload of PCA features