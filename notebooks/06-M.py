import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gower as gw
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import numpy as np

X = pd.read_csv(
    "/Users/mamduhhalawa/Desktop/Mlrepos/kaggle-project-3/data/processed/df_scaled.csv"
)

X = X.drop(columns="Unnamed: 0")

# To speed up gower
X_sample = X.sample(n=5000, random_state=42)

threshold = np.percentile(
    X_sample["spent_per_month"], 99
)  # or 95, depending on strictness
df_no_outliers = X_sample[X_sample["spent_per_month"] <= threshold].copy()

df_no_outliers.describe()

n_clusters = range(1, 5)

gower_dist = gw.gower_matrix(df_no_outliers)

scores = []

for n in range(2, 6):  # Avoid n=1 (no silhouette for a single cluster)
    clustering = AgglomerativeClustering(
        n_clusters=n, metric="precomputed", linkage="average"
    )
    labels = clustering.fit_predict(gower_dist)

    # Compute silhouette score
    score = silhouette_score(gower_dist, labels, metric="precomputed")
    scores.append((n, score))
    print(f"n_clusters={n}, silhouette={score:.3f}")

# Choose 2 clusters based on highest score
best_n = max(scores, key=lambda x: x[1])[0]
print(f"Best n_clusters: {best_n}")

clustering = AgglomerativeClustering(
    n_clusters=2, metric="precomputed", linkage="average"
)
labels = clustering.fit_predict(gower_dist)

# Add cluster labels to your DataFrame
df_no_outliers["cluster"] = labels

# Now you can inspect clusters
print(df_no_outliers.groupby("cluster").mean())  # summary stats by cluster
print(df_no_outliers["cluster"].value_counts())  # cluster sizes

outlier = X_sample[X_sample["cluster"] == 1]
print(outlier)
