import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt


interim_path = "/Users/mamduhhalawa/Desktop/Mlrepos/kaggle-project-3/data/interim/synthetic_customers.csv"

df = pd.read_csv(interim_path)

date_columns = ["last_purchase_date", "first_purchase_date"]

df["first_purchase_date"] = pd.to_datetime(df["first_purchase_date"])
df["last_purchase_date"] = pd.to_datetime(df["last_purchase_date"])

df["spent_per_month"] = (
    df["total_spent"] / (df["last_purchase_date"] - df["first_purchase_date"]).dt.days
)
df.drop(columns=["first_purchase_date", "last_purchase_date"], inplace=True)

# Assume `df` is your original DataFrame
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Reconstruct DataFrame with original column names
df_scaled = pd.DataFrame(X_scaled, columns=df.columns, index=df.index)

corr_matrix = df_scaled.corr()

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=1,
    square=True,
    linewidths=0,
)
plt.title("Feature Correlation Matrix")
plt.show()

df_scaled.to_csv(
    "/Users/mamduhhalawa/Desktop/Mlrepos/kaggle-project-3/data/processed/df_scaled.csv"
)
