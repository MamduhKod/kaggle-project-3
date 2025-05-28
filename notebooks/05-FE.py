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

# Low intercorrelation

corr_matrix = df.corr()
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

bool_cols = df.select_dtypes(include="bool").columns
non_bool_cols = df.columns.difference(bool_cols)

# Assume `df` is your original DataFrame
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[non_bool_cols])

# Reconstruct DataFrame with original column names
df_scaled_non_bool = pd.DataFrame(X_scaled, columns=non_bool_cols, index=df.index)
df_processed = pd.concat([df_scaled_non_bool, df[bool_cols]], axis=1)

df_processed.to_csv(
    "/Users/mamduhhalawa/Desktop/Mlrepos/kaggle-project-3/data/processed/df_scaled.csv"
)
