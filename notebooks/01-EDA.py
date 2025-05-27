import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import TruncatedSVD


df = pd.read_csv(
    "/Users/mamduhhalawa/Desktop/Mlrepos/kaggle-project-3/data/raw/synthetic_customers.csv"
)

for index, column in enumerate(df.columns):
    print(f"{index}: {column}")


# drop meaningsless values
columns_to_drop = ["customer_id", "name", "email"]
df = df.drop(columns=columns_to_drop)

# see spread of object types

objects = []
for column in df.columns:
    if df[column].dtype == "object":
        print(column, "object")
        objects.append(column)

for column in objects:
    object_column = df[column]
    print(object_column.nunique(), column)

df = pd.get_dummies(df, columns=["gender", "device_type"], drop_first=True)

categorical_cols = ["city", "country", "favorite_category"]
ohe = OneHotEncoder(handle_unknown="ignore")
encoded_matrix = ohe.fit_transform(df[categorical_cols])

svd = TruncatedSVD(n_components=5, random_state=42)
svd_components = svd.fit_transform(encoded_matrix)

# Step 3: Add SVD components to the DataFrame
for i in range(svd_components.shape[1]):
    df[f"cat_embed_{i}"] = svd_components[:, i]

# Optional: Drop the original categorical columns
df.drop(columns=categorical_cols, inplace=True)

# No missing values
df.isnull().sum()

df.to_csv(
    "/Users/mamduhhalawa/Desktop/Mlrepos/kaggle-project-3/data/interim/synthetic_customers.csv",
    index=False,
)
