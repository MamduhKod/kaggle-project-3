import pandas as pd

# Decided to remove high-dimension categoricals

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

# Doesnt differ a lot by order value - might remove to improve analysis
df.groupby("favorite_category")["avg_order_value"].mean()


categorical_cols = ["city", "country", "favorite_category"]
df = df.drop(columns=categorical_cols)


# No missing values
df.isnull().sum()

df.to_csv(
    "/Users/mamduhhalawa/Desktop/Mlrepos/kaggle-project-3/data/interim/synthetic_customers.csv",
    index=False,
)
