import pandas as pd
from scikitlearn import standardscaler
import datetime as dt

interim_path = "/Users/mamduhhalawa/Desktop/Mlrepos/kaggle-project-3/data/interim/synthetic_customers.csv"

df = pd.read_csv(interim_path)

date_columns = ["last_purchase_date", "first_purchase_date"]

df["first_purchase_date"] = pd.to_datetime(df["first_purchase_date"])
df["last_purchase_date"] = pd.to_datetime(df["last_purchase_date"])

df["purchase_time"]
