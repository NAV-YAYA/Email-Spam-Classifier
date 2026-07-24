#importing dataset
import pandas as pd

# Dataset URL
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"

# Load dataset
df = pd.read_csv(
    url,
    sep="\t",
    names=["label", "message"]
)

print("Dataset Loaded Successfully!\n")

print(df.head())

print("\nDataset Shape:", df.shape)

print("\nClass Distribution:")
print(df["label"].value_counts())

#exploring dataset

import pandas as pd

# Dataset URL
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"

# Load dataset
df = pd.read_csv(
    url,
    sep="\t",
    names=["label", "message"]
)

print("===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== DATASET SHAPE =====")
print(df.shape)

print("\n===== CLASS DISTRIBUTION =====")
print(df["label"].value_counts())

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== DUPLICATE ROWS =====")
print(df.duplicated().sum())

#data cleaning

# Remove duplicate rows
df = df.drop_duplicates()

print("\n===== AFTER REMOVING DUPLICATES =====")
print("Dataset Shape:", df.shape)

print("\nClass Distribution:")
print(df["label"].value_counts())