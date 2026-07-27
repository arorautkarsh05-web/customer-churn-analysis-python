import pandas as pd

df = pd.read_csv("telco.csv")

print(df.head())
# Shape of Dataset
print("Shape:", df.shape)

# Column Names
print("\nColumns:")
print(df.columns.tolist())

# Data Types
print("\nData Types:")
print(df.dtypes)

# Basic Information
print("\nInfo:")
df.info()

# Statistical Summary
print("\nSummary Statistics:")
print(df.describe())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nDuplicate Rows:", df.duplicated().sum())
print(df["Churn Label"].value_counts())

print(df["Churn Label"].value_counts(normalize=True) * 100)
num_cols = df.select_dtypes(include=['int64','float64']).columns

cat_cols = df.select_dtypes(include=['object']).columns

print("Numerical Columns:", len(num_cols))
print(num_cols)

print("\nCategorical Columns:", len(cat_cols))
print(cat_cols)
# Copy Dataset

df_clean = df.copy()

# Replace Missing Values

df_clean["Offer"] = df_clean["Offer"].fillna("No Offer")

df_clean["Internet Type"] = df_clean["Internet Type"].fillna("No Internet")

df_clean["Churn Category"] = df_clean["Churn Category"].fillna("Not Churned")

df_clean["Churn Reason"] = df_clean["Churn Reason"].fillna("Not Applicable")

print(df_clean.isnull().sum())
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6,5))

sns.countplot(
    data=df_clean,
    x="Churn Label"
)

plt.title("Customer Churn Distribution")

plt.xlabel("Churn")

plt.ylabel("Customers")

plt.show()
plt.figure(figsize=(7,5))

sns.countplot(
    data=df_clean,
    x="Gender",
    hue="Churn Label"
)

plt.title("Gender vs Churn")

plt.show()
plt.figure(figsize=(8,5))

sns.countplot(
    data=df_clean,
    x="Contract",
    hue="Churn Label"
)

plt.xticks(rotation=20)

plt.title("Contract Type vs Churn")

plt.show()
plt.figure(figsize=(8,5))

sns.countplot(
    data=df_clean,
    x="Internet Service",
    hue="Churn Label"
)

plt.title("Internet Service vs Churn")

plt.show()
plt.figure(figsize=(12,5))

sns.countplot(
    data=df_clean,
    x="Payment Method",
    hue="Churn Label"
)

plt.xticks(rotation=25)

plt.title("Payment Method vs Churn")

plt.show()
plt.figure(figsize=(8,5))

sns.boxplot(
    data=df_clean,
    x="Churn Label",
    y="Tenure in Months"
)

plt.title("Tenure vs Churn")

plt.show()
plt.figure(figsize=(8,5))

sns.boxplot(
    data=df_clean,
    x="Churn Label",
    y="Monthly Charge"
)

plt.title("Monthly Charges vs Churn")

plt.show()
plt.figure(figsize=(8,5))

sns.boxplot(
    data=df_clean,
    x="Churn Label",
    y="Total Revenue"
)

plt.title("Total Revenue vs Churn")

plt.show()
plt.figure(figsize=(8,5))

sns.boxplot(
    data=df_clean,
    x="Churn Label",
    y="Satisfaction Score"
)

plt.title("Satisfaction Score vs Churn")

plt.show()
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12,8))

corr = df_clean.select_dtypes(include=["int64","float64"]).corr()

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.show()