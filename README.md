
"""
Task-02: Data Cleaning & Exploratory Data Analysis (EDA)
Goal: Clean a Titanic-style dataset and explore relationships
between variables to find patterns and trends.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- 1. Load data ----------
csv_path = "titanic.csv"

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
else:
    print(f"'{csv_path}' not found — generating a synthetic Titanic-like sample.")
    rng = np.random.default_rng(1)
    n = 400

    df = pd.DataFrame({
        "PassengerId": range(1, n + 1),
        "Survived": rng.choice([0, 1], size=n, p=[0.62, 0.38]),
        "Pclass": rng.choice([1, 2, 3], size=n, p=[0.24, 0.21, 0.55]),
        "Sex": rng.choice(["male", "female"], size=n, p=[0.65, 0.35]),
        "Age": rng.normal(29, 14, size=n).clip(0.5, 80).round(1),
        "SibSp": rng.choice([0, 1, 2, 3], size=n, p=[0.68, 0.23, 0.06, 0.03]),
        "Fare": rng.exponential(30, size=n).round(2),
        "Embarked": rng.choice(["S", "C", "Q"], size=n, p=[0.72, 0.19, 0.09]),
    })

    # Insert missing values
    df.loc[rng.choice(n, 40, replace=False), "Age"] = np.nan
    df.loc[rng.choice(n, 5, replace=False), "Embarked"] = np.nan

print("Raw shape:", df.shape)
print(df.head())

# ---------- 2. Data Cleaning ----------
print("\nMissing values before cleaning:\n", df.isnull().sum())

# Fill missing Age values with median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing Embarked values with mode
if "Embarked" in df.columns:
    df["Embarked"] = df["Embarked"].fillna(
        df["Embarked"].mode()[0]
    )

# Remove duplicate rows
df = df.drop_duplicates()

print("\nMissing values after cleaning:\n", df.isnull().sum())
print("\nCleaned shape:", df.shape)

# ---------- 3. Exploratory Data Analysis ----------
print("\nSummary statistics:\n", df.describe(include="all"))

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Survival rate by passenger class
if "Survived" in df.columns and "Pclass" in df.columns:
    sns.barplot(
        x="Pclass", y="Survived",
        data=df, ax=axes[0, 0], palette="viridis"
    )
    axes[0, 0].set_title("Survival Rate by Passenger Class")

# Survival rate by sex
if "Survived" in df.columns and "Sex" in df.columns:
    sns.barplot(
        x="Sex", y="Survived",
        data=df, ax=axes[0, 1], palette="magma"
    )
    axes[0, 1].set_title("Survival Rate by Sex")

# Age distribution
sns.histplot(
    df["Age"], bins=20, kde=True,
    ax=axes[1, 0], color="steelblue"
)
axes[1, 0].set_title("Age Distribution")

# Correlation heatmap
numeric_df = df.select_dtypes(include=[np.number])

sns.heatmap(
    numeric_df.corr(), annot=True,
    cmap="coolwarm", ax=axes[1, 1]
)
axes[1, 1].set_title("Correlation Heatmap")

plt.tight_layout()

# Save output chart
out_path = "task2_eda.png"
plt.savefig(out_path, dpi=150)
print(f"\nSaved EDA charts to {out_path}")

# ---------- 4. Key Insights ----------
if "Survived" in df.columns and "Sex" in df.columns:
    print("\nSurvival rate by sex:\n",
          df.groupby("Sex")["Survived"].mean())

if "Survived" in df.columns and "Pclass" in df.columns:
    print("\nSurvival rate by class:\n",
          df.groupby("Pclass")["Survived"].mean())

plt.show()
