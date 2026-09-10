"""
Prodigy InfoTech - Data Science Internship
Task 02: Perform data cleaning and exploratory data analysis (EDA) on the Titanic dataset.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set clean aesthetic styling
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

def run_titanic_eda():
    print("=" * 65)
    print("PRODIGY INFOTECH - TASK 02: TITANIC DATA CLEANING & EDA")
    print("=" * 65)

    # 1. Load Dataset
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "train.csv")
    if not os.path.exists(data_path):
        data_path = "train.csv"
    
    print(f"\n[Step 1] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Dataset Dimensions: {df.shape[0]} rows, {df.shape[1]} columns\n")
    print("First 5 records:")
    print(df.head())

    # 2. Check Missing Values
    print("\n[Step 2] Checking Missing Values Before Cleaning:")
    missing = df.isnull().sum()
    print(missing[missing > 0])

    # 3. Data Cleaning
    print("\n[Step 3] Handling Missing Values & Feature Engineering...")
    # Fill missing Age with median age based on passenger class
    df["Age"] = df.groupby("Pclass")["Age"].transform(lambda x: x.fillna(x.median()))
    
    # Fill missing Embarked with the mode (most common value: 'S')
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    
    # Flag cabin presence and drop high-missing Cabin column
    df["Has_Cabin"] = df["Cabin"].apply(lambda x: 0 if pd.isna(x) else 1)
    df.drop(columns=["Cabin"], inplace=True)

    # Create FamilySize feature
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    print("Missing values after cleaning:")
    print(df.isnull().sum())

    # 4. Exploratory Visualizations
    vis_dir = os.path.join(current_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    print(f"\n[Step 4] Generating EDA Visualizations in: {vis_dir}/")

    # Plot 1: Survival rate by Gender
    plt.figure(figsize=(6, 4))
    sns.barplot(data=df, x="Sex", y="Survived", hue="Sex", palette="Blues_d", errorbar=None, legend=False)
    plt.title("Survival Rate by Gender", fontsize=13, weight="bold")
    plt.ylabel("Survival Rate")
    plt.xlabel("Gender")
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "01_survival_by_gender.png"))
    plt.close()

    # Plot 2: Survival rate by Ticket Class
    plt.figure(figsize=(6, 4))
    sns.barplot(data=df, x="Pclass", y="Survived", hue="Pclass", palette="viridis", errorbar=None, legend=False)
    plt.title("Survival Rate by Passenger Class", fontsize=13, weight="bold")
    plt.ylabel("Survival Rate")
    plt.xlabel("Passenger Class (1st, 2nd, 3rd)")
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "02_survival_by_class.png"))
    plt.close()

    # Plot 3: Age Distribution by Survival
    plt.figure(figsize=(8, 4))
    sns.histplot(data=df, x="Age", hue="Survived", kde=True, bins=30, palette="coolwarm", element="step")
    plt.title("Age Distribution: Non-Survivors (0) vs Survivors (1)", fontsize=13, weight="bold")
    plt.xlabel("Age")
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "03_age_distribution.png"))
    plt.close()

    # Plot 4: Correlation Heatmap
    plt.figure(figsize=(8, 6))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="mako", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix of Numerical Features", fontsize=13, weight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "04_correlation_matrix.png"))
    plt.close()

    print("Charts successfully saved!")

    # 5. Summary Findings
    female_surv = df[df["Sex"] == "female"]["Survived"].mean() * 100
    male_surv = df[df["Sex"] == "male"]["Survived"].mean() * 100
    p1_surv = df[df["Pclass"] == 1]["Survived"].mean() * 100
    p3_surv = df[df["Pclass"] == 3]["Survived"].mean() * 100

    print("\n[Step 5] Key Findings:")
    print(f" - Gender: Female survival was {female_surv:.1f}% compared to {male_surv:.1f}% for males.")
    print(f" - Socio-Economic Status: 1st class passengers survived at {p1_surv:.1f}%, while 3rd class stood at {p3_surv:.1f}%.")
    print(" - Age: Children under 10 had a visibly higher survival rate ('women and children first' protocol).")
    print("=" * 65)

if __name__ == "__main__":
    run_titanic_eda()
