# Task 02: Titanic Data Cleaning & Exploratory Data Analysis (EDA)

## Overview
This task focuses on exploring and cleaning the famous **Titanic: Machine Learning from Disaster** dataset from Kaggle. The goal is to uncover behavioral, demographic, and socioeconomic patterns that influenced passenger survival.

## Dataset
- `train.csv`: Contains passenger details (Age, Sex, Pclass, Fare, SibSp, Parch, Cabin, Embarked) along with the ground truth survival label (`Survived`).

## Workflow
1. **Data Inspection**: Checked data structure, datatypes, and missing value counts.
2. **Data Cleaning & Handling Missing Data**:
   - Imputed `Age` with the median age of corresponding passenger ticket classes (`Pclass`).
   - Imputed `Embarked` with the mode (`'S'`).
   - Handled `Cabin` (over 77% missing) by engineering a binary indicator `Has_Cabin` and dropping the raw column.
   - Engineered a `FamilySize` feature (`SibSp + Parch + 1`).
3. **Exploratory Data Analysis**:
   - Visualized survival distributions across gender and ticket classes.
   - Analyzed age distributions for survivors vs non-survivors.
   - Built a correlation heatmap of numeric variables.

## Key Insights
- **Gender**: Women had a survival rate of ~74.2%, compared to ~18.9% for men.
- **Socioeconomic Class**: 1st class passengers survived at 63.0%, compared to only 24.2% for 3rd class passengers.
- **Age**: Young children had prioritized access to lifeboats, showing distinct peaks in survival.

## How to Run
Run the standalone Python script:
```bash
python titanic_eda.py
```
Or open the interactive Jupyter Notebook:
```bash
jupyter notebook titanic_eda.ipynb
```
