# Titanic Data Cleaning and Visualization

## Project Overview

This project focuses on cleaning, preprocessing, and visualizing Titanic passenger data using Python.

## Tools and Libraries

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

## Data Cleaning

The dataset was examined for missing values and data types.

The following preprocessing steps were performed:

- Missing `Age` values were replaced with the median age.
- The `Cabin` column was removed because it contained many missing values.
- Categorical variables were converted into numerical form.
- `Sex` was encoded using Label Encoding.
- `Embarked` was encoded using One-Hot Encoding.

## Visualizations

The project includes:

- Age distribution of passengers
- Survival by passenger class
- Survival by gender

## Key Findings

The sample shows differences in survival based on passenger class and gender.

The analysis was performed on a small 10-row practice dataset, so the findings should not be generalized to the entire Titanic passenger population.

## Files

- `titanic data cleaning.ipynb` — Jupyter Notebook containing the complete analysis and visualizations.
- `titanic_cleaned.csv` — Cleaned and encoded dataset.

## Conclusion

The project demonstrates basic data cleaning, categorical encoding, exploratory data analysis, and data visualization using Python.
