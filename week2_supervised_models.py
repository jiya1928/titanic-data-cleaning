"""
Titanic - Week 2 Task: Supervised ML Models
Repo: titanic-data-cleaning
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load data (replace path if needed)
df = pd.read_csv('train.csv')  # or Titanic-Dataset.csv

# Simple preprocessing for Week 2 demo
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
acc_lr = accuracy_score(y_test, lr.predict(X_test))
print(f"Logistic Regression Accuracy: {acc_lr:.2f}")

# Model 2
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
acc_rf = accuracy_score(y_test, rf.predict(X_test))
print(f"Random Forest Accuracy: {acc_rf:.2f}")

print("\nClassification Report - Random Forest:")
print(classification_report(y_test, rf.predict(X_test)))
