import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('train.csv')
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Sex'] = df['Sex'].map({'male':0,'female':1})
X = df[['Pclass','Sex','Age','Fare','SibSp','Parch']]
y = df['Survived']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

with open('titanic_model.pkl','wb') as f:
    pickle.dump(model, f)

print("Model saved as titanic_model.pkl - Ready for deployment with Flask")
