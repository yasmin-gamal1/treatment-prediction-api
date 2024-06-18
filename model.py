# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 23:14:26 2024

@author: HP
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_excel("C:/Users/HP/Desktop/second_model/test.xlsx")
data

X = data[['ER', 'PR', 'HER2', 'Ki67 (%)']]
y = data['Treatment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

accuracy = rf.score(X_test, y_test)
print(f'Accuracy: {accuracy}')

importances = rf.feature_importances_
features = X.columns
indices = importances.argsort()

plt.figure(figsize=(8, 6))
plt.title('Feature Importance')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

treatment_counts = data['Treatment'].value_counts()

# Create a bar plot
plt.figure(figsize=(24, 6))
sns.barplot(x=treatment_counts.index, y=treatment_counts.values)
plt.title('Number of Patients by Treatment Type')
plt.xlabel('Treatment Type')
plt.ylabel('Number of Patients')
plt.show()


import joblib

# Save the model to a file
joblib.dump(rf, 'model.pkl')

joblib.dump(rf, "C:/Users/HP/Desktop/second_model/model.pkl")



input_data = np.array([[1, 1, 0, 50]])
prediction = rf.predict(input_data)
print("Predicted treatment:", prediction)