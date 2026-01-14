import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier

# Dataset
data = {
    'StudyHours': [1, 2, 3, 4, 5, 6, 7, 8, 2, 3],
    'Attendance': [40, 50, 60, 70, 80, 90, 85, 95, 55, 65],
    'InternalMarks': [20, 25, 35, 45, 55, 65, 70, 80, 30, 40],
    'Result': [0, 0, 0, 1, 1, 1, 1, 1, 0, 1]  # 1 = PASS, 0 = FAIL
}

df = pd.DataFrame(data)

X = df.drop('Result', axis=1)
y = df['Result']

model = DecisionTreeClassifier()
model.fit(X, y)

pickle.dump(model, open('student_model.pkl', 'wb'))

print("Student Result Model Trained Successfully")
