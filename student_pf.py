from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('student_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    hours = int(request.form['hours'])
    attendance = int(request.form['attendance'])
    marks = int(request.form['marks'])

    data = np.array([[hours, attendance, marks]])
    pred = model.predict(data)[0]

    result = "PASS ✅" if pred == 1 else "FAIL ❌"

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
