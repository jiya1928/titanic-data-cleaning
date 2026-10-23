from flask import Flask, request, render_template_string
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open('titanic_model.pkl','rb') as f:
    model = pickle.load(f)

HTML = """
<h2>Titanic Survival Predictor - AI Pioneers Internship</h2>
<form method="post">
Pclass: <input name="pclass" value="3"><br>
Sex: <select name="sex"><option value="0">Male</option><option value="1">Female</option></select><br>
Age: <input name="age" value="22"><br>
Fare: <input name="fare" value="7.25"><br>
SibSp: <input name="sibsp" value="0"><br>
Parch: <input name="parch" value="0"><br>
<button>Predict</button>
</form>
<h3>{{result}}</h3>
"""

@app.route('/', methods=['GET','POST'])
def home():
    result = ""
    if request.method == 'POST':
        data = [float(request.form['pclass']), float(request.form['sex']), float(request.form['age']), float(request.form['fare']), float(request.form['sibsp']), float(request.form['parch'])]
        pred = model.predict([data])[0]
        result = "Survived" if pred==1 else "Not Survived"
    return render_template_string(HTML, result=result)

if __name__ == '__main__':
    app.run(debug=True)
