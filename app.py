from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("model/titanic_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = [
            int(request.form["Pclass"]),
            int(request.form["Sex"]),
            float(request.form["Age"]),
            int(request.form["SibSp"]),
            int(request.form["Parch"]),
            float(request.form["Fare"]),
            int(request.form["Embarked"])
        ]

        sample = pd.DataFrame([data], columns=[
            "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"
        ])

        prediction = model.predict(sample)[0]

        result = "Survived" if prediction == 1 else "Did Not Survive"

        return render_template("index.html", result=result)

    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
