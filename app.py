from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    features = [
        float(request.form.get("gender",0)),
        float(request.form.get("married",0)),
        float(request.form.get("dependents",0)),
        float(request.form.get("education",0)),
        float(request.form.get("self_employed",0)),
        float(request.form.get("income",0)),
        float(request.form.get("coincome",0)),
        float(request.form.get("loanamount",0)),
        float(request.form.get("term",0)),
        float(request.form.get("credit",0)),
        float(request.form.get("area",0))
    ]

    features = np.array([features])

    features = scaler.transform(features)

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    if prediction == 1:
        result = "Loan Approved"
    else:
        result = "Loan Rejected"

    return render_template(
        "result.html",
        prediction=result,
        probability=round(probability*100,2)
    )


if __name__ == "__main__":
    app.run(debug=True)