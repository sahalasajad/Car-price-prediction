from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load the trained model
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        form_data = request.form.to_dict()

        make = form_data["make"]
        aspiration = form_data["aspiration"]
        body_style = form_data["body_style"]
        drive_wheels = form_data["drive_wheels"]
        engine_size = float(form_data["engine_size"])
        horsepower = float(form_data["horsepower"])
        city_mpg = float(form_data["city_mpg"])
        highway_mpg = float(form_data["highway_mpg"])
        curb_weight = float(form_data["curb_weight"])
        compression_ratio = float(form_data["compression_ratio"])

        input_df = pd.DataFrame([{
            "make": make,
            "aspiration": aspiration,
            "body-style": body_style,
            "drive-wheels": drive_wheels,
            "engine-size": engine_size,
            "horsepower": horsepower,
            "city-mpg": city_mpg,
            "highway-mpg": highway_mpg,
            "curb-weight": curb_weight,
            "compression-ratio": compression_ratio
        }])

        prediction = model.predict(input_df)[0]
        prediction_inr = round(prediction * 83.0)

        return render_template("index.html",
                               prediction_text=f"Estimated Price: ${prediction:,.2f} (~₹{prediction_inr:,.0f})",
                               form_data=form_data)
    except Exception as e:
        return render_template("index.html", prediction_text="Error: " + str(e))

@app.route("/insights")
def insights():
    metrics = []
    try:
        with open("static/metrics.txt", "r") as f:
            metrics = f.read().splitlines()
    except:
        metrics = []
    return render_template("insights.html", metrics=metrics)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

