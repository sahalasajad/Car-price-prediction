from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html', form_data={})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form inputs
        data = {
            'make': request.form['make'],
            'aspiration': request.form['aspiration'],
            'body-style': request.form['body_style'],
            'drive-wheels': request.form['drive_wheels'],
            'engine-size': float(request.form['engine_size']),
            'horsepower': float(request.form['horsepower']),
            'city-mpg': float(request.form['city_mpg']),
            'highway-mpg': float(request.form['highway_mpg']),
            'curb-weight': float(request.form['curb_weight']),
            'compression-ratio': float(request.form['compression_ratio'])
        }

        df = pd.DataFrame([data])
        usd_price = model.predict(df)[0]
        inr_price = usd_price * 83  # Convert to INR

        return render_template(
            'index.html',
            prediction_text=f"Estimated Car Price: ${usd_price:,.2f} (₹{inr_price:,.0f})",
            form_data=data
        )
    except Exception as e:
        return render_template(
            'index.html',
            prediction_text=f"Error: {str(e)}",
            form_data={}
        )

if __name__ == '__main__':
    app.run(debug=True)


@app.route('/insights')
def insights():
    try:
        with open("static/metrics.txt", "r") as f:
            metrics = f.read().splitlines()
    except:
        metrics = []
    return render_template('insights.html', metrics=metrics)

