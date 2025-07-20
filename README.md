
# 🚗 Car Price Prediction App

This project predicts the price of a car based on its specifications using a trained Machine Learning model. Built as part of an internship at **Ediglobe Technologies**.

## 🔍 Features
- Predicts price in **USD** and converts to **INR**
- Evaluation with **MAE, RMSE, R²**
- Visual insights: Actual vs Predicted chart
- Modern web UI built with Flask & HTML/CSS

## 📊 ML Stack
- Model: `RandomForestRegressor`
- Tools: `pandas`, `scikit-learn`, `matplotlib`, `seaborn`
- Deployment: Flask (local or cloud-ready)

## 💻 How to Run Locally

1. Clone this repo
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run model training:
   ```
   python model_training.py
   ```
4. Run evaluation (to generate charts):
   ```
   python evaluation.py
   ```
5. Launch app:
   ```
   python app.py
   ```

## 🔗 Live Demo
https://car-price-prediction-qts6.onrender.com

## 📁 Folder Structure

```
├── app.py
├── car_data.csv
├── evaluation.py
├── model_training.py
├── model.pkl
├── requirements.txt
├── Procfile
├── static/
│   ├── actual_vs_predicted.png
│   ├── metrics.txt
├── templates/
│   ├── index.html
│   ├── insights.html
└── README.md
```
