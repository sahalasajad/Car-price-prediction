import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pickle

# Load dataset
df = pd.read_csv('car_data.csv', encoding='latin1')
df.columns = df.columns.str.strip().str.lower()
df.dropna(inplace=True)
df = df.loc[:, ~df.columns.str.contains('^unnamed')]

# Define features and target
X = df[['make', 'aspiration', 'body-style', 'drive-wheels',
        'engine-size', 'horsepower', 'city-mpg', 'highway-mpg', 'curb-weight', 'compression-ratio']]
y = df['price']

# Preprocessing
categorical_cols = ['make', 'aspiration', 'body-style', 'drive-wheels']
numerical_cols = ['engine-size', 'horsepower', 'city-mpg', 'highway-mpg', 'curb-weight', 'compression-ratio']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
], remainder='passthrough')

# Pipeline
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())
])

# Split and train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
model_pipeline.fit(X_train, y_train)

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model_pipeline, f)

print("✅ Model trained and saved as model.pkl")

# 🔍 Evaluation
y_pred = model_pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n📊 Evaluation Metrics:")
print(f"🔸 MAE (Mean Absolute Error): {mae:.2f}")
print(f"🔸 RMSE (Root Mean Squared Error): {rmse:.2f}")
print(f"🔸 R² Score: {r2:.2f}")
