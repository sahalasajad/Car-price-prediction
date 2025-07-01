import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv('car_data.csv', encoding='latin1')
df.columns = df.columns.str.strip().str.lower()
df.dropna(inplace=True)
df = df.loc[:, ~df.columns.str.contains('^unnamed')]

# Define features and target
X = df[['make', 'aspiration', 'body-style', 'drive-wheels',
        'engine-size', 'horsepower', 'city-mpg', 'highway-mpg',
        'curb-weight', 'compression-ratio']]
y = df['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Load trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Predict
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Evaluation:")
print(f"🔸 MAE: {mae:.2f}")
print(f"🔸 RMSE: {rmse:.2f}")
print(f"🔸 R² Score: {r2:.2f}")
# Save metrics to text file for HTML display
with open("static/metrics.txt", "w") as f:
    f.write(f"MAE: {mae:.2f}\\n")
    f.write(f"RMSE: {rmse:.2f}\\n")
    f.write(f"R² Score: {r2:.2f}\\n")


# Plot 1: Actual vs Predicted
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.grid(True)
plt.tight_layout()
plt.savefig("static/actual_vs_predicted.png")
plt.show()

# Plot 2: Feature Importances
importances = model.named_steps['regressor'].feature_importances_
feature_names = model.named_steps['preprocessor'].get_feature_names_out()

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title("Top 10 Feature Importances")
plt.tight_layout()
plt.savefig("static/feature_importance.png")
plt.show()
