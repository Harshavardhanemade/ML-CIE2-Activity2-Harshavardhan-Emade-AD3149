"""
Activity 2: Simple House Price Prediction
Dataset: California Housing (scikit-learn / StatLib)
Run: python run_project.py
The script downloads the official dataset on first run; internet is required only for this first download.
"""
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

# 1. Load dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame.copy()
df.to_csv(OUT / "california_housing_loaded.csv", index=False)
print("Dataset shape:", df.shape)
print(df.head())
print("\nData types:\n", df.dtypes)
print("\nSummary statistics:\n", df.describe().round(3))
print("\nMissing values:\n", df.isna().sum())
print("Duplicate rows:", int(df.duplicated().sum()))

# 2. Choose one meaningful numerical input and continuous target.
# MedInc is median household income in a census block group.
# MedHouseVal is median house value in units of $100,000.
feature = "MedInc"
target = "MedHouseVal"
X = df[[feature]]
y = df[target]

# 3. Visualise feature-target relationship
plt.figure(figsize=(8, 5))
plt.scatter(X[feature], y, alpha=0.25, s=10)
plt.title("Median Income vs Median House Value")
plt.xlabel("Median income (tens of thousands of USD)")
plt.ylabel("Median house value (units of $100,000)")
plt.tight_layout()
plt.savefig(FIG / "income_vs_house_value.png", dpi=160)
plt.close()

# 4. Train-test split (80:20); test set is not used for fitting.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"\nTraining observations: {len(X_train)}")
print(f"Testing observations: {len(X_test)}")

# 5. Train simple linear regression
model = LinearRegression()
model.fit(X_train, y_train)
coef = float(model.coef_[0])
intercept = float(model.intercept_)
print("\nIntercept:", intercept)
print("Coefficient:", coef)
print(f"Model equation: MedHouseVal = {intercept:.6f} + ({coef:.6f} × MedInc)")

# 6. Test predictions and metrics
pred = model.predict(X_test)
mae = mean_absolute_error(y_test, pred)
mse = mean_squared_error(y_test, pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, pred)
comparison = pd.DataFrame({
    "Actual_MedHouseVal": y_test.reset_index(drop=True),
    "Predicted_MedHouseVal": pd.Series(pred)
})
comparison.head(20).to_csv(OUT / "actual_vs_predicted_sample.csv", index=False)
print("\nActual vs predicted sample:\n", comparison.head(10).round(3))
metrics = {
    "MAE": float(mae), "MSE": float(mse), "RMSE": float(rmse), "R2": float(r2),
    "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
    "coefficient": coef, "intercept": intercept,
    "target_units": "hundreds of thousands of US dollars"
}
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
print("\nEvaluation metrics (target units are $100,000):")
for k, v in metrics.items():
    print(f"{k}: {v}")

# 7. Actual vs predicted plot with perfect-prediction reference line
plt.figure(figsize=(7, 6))
plt.scatter(y_test, pred, alpha=0.25, s=10)
low = min(float(y_test.min()), float(pred.min()))
high = max(float(y_test.max()), float(pred.max()))
plt.plot([low, high], [low, high], linestyle="--", label="Perfect prediction")
plt.title("Actual vs Predicted Median House Values")
plt.xlabel("Actual median house value ($100,000 units)")
plt.ylabel("Predicted median house value ($100,000 units)")
plt.legend()
plt.tight_layout()
plt.savefig(FIG / "actual_vs_predicted.png", dpi=160)
plt.close()

# 8. New example, kept within the observed training-feature range.
new_income = float(X_train[feature].median())
new_prediction = float(model.predict(pd.DataFrame({feature: [new_income]}))[0])
new_result = {
    "MedInc": new_income,
    "Predicted_MedHouseVal_100k_USD": new_prediction,
    "Predicted_USD_approx": new_prediction * 100000
}
(OUT / "new_prediction.json").write_text(json.dumps(new_result, indent=2))
print("\nNew example:", new_result)

# Save the fitted model for reproducibility
import joblib
joblib.dump(model, OUT / "linear_regression_model.joblib")
print("\nSaved outputs/, figures/, and trained model.")
