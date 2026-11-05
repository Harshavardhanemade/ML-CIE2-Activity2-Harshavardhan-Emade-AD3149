# Activity 2 – Simple House Price Prediction

**Course:** Machine Learning (CIE 2), Unit II – Supervised Learning: Regression  
**Algorithm:** Simple Linear Regression  
**Dataset:** California Housing (20,640 observations; 8 numerical input features and a continuous target)

## Objective
Predict a continuous house-value target from a numerical property/district feature, train a Linear Regression model, and evaluate it on unseen test data.

## Dataset and source
This project uses the California Housing dataset distributed by scikit-learn and originally obtained from the StatLib repository. The dataset represents 1990 California census block groups, not individual current property listings.

- Scikit-learn dataset documentation: https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset
- Original StatLib source: https://lib.stat.cmu.edu/datasets/houses.zip
- Dataset reference: Pace, R. Kelley and Ronald Barry (1997), “Sparse Spatial Autoregressions,” Statistics & Probability Letters, 33, 291–297.

The notebook/script downloads the dataset through `fetch_california_housing` on first run. Internet access is required for that initial download. No fabricated dataset or pre-filled metric values are included.

## Selected variables
- **Input X:** `MedInc` – median income in the block group.
- **Target y:** `MedHouseVal` – median house value, expressed in units of $100,000.
- `MedInc` is used as the single input feature so that the coefficient, intercept, and equation can be explained clearly. The relationship is predictive association, not proof of causation.

## Workflow
1. Load and inspect the dataset (shape, sample, data types, descriptive statistics).
2. Check missing values and duplicate rows.
3. Plot median income against median house value.
4. Split data into 80% training and 20% testing using `random_state=42`.
5. Train `LinearRegression` using training data only.
6. Generate test-set predictions and show actual/predicted values.
7. Calculate MAE, MSE, RMSE and R².
8. Plot actual vs predicted values with a perfect-prediction reference line.
9. Predict for a new median-income example chosen from the training data's median.
10. Save the model, metrics, sample predictions, and figures.

## Run in Google Colab
1. Upload `Activity_2_House_Price_Prediction.zip` and extract it, or upload the notebook.
2. Open `Activity_2_House_Price_Prediction.ipynb`.
3. Run all cells from top to bottom. The first cell installs dependencies if needed.
4. The dataset downloads automatically on first run.
5. Review the printed metrics and generated figures. Do not copy metric values from another run.

## Run locally
```bash
pip install -r requirements.txt
python run_project.py
```

## Output files
After running, the project creates:
- `outputs/california_housing_loaded.csv`
- `outputs/actual_vs_predicted_sample.csv`
- `outputs/metrics.json`
- `outputs/new_prediction.json`
- `outputs/linear_regression_model.joblib`
- `figures/income_vs_house_value.png`
- `figures/actual_vs_predicted.png`

## Metric interpretation
- **MAE:** average absolute prediction error, in units of $100,000.
- **MSE:** average squared prediction error, in squared target units.
- **RMSE:** square root of MSE; in units of $100,000.
- **R²:** compares the model with a mean-target baseline; it can be negative on test data.

## Limitations
This is a simple one-feature model using district-level historical data from 1990. Real property prices also depend on location, property condition, size, amenities, demand, interest rates and many other factors. The dataset's target has a ceiling effect. The model should not be used as a current real-estate valuation tool.

## Faculty acknowledgement
This project was completed as part of the Machine Learning CIE 2 activity. Faculty reference and Machine Learning learning resources:  
https://github.com/Princy2129

Faculty profile: [Princy2129](https://github.com/Princy2129)

## GitHub publication
Create your own **public** repository, for example `ML-CIE2-Activity2-ROLLNO`. Replace `ROLLNO` with your actual roll number. Upload the notebook, script, README and requirements file. Publish the repository URL for faculty review. This is your repository; do not attempt to add it to or modify the faculty's GitHub account.
