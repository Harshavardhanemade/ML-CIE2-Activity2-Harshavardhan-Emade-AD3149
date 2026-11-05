# CIE 2 – Activity 2 Report
## Simple House Price Prediction using Linear Regression

**Student Name:** ____________________  
**Roll Number:** ____________________  
**Class/Division:** T.Y. B.Tech AI & DS / ______  
**Course:** Machine Learning [ADPC509]  
**Academic Year:** 2026–27

### 1. Aim
To build a Linear Regression model for predicting a continuous house-value target and evaluate it using regression metrics.

### 2. Dataset
The California Housing dataset from scikit-learn (originally StatLib) is used. It contains 20,640 observations, eight numerical predictive features and a continuous median-house-value target. Each observation represents a California census block group from the 1990 census; it is not an individual property listing.

Source: https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset

### 3. Input and target
- Input feature (X): `MedInc` – median income in a block group.
- Target (y): `MedHouseVal` – median house value in units of $100,000.

Median income is selected as the single numerical input to demonstrate simple linear regression and make the model equation interpretable.

### 4. Data inspection and preparation
The notebook displays the first and last records, dataset dimensions, data types, descriptive statistics, missing-value counts and duplicate-row count. The official dataset documentation reports no missing values. The notebook checks this in the loaded copy. No ID column is used as a predictor. No unsupported row deletion or scaling is applied.

### 5. Relationship visualisation
A scatter plot places `MedInc` on the horizontal axis and `MedHouseVal` on the vertical axis. The plot is used to inspect the association. A visible association does not establish causation.

### 6. Train-test split
The data is divided into 80% training and 20% testing with `random_state=42`. The training set is used to estimate model parameters; the test set is reserved for evaluation on unseen observations.

### 7. Linear Regression model
The fitted model has the form:

`MedHouseVal = intercept + (coefficient × MedInc)`

The actual intercept and coefficient are printed by the executed notebook/script. The coefficient represents the estimated change in target units for a one-unit increase in `MedInc`; the intercept is the model's estimated target when `MedInc` is zero.

### 8. Predictions and evaluation
The notebook/script presents actual and predicted test values and calculates:
- MAE – average absolute error.
- MSE – average squared error.
- RMSE – error in the target's units.
- R² – comparison with a mean-based baseline; a negative test R² is possible.

**Important:** Run all notebook cells to populate the actual coefficient, intercept, predictions and metric values. They are intentionally not invented in this report.

### 9. Actual vs predicted plot
The second plot compares actual test values with predictions and includes a 45-degree perfect-prediction reference line. Points closer to the line indicate more accurate predictions.

### 10. New example prediction
The code selects the median `MedInc` value from the training set (within the training feature range) and predicts its median house value. The input and prediction are saved in `outputs/new_prediction.json`. State the resulting value and units after executing the notebook.

### 11. Conclusion
This activity demonstrates a complete introductory regression workflow: dataset inspection, feature and target selection, visualisation, train-test split, Linear Regression training, prediction, metric-based evaluation and a new example prediction. A one-feature model is easy to interpret but cannot represent all factors that affect house prices. The data is historical and aggregated at census-block-group level, so it is not suitable for estimating current prices of individual houses.

### Faculty acknowledgement
This project was completed as part of the Machine Learning CIE 2 activity. Faculty reference and Machine Learning learning resources: https://github.com/Princy2129
