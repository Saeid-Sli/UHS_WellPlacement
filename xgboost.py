pip install scikit-learn==1.2.2
pip install --upgrade xgboost
pip install bayesian-optimization# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from bayes_opt import BayesianOptimization
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Step 1: Load dataset
data = pd.read_csv('dataset.csv')

# Step 2: Split dataset into features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Step 3: Split dataset into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Step 4: Normalize features and target
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = MinMaxScaler()
y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_val = scaler_y.transform(y_val.values.reshape(-1, 1))
y_test = scaler_y.transform(y_test.values.reshape(-1, 1))

# Step 5: Define objective function for Bayesian optimization
def xgb_cv(n_estimators, max_depth, learning_rate, min_child_weight, subsample, colsample_bytree):
    model = XGBRegressor(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        learning_rate=learning_rate,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=42,
        n_jobs=1
    )
    cval = cross_val_score(model, X_train_scaled, y_train.ravel(), scoring='neg_mean_squared_error', cv=5)
    return cval.mean()

# Step 6: Define hyperparameter search space
pbounds = {
    'n_estimators': (100, 1000),
    'max_depth': (3, 10),
    'learning_rate': (0.01, 0.3),
    'min_child_weight': (1, 10),
    'subsample': (0.5, 1),
    'colsample_bytree': (0.5, 1)
}

# Step 7: Perform Bayesian optimization
optimizer = BayesianOptimization(f=xgb_cv, pbounds=pbounds, random_state=42)
optimizer.maximize(init_points=10, n_iter=10)

# Step 8: Train final XGBoost model with best hyperparameters
best_params = optimizer.max['params']
best_params['n_estimators'] = int(best_params['n_estimators'])
best_params['max_depth'] = int(best_params['max_depth'])

xgb_model = XGBRegressor(**best_params, random_state=42, n_jobs=1)
eval_set = [(X_train_scaled, y_train), (X_val_scaled, y_val)]

start_train = time.time()
xgb_model.fit(X_train_scaled, y_train.ravel(), eval_set=eval_set, verbose=False)
train_time = time.time() - start_train
print(f"Train time: {train_time:.2f} seconds")

# Step 9: Evaluate model on validation and test sets
y_val_pred = xgb_model.predict(X_val_scaled)
y_test_pred = xgb_model.predict(X_test_scaled)

# Reverse normalization for target variable
y_val_pred_orig = scaler_y.inverse_transform(y_val_pred.reshape(-1, 1))
y_val_orig = scaler_y.inverse_transform(y_val)
y_test_pred_orig = scaler_y.inverse_transform(y_test_pred.reshape(-1, 1))
y_test_orig = scaler_y.inverse_transform(y_test)

# Calculate metrics
val_mse = mean_squared_error(y_val_orig, y_val_pred_orig)
val_r2 = r2_score(y_val_orig, y_val_pred_orig)
test_mse = mean_squared_error(y_test_orig, y_test_pred_orig)
test_r2 = r2_score(y_test_orig, y_test_pred_orig)

print(f"Validation MSE: {val_mse:.4f}, Validation R^2: {val_r2:.4f}")
print(f"Test MSE: {test_mse:.4f}, Test R^2: {test_r2:.4f}")

# Step 10: Cross-Validation
cv_scores = cross_val_score(xgb_model, X_train_scaled, y_train.ravel(), scoring='r2', cv=5)
print("Cross-validated R^2 scores:", cv_scores)
print("Mean Cross-validated R^2:", np.mean(cv_scores))

# Step 11: Residual Analysis
errors = y_test_orig.ravel() - y_test_pred_orig.ravel()
plt.figure(figsize=(8, 6))
plt.scatter(y_test_orig, errors, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Observed H2_Recovery')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

# Step 12: Simpler Model Baseline (Linear Regression)
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)
y_test_pred_linear = linear_model.predict(X_test_scaled)
y_test_pred_linear_orig = scaler_y.inverse_transform(y_test_pred_linear.reshape(-1, 1))
linear_r2 = r2_score(y_test_orig, y_test_pred_linear_orig)
print("Linear Regression Test R^2:", linear_r2)

# Step 13: Visualize Results

# Predicted vs. Observed Plot with increased font sizes
plt.figure(figsize=(8, 6))
plt.plot(y_test_pred_orig, y_test_orig, 'bo')
x = np.linspace(min(y_test_orig), max(y_test_orig), 1000)
plt.plot(x, x, 'r-')
plt.title(f'XGBoost Algorithm: R2={test_r2:.4f}', fontsize=16)  # Increased font size for title
plt.xlabel('Predicted H2 Recovery', fontsize=14)  # Increased font size for x-axis label
plt.ylabel('Observed H2 Recovery', fontsize=14)  # Increased font size for y-axis label
plt.tick_params(axis='both', which='major', labelsize=12)  # Increased font size for tick labels
plt.savefig('xgb_r2.png', dpi=300)
plt.show()


# Distribution of Errors
plt.figure(figsize=(8, 6))
sns.histplot(errors, kde=True, bins=30)
plt.title('XGBoost Algorithm: Distribution of Errors',fontsize=16)
plt.xlabel('Error',fontsize=14)
plt.ylabel('Frequency',fontsize=14)
plt.savefig('xg_diserror.png', dpi=300)
plt.show()

# Learning Curve
results = xgb_model.evals_result()
epochs = len(results['validation_0']['rmse'])
x_axis = range(epochs)

plt.figure(figsize=(8, 6))
plt.plot(x_axis, results['validation_0']['rmse'], label='Train RMSE')
plt.plot(x_axis, results['validation_1']['rmse'], label='Validation RMSE')
plt.xlabel('Number of Iterations', fontsize=14)  # Increased font size for x-axis label
plt.ylabel('RMSE', fontsize=14)  # Increased font size for y-axis label
plt.title('XGBoost Algorithm: Learning Curve', fontsize=16)  # Increased font size for title
plt.legend(fontsize=12)  # Increased font size for legend
plt.tick_params(axis='both', which='major', labelsize=12)  # Increased font size for tick labels
plt.savefig('xg_learning.png', dpi=300)
plt.show()


# Step 14: Save Results
best_params['Train Time'] = train_time
best_params['Validation MSE'] = val_mse
best_params['Validation R^2'] = val_r2
best_params['Test MSE'] = test_mse
best_params['Test R^2'] = test_r2

# Save best parameters to CSV
results_df = pd.DataFrame([best_params])
results_df.to_csv('xgb_best_params.csv', index=False)

print("Results saved to xgb_best_params.csv")
print(xgb_model.get_params()['n_jobs'])
