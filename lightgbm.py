pip install bayesian-optimizationimport numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
import seaborn as sns
import time
import csv
import warnings

# Suppress FutureWarning from scikit-learn
warnings.filterwarnings('ignore', category=FutureWarning)

# Step 2: Load dataset
data = pd.read_csv('dataset.csv')

# Replace whitespaces in feature names with underscores
data.columns = data.columns.str.replace(' ', '_')

# Step 3: Split dataset into features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split dataset into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define evaluation set
eval_s = [(X_train, y_train), (X_val, y_val)]

# Step 4: Define objective function for Bayesian optimization
def lgbm_cv(n_estimators, max_depth, learning_rate, min_child_weight, subsample, colsample_bytree, num_leaves):
    estimator = lgb.LGBMRegressor(
        num_leaves=int(num_leaves),
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        learning_rate=learning_rate,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=42,
        force_col_wise=True,
        n_jobs=1
    )
    cval = cross_val_score(estimator, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
    return cval.mean()


# Step 5: Define search space for hyperparameters
pbounds = {
    'num_leaves': (5, 10),
    'n_estimators': (50, 100),
    'max_depth': (3, 6),
    'learning_rate': (0.01, 0.1),
    'min_child_weight': (1, 6),
    'subsample': (0.5, 1),
    'colsample_bytree': (0.5, 1)
}

# Step 6: Run Bayesian optimization
optimizer = BayesianOptimization(
    f=lgbm_cv,
    pbounds=pbounds,
    random_state=42,
)

optimizer.maximize(init_points=10, n_iter=10)

# Step 7: Train LightGBM model using best hyperparameters
best_params = optimizer.max['params']
best_params['n_estimators'] = int(best_params['n_estimators'])
best_params['num_leaves'] = int(best_params['num_leaves'])
best_params['max_depth'] = int(best_params['max_depth'])
lgbm_model = lgb.LGBMRegressor(**best_params, random_state=42, force_col_wise=True, n_jobs=1)

start = time.time()
lgbm_model.fit(X_train, y_train, eval_set=eval_s, eval_metric='rmse')
stop = time.time()
train_time = stop - start
print('Train time:', train_time)

# Step 8: Evaluate model on validation set
start = time.time()
y_val_pred = lgbm_model.predict(X_val)
stop = time.time()
val_time = stop - start
mse_val = mean_squared_error(y_val, y_val_pred)
r2_val = r2_score(y_val, y_val_pred)
print(f'Validation MSE: {mse_val}, Validation R-squared: {r2_val}, Validation Time: {val_time}')

# Step 9: Evaluate model on test set
start = time.time()
y_test_pred = lgbm_model.predict(X_test)
stop = time.time()
test_time = stop - start
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
print(f'Test MSE: {mse_test}, Test R-squared: {r2_test}, Test Time: {test_time}')

# Plotting the result with accurate R2 formatting
R2_formatted = f"{r2_test:.4f}"  # Format R2 to 4 decimal places
Measures = f"R2={R2_formatted}"
x = np.linspace(min(min(y_test_pred), min(y_test)), max(max(y_test_pred), max(y_test)), 1000)
fig = plt.figure()
fig.set_figheight(6)
fig.set_figwidth(8)
plt.plot(y_test_pred, y_test, 'bo', x, x, 'r')
plt.title(f'LightGBM Algorithm: {Measures}', fontsize=16)  # Use formatted R2 in title with increased font size
plt.xlabel('Predicted H2 Recovery', fontsize=14)
plt.ylabel('Observed H2 Recovery', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=12)  # Increase font size for tick labels
plt.savefig('lgbm_r2.png', dpi=300)
plt.show()



# Distribution of Errors
errors = y_test - y_test_pred
plt.figure(figsize=(8, 6))
sns.histplot(errors, kde=True, bins=30)
plt.title('LightGBM Algorithm: Distribution of Errors',fontsize=16)
plt.xlabel('Error',fontsize=14)
plt.ylabel('Frequency',fontsize=14)
plt.savefig('lg_diserror.png', dpi=300)
plt.show()

# Learning Curve with increased font sizes
results = lgbm_model.evals_result_
epochs = len(results['training']['rmse'])
x_axis = range(0, epochs)

plt.figure(figsize=(8, 6))
plt.plot(x_axis, results['training']['rmse'], label='Train RMSE')
plt.plot(x_axis, results['valid_1']['rmse'], label='Validation RMSE')
plt.xlabel('Number of Iterations', fontsize=14)  # Increased font size for x-axis label
plt.ylabel('RMSE', fontsize=14)  # Increased font size for y-axis label
plt.title('LightGBM Algorithm: Learning Curve', fontsize=16)  # Increased font size for title
plt.legend(fontsize=12)  # Increased font size for legend
plt.tick_params(axis='both', which='major', labelsize=12)  # Increased font size for tick labels
plt.savefig('lg_learning.png', dpi=300)
plt.show()


# Save best parameters to CSV
with open('lgb_best_params.csv', mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['parameter', 'value'])
    writer.writeheader()
    for param, value in best_params.items():
        writer.writerow({'parameter': param, 'value': value})

# Save model times to CSV
with open('lgb_model_times.csv', mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['stage', 'time'])
    writer.writeheader()
    writer.writerow({'stage': 'train', 'time': train_time})
    writer.writerow({'stage': 'val', 'time': val_time})
    writer.writerow({'stage': 'test', 'time': test_time})

# Feature Importance
feature_importance = lgbm_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importance})
feature_importance_df['Importance_percentage'] = (feature_importance / feature_importance.sum()) * 100
feature_importance_df = feature_importance_df.sort_values(by='Importance_percentage', ascending=False)

# Plot Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance_percentage', y='Feature', data=feature_importance_df, color='teal', width=0.6)
plt.title('LightGBM Algorithm: Feature Importance')
plt.xlabel('Importance (%)')
plt.ylabel('Feature')
plt.savefig('lgbm_importance_percentage.png', dpi=300)
plt.show()
print(lgbm_model.get_params()['n_jobs'])

