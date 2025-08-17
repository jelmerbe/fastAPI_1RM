import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
import pickle

# Load dataset
data = pd.read_csv("StrengthLevel_data.csv")

# Prepare features (X) and targets (y)
X = data.iloc[:, 3:]  # Assuming first 3 columns are metadata
y = X.copy()

# Iterate over each exercise (column in y)
models = {}
for exercise in y.columns:
    # Prepare target for the current exercise
    target = y[exercise]
    
    # Drop rows where the target is NaN
    valid_rows = ~target.isna()
    X_valid = X[valid_rows]
    y_valid = target[valid_rows]
    
    # Train the model for the current exercise
    model = XGBRegressor(objective="reg:squarederror", eval_metric="rmse")
    model.fit(X_valid, y_valid)
    
    # Save the model for the current exercise
    models[exercise] = model

# Save all models to a file
with open("xgb_1rm_models.pkl", "wb") as f:
    pickle.dump(models, f)

# Predict missing values for a user
# user_data = X.iloc[0].copy()  # Example user
# user_data[user_data == -1] = np.nan  # Restore NaN for missing values
# predicted_1rms = model.predict(user_data.values.reshape(1, -1))

# print(predicted_1rms)
