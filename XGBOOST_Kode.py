#XGBOOST
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import os

train = pd.read_csv("C:/Users/karlk/Documents/Bachelor/PowerData/DataSu/Training_Power_Data.csv")
test = pd.read_csv("C:/Users/karlk/Documents/Bachelor/PowerData/DataSu/Test_Power_Data.csv")
train.columns = train.columns.str.strip().str.replace("\\xa0", "")
test.columns = test.columns.str.strip().str.replace("\\xa0", "")
features = ["Package Temperature_0(C)", "CPU Utilization(%)"]
targets = ["Processor Power_0(Watt)"]

train.dropna(subset=features + targets, inplace=True)
test.dropna(subset=features, inplace=True)

scaler_X = StandardScaler()
train[features] = scaler_X.fit_transform(train[features])
test[features] = scaler_X.transform(test[features])

X_train = train[features].values
y_train_proc = train[targets[0]].values

X_test = test[features].values

model_proc = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=50, max_depth=3, n_jobs=-1)

model_proc.fit(X_train, y_train_proc)


pred_proc = model_proc.predict(X_test)


df_out = test.copy()
df_out["Predicted Processor Power (W)"] = pred_proc

df_out[["System Time", "Predicted Processor Power (W)"]].to_csv(
    "C:/Users/karlk/Documents/Bachelor/PowerData/DataSu/Filtered_  XGBoost_Power_Predictions_CLEAN.csv", index=False)