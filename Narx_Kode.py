#Narx
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import RegressorChain
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
scaler_y = StandardScaler()
X_train = scaler_X.fit_transform(train[features])
y_train = scaler_y.fit_transform(train[targets])
X_test = scaler_X.transform(test[features])

def create_lagged_features(X, y, lag=2):
    X_lagged, y_lagged = [], []
    for i in range(lag, len(X)):
        lagged_input = np.hstack([X[i - j] for j in range(lag, 0, -1)])
        X_lagged.append(np.hstack((lagged_input, y[i - 1])))
        y_lagged.append(y[i])
    return np.array(X_lagged), np.array(y_lagged)

lag = 2
X_narx, y_narx = create_lagged_features(X_train, y_train, lag)
model = RegressorChain(LinearRegression())
model.fit(X_narx, y_narx)

def narx_forecast(model, X_input, y_seed, lag):
    preds = []
    x = list(X_input[:lag])
    y = list(y_seed)
    for i in range(lag, len(X_input)):
        x_input = np.hstack([x[-j] for j in range(lag, 0, -1)])
        input_vector = np.hstack((x_input, y[-1]))
        y_pred = model.predict([input_vector])[0]
        preds.append(y_pred)
        x.append(X_input[i])
        y.append(y_pred)
    return np.array(preds)

y_init = list(y_train[-2:])
y_preds_scaled = narx_forecast(model, X_test, y_init, lag)
y_preds = scaler_y.inverse_transform(y_preds_scaled)

result = test.iloc[lag:].copy().reset_index(drop=True)
result["Predicted Processor Power (W)"] = y_preds[:, 0]

result[["System Time", "Predicted Processor Power (W)"]].to_csv(
    "C:/Users/karlk/Documents/Bachelor/PowerData/DataSu/Filtered_NARX_Power_Predictions_CLEAN.csv", index=False)