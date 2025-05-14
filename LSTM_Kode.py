# LSTM
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import os

torch.manual_seed(42)
torch.use_deterministic_algorithms(True)


train_path = "C:/Users/karlk/Documents/Bachelor/PowerData/DataSu/Training_Power_Data.csv"
test_path = "C:/Users/karlk/Documents/Bachelor/PowerData/DataSu/Test_Power_Data.csv"
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
train.columns = train.columns.str.strip().str.replace("\\xa0", "")
test.columns = test.columns.str.strip().str.replace("\\xa0", "")

features = ["Package Temperature_0(C)", "CPU Utilization(%)"]
targets = ["Processor Power_0(Watt)"]

train.dropna(subset=features + targets, inplace=True)
test.dropna(subset=features, inplace=True)


scaler_X = StandardScaler()
scaler_y_pwr = StandardScaler()

train[features] = scaler_X.fit_transform(train[features])
train["scaled_pwr"] = scaler_y_pwr.fit_transform(train[[targets[0]]])
test[features] = scaler_X.transform(test[features])


def create_sequences(data, labels, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps])
        y.append(labels[i+time_steps])
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32)

time_steps = 20
X_lstm, y_pwr = create_sequences(train[features].values, train["scaled_pwr"].values, time_steps)



class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

lstm_pwr = LSTMModel(len(features))


def train_lstm(model, X, y, epochs=30):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X).squeeze()
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()

train_lstm(lstm_pwr, X_lstm, y_pwr)



X_test_lstm = []
for i in range(len(test) - time_steps):
    X_test_lstm.append(test[features].iloc[i:i+time_steps].values)
X_test_lstm = torch.tensor(np.array(X_test_lstm), dtype=torch.float32)

lstm_pred_pwr = lstm_pwr(X_test_lstm).detach().numpy().flatten()


lstm_pred_pwr = scaler_y_pwr.inverse_transform(lstm_pred_pwr.reshape(-1, 1)).flatten()


df_out = test.iloc[time_steps:].reset_index(drop=True)
df_out["Predicted Processor Power (W)"] = lstm_pred_pwr

df_out[["System Time", "Predicted Processor Power (W)"]].to_csv(
    "C:/Users/karlk/Documents/Bachelor/PowerData/DataSu/FFiltered_LSTM_Power_Predictions_CLEAN.csv", index=False)
