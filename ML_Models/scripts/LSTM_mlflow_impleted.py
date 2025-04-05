import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import losses
import tensorflow as tf
import keras_tuner as kt
import shap
import pickle
import os

import mlflow
import mlflow.tensorflow

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ==== Load Data ====
df = pd.read_excel("/Users/sheethunaik/Documents/Forecast_MLOPS/SupplyChainOptimization/Supermarket Transactions New.xlsx")
df.columns = df.columns.str.strip()  # Remove any extra spaces

# Convert and clean data
df['Quantity'] = df['Quantity'].astype(int)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Date features
df = df.sort_values('Date')
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day
df['dayofweek'] = df['Date'].dt.dayofweek
df['dayofyear'] = df['Date'].dt.dayofyear
df['quarter'] = df['Date'].dt.quarter

products = df['Product Name'].unique()
df = df.sort_values(['Product Name', 'Date'])

# Rolling & Lag features
df['rolling_mean_7d'] = df.groupby('Product Name')['Quantity'].transform(
    lambda x: x.rolling(window=7, min_periods=1).mean())

df['rolling_std_7d'] = df.groupby('Product Name')['Quantity'].transform(
    lambda x: x.rolling(window=7, min_periods=1).std().fillna(0))

for lag in [1, 2, 3, 7]:
    df[f'lag_{lag}d'] = df.groupby('Product Name')['Quantity'].shift(lag)


df = df.fillna(0)

# Encode + scale
label_encoder = LabelEncoder()
df['product_encoded'] = label_encoder.fit_transform(df['Product Name'])

features = ['year', 'month', 'day', 'dayofweek', 'dayofyear', 'quarter',
            'product_encoded', 'rolling_mean_7d', 'rolling_std_7d',
            'lag_1d', 'lag_2d', 'lag_3d', 'lag_7d']
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(df[features])
y_scaled = scaler_y.fit_transform(df[['Quantity']])

with open('scaler_X.pkl', 'wb') as f:
    pickle.dump(scaler_X, f)
with open('scaler_y.pkl', 'wb') as f:
    pickle.dump(scaler_y, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Sequence creation
def create_sequences(X, y, time_steps=10):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

all_X_seq = []
all_y_seq = []
for product in df['product_encoded'].unique():
    product_indices = df[df['product_encoded'] == product].index
    product_X = X_scaled[product_indices]
    product_y = y_scaled[product_indices]
    if len(product_X) > 15:
        X_seq, y_seq = create_sequences(product_X, product_y, time_steps=5)
        all_X_seq.append(X_seq)
        all_y_seq.append(y_seq)

X_seq = np.vstack(all_X_seq)
y_seq = np.vstack(all_y_seq)

X_train_val, X_test, y_train_val, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

# ======== Keras Tuner ========
class LSTMHyperModel(kt.HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape
    def build(self, hp):
        model = Sequential()
        model.add(LSTM(units=hp.Int('units_1', 32, 128, step=32),
                       activation=hp.Choice('activation_1', ['relu', 'tanh']),
                       return_sequences=True,
                       input_shape=self.input_shape))
        model.add(Dropout(hp.Float('dropout_1', 0.0, 0.5, step=0.1)))
        model.add(LSTM(units=hp.Int('units_2', 16, 64, step=16),
                       activation=hp.Choice('activation_2', ['relu', 'tanh'])))
        model.add(Dropout(hp.Float('dropout_2', 0.0, 0.5, step=0.1)))
        model.add(Dense(hp.Int('dense_units', 8, 32, step=8),
                        activation=hp.Choice('dense_activation', ['relu', 'linear'])))
        model.add(Dense(1))
        optimizer = hp.Choice('optimizer', ['adam', 'rmsprop'])
        lr = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
        opt = tf.keras.optimizers.Adam(learning_rate=lr) if optimizer == 'adam' else tf.keras.optimizers.RMSprop(learning_rate=lr)
        model.compile(optimizer=opt, loss=losses.MeanSquaredError())
        return model

input_shape = (X_train.shape[1], X_train.shape[2])
tuner = kt.BayesianOptimization(
    LSTMHyperModel(input_shape),
    objective='val_loss',
    max_trials=20,
    directory='keras_tuner_dir',
    project_name='lstm_demand_forecasting'
)

tuner.search(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_val, y_val), callbacks=[EarlyStopping(monitor='val_loss', patience=5)], verbose=1)

# Best model
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.hypermodel.build(best_hps)

X_train_full = np.concatenate([X_train, X_val], axis=0)
y_train_full = np.concatenate([y_train, y_val], axis=0)
history = best_model.fit(X_train_full, y_train_full, epochs=50, batch_size=32,
                         validation_split=0.1, callbacks=[EarlyStopping(monitor='val_loss', patience=10)], verbose=1)

train_loss = best_model.evaluate(X_train_full, y_train_full, verbose=0)
test_loss = best_model.evaluate(X_test, y_test, verbose=0)

y_pred_scaled = best_model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test)
rmse = np.sqrt(np.mean((y_pred - y_test_actual) ** 2))

# Plots
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
plt.savefig('training_history.png'); plt.close()

plt.figure(figsize=(12, 6))
plt.plot(y_test_actual[:100], label='Actual')
plt.plot(y_pred[:100], label='Predicted')
plt.title('Predicted vs Actual')
plt.xlabel('Sample'); plt.ylabel('Quantity'); plt.legend()
plt.savefig('prediction_results.png'); plt.close()

with open('hyperparameter_results.txt', 'w') as f:
    f.write("Best hyperparameters:\n")
    for param, value in best_hps.values.items():
        f.write(f"{param}: {value}\n")
    f.write(f"\nTrain Loss: {train_loss:.4f}\nTest Loss: {test_loss:.4f}\nRMSE: {rmse:.4f}\n")

# ==== MLflow Tracking ====
mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("DemandForecasting-LSTM")

def validate_model_health(rmse_val, threshold=1000):
    if rmse_val > threshold:
        print(f"❌ Model RMSE {rmse_val:.2f} exceeds threshold {threshold}")
        return False
    print(f"✅ Model RMSE {rmse_val:.2f} is within healthy range.")
    return True

with mlflow.start_run(run_name="lstm_demand_model"):
    mlflow.log_metric("train_loss", train_loss)
    mlflow.log_metric("test_loss", test_loss)
    mlflow.log_metric("rmse", rmse)

    best_model.save("best_lstm_model.keras")
    mlflow.keras.log_model(best_model, artifact_path="lstm_model")


    for file in ["training_history.png", "prediction_results.png", "hyperparameter_results.txt"]:
        if os.path.exists(file):
            mlflow.log_artifact(file)

    health_passed = validate_model_health(rmse)
    mlflow.log_metric("model_health_passed", int(health_passed))

    if health_passed:
        mlflow.register_model(
            model_uri=f"runs:/{mlflow.active_run().info.run_id}/lstm_model",
            name="LSTMDemandForecast"
        )
