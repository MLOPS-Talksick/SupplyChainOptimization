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

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

df = pd.read_csv("C:/Users/svaru/Downloads/processed_data.csv")

# Convert data types
df['Total Quantity'] = df['Total Quantity'].astype(int)
df['Date'] = pd.to_datetime(df['Date']) #, format='%d-%m-%Y')

# Sort data by date
df = df.sort_values('Date')

# Extract date features
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day
df['dayofweek'] = df['Date'].dt.dayofweek
df['dayofyear'] = df['Date'].dt.dayofyear
df['quarter'] = df['Date'].dt.quarter


products = df['Product Name'].unique()


# Calculate additional features that might help prediction
# 1. Rolling statistics for each product
df = df.sort_values(['Product Name', 'Date'])
df['rolling_mean_7d'] = df.groupby('Product Name')['Total Quantity'].transform(
    lambda x: x.rolling(window=7, min_periods=1).mean())
df['rolling_std_7d'] = df.groupby('Product Name')['Total Quantity'].transform(
    lambda x: x.rolling(window=7, min_periods=1).std().fillna(0))

# 2. Lag features
for lag in [1, 2, 3, 7]:  # 1-day, 2-day, 3-day, and 7-day lags
    df[f'lag_{lag}d'] = df.groupby('Product Name')['Total Quantity'].shift(lag)

# Fill NaN values
df = df.fillna(0)

# Encode product names
label_encoder = LabelEncoder()
df['product_encoded'] = label_encoder.fit_transform(df['Product Name'])

# Normalize numeric features
features = ['year', 'month', 'day', 'dayofweek', 'dayofyear', 'quarter', 
            'product_encoded', 'rolling_mean_7d', 'rolling_std_7d',
            'lag_1d', 'lag_2d', 'lag_3d', 'lag_7d']

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(df[features])
y_scaled = scaler_y.fit_transform(df[['Total Quantity']])

with open('scaler_X.pkl', 'wb') as f:
    pickle.dump(scaler_X, f)
    
with open('scaler_y.pkl', 'wb') as f:
    pickle.dump(scaler_y, f)
    
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Prepare data for LSTM (time steps)
def create_sequences(X, y, time_steps=10):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

# Group by product_encoded, then create sequences for each product
all_X_seq = []
all_y_seq = []

for product in df['product_encoded'].unique():
    product_indices = df[df['product_encoded'] == product].index
    product_X = X_scaled[product_indices]
    product_y = y_scaled[product_indices]
    
    if len(product_X) > 15:  # Ensure we have enough data points for sequences
        X_seq, y_seq = create_sequences(product_X, product_y, time_steps=5)
        all_X_seq.append(X_seq)
        all_y_seq.append(y_seq)

# Combine all sequences
X_seq = np.vstack(all_X_seq)
y_seq = np.vstack(all_y_seq)

# Split data into training (60%), validation (20%), and test (20%) sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)  # 0.25 of 80% is 20% of the total

# ======== HYPERPARAMETER TUNING WITH KERAS TUNER ========

class LSTMHyperModel(kt.HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        model = Sequential()
        
        # First LSTM layer with tunable units
        units_1 = hp.Int('units_1', min_value=32, max_value=128, step=32)
        dropout_1 = hp.Float('dropout_1', min_value=0.0, max_value=0.5, step=0.1)
        
        model.add(LSTM(units=units_1, 
                       activation=hp.Choice('activation_1', values=['relu', 'tanh']),
                       return_sequences=True,
                       input_shape=self.input_shape))
        model.add(Dropout(dropout_1))
        
        # Second LSTM layer with tunable units
        units_2 = hp.Int('units_2', min_value=16, max_value=64, step=16)
        dropout_2 = hp.Float('dropout_2', min_value=0.0, max_value=0.5, step=0.1)
        
        model.add(LSTM(units=units_2, 
                       activation=hp.Choice('activation_2', values=['relu', 'tanh'])))
        model.add(Dropout(dropout_2))
        
        # Dense hidden layer
        dense_units = hp.Int('dense_units', min_value=8, max_value=32, step=8)
        model.add(Dense(units=dense_units, 
                        activation=hp.Choice('dense_activation', values=['relu', 'linear'])))
        
        # Output layer
        model.add(Dense(1))
        
        # Compile model with tunable learning rate
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
        optimizer = hp.Choice('optimizer', values=['adam', 'rmsprop'])
        
        if optimizer == 'adam':
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        else:
            opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
            
        model.compile(optimizer=opt, loss= losses.MeanSquaredError())
        return model

# Initialize the tuner
input_shape = (X_train.shape[1], X_train.shape[2])
hypermodel = LSTMHyperModel(input_shape)

tuner = kt.BayesianOptimization(
    hypermodel,
    objective='val_loss',
    max_trials=20,
    directory='keras_tuner_dir',
    project_name='lstm_demand_forecasting'
)

# Define early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Search for the best hyperparameters
print("Starting hyperparameter search...")
tuner.search(
    X_train, y_train,
    epochs=30,  # Maximum number of epochs for each trial
    batch_size=32,
    validation_data=(X_val, y_val),  # Use separate validation set
    callbacks=[early_stopping],
    verbose=1
)

# Get the best hyperparameters and build the model
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best hyperparameters found:")
for param, value in best_hps.values.items():
    print(f"{param}: {value}")

# Build the model with the best hyperparameters
best_model = tuner.hypermodel.build(best_hps)

best_model.summary()

# Train the best model on combined train+validation sets for final training
# This is a common approach after hyperparameter tuning
X_train_full = np.concatenate([X_train, X_val], axis=0)
y_train_full = np.concatenate([y_train, y_val], axis=0)

# Define a separate early stopping callback for final training
final_early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# For monitoring purposes, we'll use a small validation split
history = best_model.fit(
    X_train_full, y_train_full,
    epochs=50,
    batch_size=32,
    validation_split=0.1,  # Small validation split just for monitoring
    callbacks=[final_early_stopping],
    verbose=1
)

# Evaluate the model
train_loss = best_model.evaluate(X_train_full, y_train_full, verbose=0)  # Evaluate on all training data
test_loss = best_model.evaluate(X_test, y_test, verbose=0)  # Test set was kept separate throughout
print(f'Train Loss: {train_loss:.4f}')
print(f'Test Loss: {test_loss:.4f}')

# Make predictions
y_pred_scaled = best_model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test)

# Calculate RMSE
rmse = np.sqrt(np.mean((y_pred - y_test_actual) ** 2))
print(f'RMSE: {rmse:.4f}')

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.tight_layout()
plt.savefig('training_history.png', bbox_inches='tight', dpi=300)
plt.close()  # Close the figure to ensure it's saved properly

# Plot predictions vs actual for a sample of test data
plt.figure(figsize=(12, 6))
plt.plot(y_test_actual[:100], label='Actual')
plt.plot(y_pred[:100], label='Predicted')
plt.title('Predicted vs Actual Quantities')
plt.ylabel('Quantity')
plt.xlabel('Sample Index')
plt.legend()
plt.tight_layout()
plt.savefig('prediction_results.png', bbox_inches='tight', dpi=300)
plt.close()  # Close the figure to ensure it's saved properly

# # Function to predict future quantities for a specific product - using the best model
# def predict_future_quantities(product_name, days=7):
#     # Get the last sequence for the product
#     product_idx = label_encoder.transform([product_name])[0]
#     product_data = df[df['product_encoded'] == product_idx].sort_values('Date')
    
#     if len(product_data) < 5:  # Need at least time_steps data points
#         return "Not enough historical data for this product"
    
#     last_date = product_data['Date'].max()
#     print(f"Last date for {product_name}: {last_date}")
#     future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days)
    
#     # Extract the last known sequence
#     last_sequence = X_scaled[product_data.index[-5:]]
    
#     predictions = []
#     current_sequence = last_sequence.copy()
    
#     for i in range(days):
#         # Reshape for prediction
#         current_sequence_reshaped = current_sequence.reshape(1, 5, -1)
        
#         # Predict next day
#         next_pred_scaled = best_model.predict(current_sequence_reshaped)
#         next_pred = scaler_y.inverse_transform(next_pred_scaled)[0][0]
#         predictions.append(next_pred)
        
#         # Get feature values for the next day
#         next_date = future_dates[i]
#         next_features = np.zeros(len(features))
        
#         # Update date features
#         next_features[0] = next_date.year
#         next_features[1] = next_date.month
#         next_features[2] = next_date.day
#         next_features[3] = next_date.dayofweek
#         next_features[4] = next_date.dayofyear
#         next_features[5] = next_date.quarter
        
#         # Product features remain the same
#         next_features[6] = product_idx  # Moved from index 7

#         # Update lag features based on predictions
#         if i == 0:
#             # For the first prediction, use values from the dataset
#             next_features[7] = product_data['rolling_mean_7d'].iloc[-1]  # rolling_mean_7d
#             next_features[8] = product_data['rolling_std_7d'].iloc[-1]   # rolling_std_7d
#             next_features[9] = product_data['Total Quantity'].iloc[-1]   # lag_1d
#             next_features[10] = product_data['Total Quantity'].iloc[-2] if len(product_data) > 1 else 0  # lag_2d
#             next_features[11] = product_data['Total Quantity'].iloc[-3] if len(product_data) > 2 else 0  # lag_3d
#             next_features[12] = product_data['Total Quantity'].iloc[-7] if len(product_data) > 6 else 0  # lag_7d
#         else:
#             # For subsequent predictions, use the predicted values
#             if i >= 7:
#                 next_features[7] = np.mean(predictions[i-7:i])  # rolling_mean_7d
#                 next_features[8] = np.std(predictions[i-7:i]) if len(predictions[i-7:i]) > 1 else 0  # rolling_std_7d
#             else:
#                 # For the first few days, use a mix of historical and predicted
#                 historical = list(product_data['Total Quantity'].iloc[-(7-i):])
#                 predicted = predictions[:i]
#                 combined = historical + predicted
#                 next_features[7] = np.mean(combined)  # rolling_mean_7d
#                 next_features[8] = np.std(combined) if len(combined) > 1 else 0  # rolling_std_7d

#             next_features[9] = predictions[i-1]  # lag_1d
#             next_features[10] = predictions[i-2] if i >= 2 else product_data['Total Quantity'].iloc[-1]  # lag_2d
#             next_features[11] = predictions[i-3] if i >= 3 else product_data['Total Quantity'].iloc[-2]  # lag_3d
#             next_features[12] = predictions[i-7] if i >= 7 else product_data['Total Quantity'].iloc[-6]  # lag_7d
        
#         # Scale the features
#         next_features_scaled = scaler_X.transform(next_features.reshape(1, -1))
        
#         # Update the sequence for the next iteration (remove first, add latest)
#         current_sequence = np.vstack([current_sequence[1:], next_features_scaled])
    
#     # Create a DataFrame with the predictions
#     future_df = pd.DataFrame({
#         'Date': future_dates,
#         'Product Name': product_name,
#         'Predicted_Quantity': [max(1, int(round(pred))) for pred in predictions]
#     })
    
#     return future_df

# # Example: Predict next 30 days for one product
# sample_product = products[4]
# future_predictions = predict_future_quantities(sample_product, days=7)
# print(f"\nFuture predictions for {sample_product}:")
# print(future_predictions)

# # Function to predict for all products
# def predict_all_products(days=30):
#     all_predictions = []
#     for product in products:
#         pred_df = predict_future_quantities(product, days)
#         if isinstance(pred_df, pd.DataFrame):
#             all_predictions.append(pred_df)
    
#     if all_predictions:
#         return pd.concat(all_predictions, ignore_index=True)
#     else:
#         return "No predictions available"
        
# # Save full predictions for all products
# all_product_predictions = predict_all_products(days=30)
# if isinstance(all_product_predictions, pd.DataFrame):
#     all_product_predictions.to_csv('all_product_predictions.csv', index=False)
#     print("All product predictions saved to 'all_product_predictions.csv'")


# Save the best model
best_model.save('best_lstm_model.keras')
print("Best model saved as 'best_lstm_model.keras'")

# Save the hyperparameter tuning results
with open('hyperparameter_results.txt', 'w') as f:
    f.write("Best hyperparameters:\n")
    for param, value in best_hps.values.items():
        f.write(f"{param}: {value}\n")
    f.write(f"\nTrain Loss: {train_loss:.4f}\n")
    f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write(f"RMSE: {rmse:.4f}\n")

print("Hyperparameter tuning results saved to 'hyperparameter_results.txt'")