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
from google.cloud import aiplatform
from tensorflow.keras.models import load_model
from Data_Pipeline.scripts.logger import logger
from Data_Pipeline.scripts.utils import send_email
from google.cloud import storage
from ML_Models.scripts.utils import get_latest_data_from_cloud_sql

logger.info("Starting LSTM model training script")
# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

email = "talksick530@gmail.com"
# email = "svarunanusheel@gmail.com"

query = """
        SELECT 
            sale_date AS 'Date', 
            product_name AS 'Product Name', 
            total_quantity AS 'Total Quantity'
        FROM SALES
        ORDER BY sale_date;
    """

try:
    df = get_latest_data_from_cloud_sql(query=query)
    logger.info("Data fetched from Cloud SQL")
except Exception as e:
    logger.error(f"Error fetching data from Cloud SQL: {e}")
    send_email(
        emailid=email,
        subject="Error fetching data from Cloud SQL",
        body=f"An error occurred during fetching data from Cloud SQL: {str(e)}",
    )
    raise

current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

# Convert data types
try:
    df['Total Quantity'] = df['Total Quantity'].astype(int)
    df['Date'] = pd.to_datetime(df['Date'])  # , format='%d-%m-%Y')
except Exception as e:
    logger.error(f"Error in data type conversion: {e}")
    send_email(
        emailid=email,
        subject="Error in data type conversion",
        body=f"An error occurred during data type conversion: {str(e)}",
    )
    raise

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

logger.info("Starting feature engineering")
# 1. Rolling statistics for each product
try:
    df = df.sort_values(['Product Name', 'Date'])
    df['rolling_mean_7d'] = df.groupby('Product Name')['Total Quantity'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean())
    df['rolling_std_7d'] = df.groupby('Product Name')['Total Quantity'].transform(
        lambda x: x.rolling(window=7, min_periods=1).std().fillna(0))
except Exception as e:
    logger.error(f"Error in feature engineering: {e}")
    send_email(
        emailid=email,
        subject="Error in feature engineering",
        body=f"An error occurred during feature engineering: {str(e)}",
    )
    raise

# 2. Lag features
for lag in [1, 2, 3, 7]:  # 1-day, 2-day, 3-day, and 7-day lags
    df[f'lag_{lag}d'] = df.groupby('Product Name')['Total Quantity'].shift(lag)

# Fill NaN values
df = df.fillna(0)

# Encode product names
try:
    logger.info("Encoding product names")
    label_encoder = LabelEncoder()
    df['product_encoded'] = label_encoder.fit_transform(df['Product Name'])
except Exception as e:
    logger.error(f"Error encoding product names: {e}")
    send_email(
        emailid=email,
        subject="Error encoding product names",
        body=f"An error occurred during encoding product names: {str(e)}",
    )
    raise

# Normalize numeric features
features = ['year', 'month', 'day', 'dayofweek', 'dayofyear', 'quarter', 
            'product_encoded', 'rolling_mean_7d', 'rolling_std_7d',
            'lag_1d', 'lag_2d', 'lag_3d', 'lag_7d']

logger.info("creating preprocessing objects")
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

try:
    X_scaled = scaler_X.fit_transform(df[features])
    y_scaled = scaler_y.fit_transform(df[['Total Quantity']])
except Exception as e:
    logger.error(f"Error in scaling data: {e}")
    send_email(
        emailid=email,
        subject="Error in scaling data",
        body=f"An error occurred during scaling data: {str(e)}",
    )
    raise

try:
    with open('scaler_X.pkl', 'wb') as f:
        pickle.dump(scaler_X, f)
    
    with open('scaler_y.pkl', 'wb') as f:
        pickle.dump(scaler_y, f)
    
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
except Exception as e:
    logger.error(f"Error saving preprocessing objects: {e}")
    send_email(
        emailid=email,
        subject="Error saving preprocessing objects",
        body=f"An error occurred while saving preprocessing objects: {str(e)}",
    )
    raise

logger.info("Preprocessing objects saved")

# Prepare data for LSTM (time steps)
def create_sequences(X, y, time_steps=10):
    logger.info("Creating sequences")
    Xs, ys = [], []
    try:
        for i in range(len(X) - time_steps):
            Xs.append(X[i:(i + time_steps)])
            ys.append(y[i + time_steps])
    except Exception as e:
        logger.error(f"Error in creating sequences: {e}")
        send_email(
            emailid=email,
            subject="Error in creating sequences",
            body=f"An error occurred during creating sequences: {str(e)}",
        )
        raise
    return np.array(Xs), np.array(ys)

# Group by product_encoded, then create sequences for each product
all_X_seq = []
all_y_seq = []

for product in df['product_encoded'].unique():
    logger.info("Creating sequences for product: %s", product)
    try:
        product_indices = df[df['product_encoded'] == product].index
        product_X = X_scaled[product_indices]
        product_y = y_scaled[product_indices]
        
        if len(product_X) > 15:  # Ensure we have enough data points for sequences
            X_seq, y_seq = create_sequences(product_X, product_y, time_steps=5)
            all_X_seq.append(X_seq)
            all_y_seq.append(y_seq)
    except Exception as e:
        logger.error(f"Error creating sequences for product {product}: {e}")
        send_email(
            emailid=email,
            subject=f"Error creating sequences for {product}",
            body=f"An error occurred creating sequences for {product}: {str(e)}",
        )
        continue

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
        logger.info("Building LSTM model")
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

try:
    logger.info("Initializing Keras Tuner")
    tuner = kt.BayesianOptimization(
        hypermodel,
        objective='val_loss',
        max_trials=20,
        directory='keras_tuner_dir',
        project_name='lstm_demand_forecasting'
    )
except Exception as e:
    logger.error(f"Error initializing Keras Tuner: {e}")
    send_email(
        emailid=email,
        subject="Error initializing Keras Tune",
        body=f"An error occurred during initializing Keras Tune: {str(e)}",
    )
    raise

# Define early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

try:
    logger.info("Starting hyperparameter search")
    tuner.search(
        X_train, y_train,
        epochs=30,  # Maximum number of epochs for each trial
        batch_size=32,
        validation_data=(X_val, y_val),  # Use separate validation set
        callbacks=[early_stopping],
        verbose=1
    )
except Exception as e:
    logger.error(f"Error in hyperparameter search: {e}")
    send_email(
        emailid=email,
        subject="Error in hyperparameter search",
        body=f"An error occurred during hyperparameter search: {str(e)}",
    )
    raise

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
try:
    logger.info("Plotting training history")
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'training_history_{current_datetime}.png', bbox_inches='tight', dpi=300)
    plt.close()  # Close the figure to ensure it's saved properly
except Exception as e:
    logger.error(f"Error plotting training history: {e}")
    send_email(
        emailid=email,
        subject="Error plotting training history",
        body=f"An error occurred while plotting training history: {str(e)}",
    )

# Plot predictions vs actual for a sample of test data
try:
    logger.info("plotting predictions vs actual")
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_actual[:100], label='Actual')
    plt.plot(y_pred[:100], label='Predicted')
    plt.title('Predicted vs Actual Quantities')
    plt.ylabel('Quantity')
    plt.xlabel('Sample Index')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'prediction_results_{current_datetime}.png', bbox_inches='tight', dpi=300)
    plt.close()  # Close the figure to ensure it's saved properly
except Exception as e:
    logger.error(f"Error plotting predictions vs actual: {e}")
    send_email(
        emailid=email,
        subject="Error plotting predictions vs actual",
        body=f"An error occurred while plotting predictions vs actual: {str(e)}",
    )

def upload_to_gcs(file_name, gcs_bucket_name):
    # Initialize Google Cloud Storage client
    storage_client = storage.Client()

    # Upload Keras model
    try:
        logger.info(f"Uploading {file_name} to GCS")
        bucket = storage_client.bucket(gcs_bucket_name)
        blob = bucket.blob(file_name)
        blob.upload_from_filename(file_name)
        print(f'Uploaded {file_name} to GCS: gs://{gcs_bucket_name}/{file_name}')
        return True
    except Exception as e:
        print(f'Error uploading {file_name} to GCS: {e}')
        send_email(
            emailid=email,
            subject=f"Error uploading {file_name} to GCS",
            body=f"An error occurred while uploading {file_name} to GCS: {str(e)}",
        )
        return False

def save_artifacts(model, model_name = "lstm_model.keras"):
    """
    Save model locally and to GCS bucket.
    Returns True if successful, False otherwise.
    """
    try:
        logger.info("Saving model artifacts")
        model.save(model_name)

        upload_to_gcs(model_name, "trained-model-1")
        upload_to_gcs("scaler_y.pkl", "model_training_1")
        upload_to_gcs("label_encoder.pkl", "model_training_1")
        upload_to_gcs("scaler_X.pkl", "model_training_1")
        upload_to_gcs(f"hyperparameter_results_{current_datetime}.txt", "model_training_1")
        upload_to_gcs(f'training_history_{current_datetime}.png', "model_training_1")
        upload_to_gcs(f'prediction_results_{current_datetime}.png', "model_training_1")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        send_email(
            emailid=email,
            subject="Error saving model artifacts",
            body=f"An error occurred while saving model artifacts: {str(e)}",
        )
        return False

save_artifacts(best_model, 'lstm_model.keras')
print("Best model saved as 'lstm_model.keras'")

# Save the hyperparameter tuning results
try:
    logger.info("saving hyperparameter tuning results")
    with open('hyperparameter_results.txt', 'w') as f:
        f.write("Best hyperparameters:\n")
        for param, value in best_hps.values.items():
            f.write(f"{param}: {value}\n")
        f.write(f"\nTrain Loss: {train_loss:.4f}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
except Exception as e:
    print(f"Error saving hyperparameter tuning results: {e}")
    send_email(
        emailid=email,
        subject="Error saving hyperparameter tuning result",
        body=f"An error occurred while saving hyperparameter tuning result: {str(e)}",
    )


try:
    logger.info("Deleting temporary files")
    file_paths = ["scaler_y.pkl", "label_encoder.pkl", "scaler_X.pkl", 
                  f"hyperparameter_results_{current_datetime}.txt", 
                  f'training_history_{current_datetime}.png', 
                  f'prediction_results_{current_datetime}.png']


    # Loop through the list and delete each file
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"File {file_path} deleted successfully.")
        else:
            print(f"The file {file_path} does not exist.")

except Exception as e:
    print(f"Error deleting file: {e}")
    send_email(
        emailid=email,
        subject="Error deleting file",
        body=f"An error occurred while deleting file: {str(e)}",
    )

send_email(
    emailid=email,
    subject="Model run successful",
    body=f"Successfully trained and saved the LSTM model. \n\n"
    f"Train Loss: {train_loss:.4f}\n",
)
