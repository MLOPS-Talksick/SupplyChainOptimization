# -*- coding: utf-8 -*-
"""prophet

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1vuK0o5r-CrKmAZem1_SyK1tjX4ZSzQwC
"""

import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm.notebook import tqdm
import warnings
import logging
import sys

# Completely suppress all warnings
warnings.filterwarnings('ignore')

# Set up a more aggressive logging configuration
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
logging.getLogger('stan').setLevel(logging.ERROR)

# Create a null handler to completely suppress logs
class NullHandler(logging.Handler):
    def emit(self, record):
        pass

# Apply null handler to relevant loggers
for logger_name in ['prophet', 'cmdstanpy', 'stan', 'matplotlib']:
    logger = logging.getLogger(logger_name)
    logger.addHandler(NullHandler())
    logger.propagate = False

# Silence stdout temporarily during Prophet model fitting
class SuppressStdoutStderr:
    def __enter__(self):
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = open('/dev/null', 'w')
        sys.stderr = open('/dev/null', 'w')

    def __exit__(self, *args):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self.stdout
        sys.stderr = self.stderr

df = pd.read_csv('ML.csv')
df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
products = df['Product Name'].unique().tolist()

def add_lag_features(df, product_name, lag_days=[1, 7]):
    product_df = df[df['Product Name'] == product_name].copy()
    product_df = product_df.sort_values('Date')

    for lag in lag_days:
        product_df[f'lag_{lag}'] = product_df['Total Quantity'].shift(lag)
    product_df['rolling_mean_7'] = product_df['Total Quantity'].rolling(window=7).mean()
    return product_df.dropna()

def train_prophet_model(product_data, product_name, use_hyperparams=False):
    prophet_data = product_data[['Date', 'Total Quantity']].rename(
        columns={'Date': 'ds', 'Total Quantity': 'y'}
    )

    regressors = []
    for col in ['lag_1', 'lag_7', 'rolling_mean_7']:
        if col in product_data.columns:
            prophet_data[col] = product_data[col]
            regressors.append(col)

    train_size = int(len(prophet_data) * 0.8)
    train_data = prophet_data.iloc[:train_size]
    test_data = prophet_data.iloc[train_size:]

    if len(train_data) < 2*7:
        print(f"Not enough data for {product_name}. Skipping.")
        return None, None, None, None, None

    best_params = {
        'yearly_seasonality': True,
        'weekly_seasonality': True,
        'daily_seasonality': False,
        'seasonality_mode': 'multiplicative',
        'changepoint_prior_scale': 0.1,
        'seasonality_prior_scale': 1.0
    }

    model = Prophet(**best_params)

    for regressor in regressors:
        model.add_regressor(regressor)

    # Use the suppression context manager during fitting
    with SuppressStdoutStderr():
        model.fit(train_data)

    forecast = model.predict(test_data)

    mae = mean_absolute_error(test_data['y'], forecast['yhat'])
    rmse = np.sqrt(mean_squared_error(test_data['y'], forecast['yhat']))
    r2 = r2_score(test_data['y'], forecast['yhat'])

    last_date = prophet_data['ds'].max()
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=30
    )
    future = pd.DataFrame({'ds': future_dates})

    if regressors:
        for regressor in regressors:
            if regressor == 'lag_1':
                future[regressor] = np.append(
                    prophet_data['y'].values[-1:],
                    [prophet_data['y'].mean()] * (len(future) - 1)
                )
            elif regressor == 'lag_7':
                last_values = prophet_data['y'].values[-7:]
                if len(last_values) < 7:
                    last_values = np.pad(
                        last_values,
                        (0, 7 - len(last_values)),
                        'constant',
                        constant_values=prophet_data['y'].mean()
                    )
                future[regressor] = np.append(
                    last_values,
                    [prophet_data['y'].mean()] * (len(future) - 7)
                )
            elif regressor == 'rolling_mean_7':
                future[regressor] = prophet_data['y'].mean()

    # Use the suppression context manager during prediction
    with SuppressStdoutStderr():
        future_forecast = model.predict(future)

    return model, future_forecast, mae, rmse, r2

models = {}
forecasts = {}
metrics = {}

for product in tqdm(products, desc="Processing products"):
    product_data = add_lag_features(df, product)

    if len(product_data) < 30:
        print(f"Not enough data for {product}. Skipping.")
        continue

    model, forecast, mae, rmse, r2 = train_prophet_model(
        product_data,
        product,
        use_hyperparams=False
    )

    if model is not None:
        models[product] = model
        forecasts[product] = forecast
        metrics[product] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}

if metrics:
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
    print("\nModel Performance Metrics:")
    print(metrics_df)
    print(f"Average MAE: {metrics_df['MAE'].mean():.2f}")
    print(f"Average RMSE: {metrics_df['RMSE'].mean():.2f}")
    print(f"Average R²: {metrics_df['R2'].mean():.4f}")

consolidated_forecast = pd.DataFrame()

for product in products:
    if product in forecasts:
        product_forecast = forecasts[product]
        temp_df = pd.DataFrame({
            'Date': product_forecast['ds'],
            'Product Name': product,
            'Forecasted_Quantity': product_forecast['yhat'],
            'Lower_Bound': product_forecast['yhat_lower'],
            'Upper_Bound': product_forecast['yhat_upper']
        })
        consolidated_forecast = pd.concat([consolidated_forecast, temp_df])

if not consolidated_forecast.empty:
    consolidated_forecast = consolidated_forecast.sort_values(['Date', 'Product Name'])
    consolidated_forecast.to_csv('Forecast.csv', index=False)
    print("Forecast saved to 'Forecast.csv'")

import os
import pickle
import logging
import math
import numpy as np
import pandas as pd
import mysql.connector
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import ks_2samp
import warnings
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

# -----------------------
# 1. HELPER FUNCTIONS
# -----------------------
def load_model(pickle_path: str):
    """Loads a pickle file and returns the model object."""
    with open(pickle_path, 'rb') as f:
        model = pickle.load(f)
    return model

def save_model(model, pickle_path: str):
    """Saves a model to a pickle file."""
    with open(pickle_path, 'wb') as f:
        pickle.dump(model, f)
    logging.info(f"Model saved to {pickle_path}")

def get_latest_data_from_mysql(host, user, password, database, query):
    """Connects to MySQL, runs a query, returns a DataFrame of the results."""
    conn = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def compute_rmse(y_true, y_pred):
    """Computes Root Mean Squared Error."""
    return math.sqrt(mean_squared_error(y_true, y_pred))

def compute_mape(y_true, y_pred):
    """
    Computes Mean Absolute Percentage Error.
    y_true: array-like of actual values
    y_pred: array-like of predicted values
    Returns MAPE in percentage.
    Ignores zero or near-zero actual values to avoid division-by-zero errors.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def ks_test_drift(ref_data, new_data, alpha=0.05):
    """
    Performs the Kolmogorov–Smirnov test to detect distribution drift.
    Returns True if drift is detected, else False.

    alpha (float): significance level. If p-value < alpha => drift.
    """
    stat, p_value = ks_2samp(ref_data, new_data)
    return p_value < alpha  # True => drift

# -----------------------
# 2. DRIFT DETECTION
# -----------------------
def check_data_drift(ref_df, new_df, numeric_cols=None, alpha=0.05):
    """
    Checks data drift for numeric columns by running K-S test on each column.
    ref_df: reference dataset (training or older stable data)
    new_df: new dataset
    numeric_cols: list of columns to test. If None, will auto-detect numeric columns.
    alpha: significance level for K-S test
    Returns: (bool, list_of_drifts)
        - bool: True if at least one column drifted significantly
        - list_of_drifts: list of columns that show drift
    """
    if numeric_cols is None:
        numeric_cols = ref_df.select_dtypes(include=[np.number]).columns.tolist()

    drift_detected_cols = []
    for col in numeric_cols:
        if col not in new_df.columns:
            continue  # skip if col missing in new_df
        ref_values = ref_df[col].dropna()
        new_values = new_df[col].dropna()

        if len(ref_values) < 2 or len(new_values) < 2:
            continue  # skip if not enough data for test

        drift_found = ks_test_drift(ref_values, new_values, alpha)
        if drift_found:
            drift_detected_cols.append(col)

    return (len(drift_detected_cols) > 0, drift_detected_cols)

def check_concept_drift(ref_errors, new_errors, alpha=0.05):
    """
    Performs concept drift detection by comparing the distribution of
    residuals (errors) between reference data and new data.
    If the error distributions differ significantly (K-S test), we assume concept drift.

    alpha (float): significance level. If p-value < alpha => concept drift.
    Returns True if concept drift is detected, else False.
    """
    if len(ref_errors) < 2 or len(new_errors) < 2:
        return False
    return ks_test_drift(ref_errors, new_errors, alpha)

# -----------------------
# 3. PROPHET MODEL FUNCTIONS
# -----------------------
def add_lag_features(df, product_name, lag_days=[1, 7]):
    """
    Add lag features and rolling mean to the product dataframe.
    Product dataframe should have 'Date' and 'Total Quantity' columns.
    """
    product_df = df[df['Product Name'] == product_name].copy()
    product_df = product_df.sort_values('Date')

    for lag in lag_days:
        product_df[f'lag_{lag}'] = product_df['Total Quantity'].shift(lag)
    product_df['rolling_mean_7'] = product_df['Total Quantity'].rolling(window=7).mean()
    return product_df.dropna()

def train_prophet_model(product_data, product_name, use_hyperparams=False):
    """
    Train a Prophet model for the given product data.
    Returns the trained model, forecast, and evaluation metrics.
    """
    prophet_data = product_data[['Date', 'Total Quantity']].rename(
        columns={'Date': 'ds', 'Total Quantity': 'y'}
    )

    regressors = []
    for col in ['lag_1', 'lag_7', 'rolling_mean_7']:
        if col in product_data.columns:
            prophet_data[col] = product_data[col]
            regressors.append(col)

    # Split data into train and test
    train_size = int(len(prophet_data) * 0.8)
    train_data = prophet_data.iloc[:train_size]
    test_data = prophet_data.iloc[train_size:]

    if len(train_data) < 2*7:
        logging.info(f"Not enough data for {product_name}. Skipping.")
        return None, None, None, None, None

    best_params = {
        'yearly_seasonality': True,
        'weekly_seasonality': True,
        'daily_seasonality': False,
        'seasonality_mode': 'multiplicative',
        'changepoint_prior_scale': 0.1,
        'seasonality_prior_scale': 1.0
    }

    model = Prophet(**best_params)
    for regressor in regressors:
        model.add_regressor(regressor)

    model.fit(train_data)

    forecast = model.predict(test_data)

    mae = mean_absolute_error(test_data['y'], forecast['yhat'])
    rmse = np.sqrt(mean_squared_error(test_data['y'], forecast['yhat']))
    r2 = r2_score(test_data['y'], forecast['yhat'])

    # Create future forecast data
    last_date = prophet_data['ds'].max()
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=30
    )
    future = pd.DataFrame({'ds': future_dates})

    # Add regressor values for future dates
    if regressors:
        for regressor in regressors:
            if regressor == 'lag_1':
                future[regressor] = np.append(
                    prophet_data['y'].values[-1:],
                    [prophet_data['y'].mean()] * (len(future) - 1)
                )
            elif regressor == 'lag_7':
                last_values = prophet_data['y'].values[-7:]
                if len(last_values) < 7:
                    last_values = np.pad(
                        last_values,
                        (0, 7 - len(last_values)),
                        'constant',
                        constant_values=prophet_data['y'].mean()
                    )
                future[regressor] = np.append(
                    last_values,
                    [prophet_data['y'].mean()] * (len(future) - 7)
                )
            elif regressor == 'rolling_mean_7':
                future[regressor] = prophet_data['y'].mean()

    # Generate forecast for future dates
    future_forecast = model.predict(future)
    return model, future_forecast, mae, rmse, r2

def get_prophet_predictions(model, features):
    """Convert prophet model results to match the format expected by the monitoring framework."""
    # For Prophet, features should include necessary columns for prediction
    prophet_features = features.copy()

    # Rename date column to ds (Prophet requirement)
    if 'date' in prophet_features.columns:
        prophet_features = prophet_features.rename(columns={'date': 'ds'})

    # Generate predictions
    forecast = model.predict(prophet_features)
    return forecast['yhat'].values

# -----------------------
# 4. MAIN SCRIPT
# -----------------------
def main():
    # 1. Load configuration from environment variables or config file
    host = os.getenv("MYSQL_HOST")
    user = os.getenv("MYSQL_USER")
    password = os.getenv("MYSQL_PASSWORD")
    database = os.getenv("MYSQL_DATABASE")
    model_dir = os.getenv("MODEL_DIR", "./models")

    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)

    # Performance thresholds
    rmse_threshold = float(os.getenv("RMSE_THRESHOLD", 10.0))
    mape_threshold = float(os.getenv("MAPE_THRESHOLD", 50.0))     # in percentage
    alpha_drift = float(os.getenv("ALPHA_DRIFT", 0.05))          # significance level for K-S tests

    # 2. Fetch data from MySQL
    # Adjust this query to match your table structure and include product information
    query_data = """
        SELECT
            date as Date,
            product_name as 'Product Name',
            quantity as 'Total Quantity',
            feature1,
            feature2
        FROM your_table
        ORDER BY date;
    """
    df = get_latest_data_from_mysql(host, user, password, database, query_data)

    # Ensure Date is datetime type
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    products = df['Product Name'].unique().tolist()

    # 3. Split data into recent and reference
    cutoff_date = df['Date'].max() - pd.Timedelta(days=7)
    new_df = df[df['Date'] >= cutoff_date].copy()
    ref_df = df[df['Date'] < cutoff_date].copy()

    logging.info(f"Processing {len(products)} products")

    models = {}
    forecasts = {}
    metrics = {}
    degraded_models = []

    # 4. Process each product
    for product in products:
        model_pickle_path = os.path.join(model_dir, f"model_{product.replace(' ', '_')}.pkl")

        # Check if there's enough data for the product
        product_data = add_lag_features(df, product)
        if len(product_data) < 30:
            logging.info(f"Not enough data for {product}. Skipping.")
            continue

        # Check if we already have a model for this product
        try:
            current_model = load_model(model_pickle_path)
            logging.info(f"Loaded existing model for {product}")
            model_exists = True
        except (FileNotFoundError, EOFError):
            logging.info(f"No existing model found for {product}. Will train new model.")
            model_exists = False

        # Prepare validation data for this product
        new_product_df = new_df[new_df['Product Name'] == product].copy()
        if new_product_df.empty:
            logging.info(f"No new data for {product}. Skipping validation.")
            continue

        # If model exists, validate it
        if model_exists:
            # 5. Validate model
            # Convert new_product_df to the format expected by Prophet for prediction
            prophet_features = new_product_df[['Date']].rename(columns={'Date': 'ds'})

            # Add regressors if the model uses them
            for col in ['lag_1', 'lag_7', 'rolling_mean_7']:
                if col in new_product_df.columns:
                    prophet_features[col] = new_product_df[col]

            # Generate predictions
            forecast = current_model.predict(prophet_features)
            y_pred = forecast['yhat'].values
            y_true = new_product_df['Total Quantity'].values

            # Compute metrics
            rmse_new = compute_rmse(y_true, y_pred)
            mape_new = compute_mape(y_true, y_pred)
            logging.info(f"Validation Results for {product}:")
            logging.info(f"  - RMSE: {rmse_new:.4f}")
            logging.info(f"  - MAPE: {mape_new:.2f}%")

            # Check for drift
            ref_product_df = ref_df[ref_df['Product Name'] == product].copy()

            if not ref_product_df.empty:
                # Data drift check
                numeric_cols = ['Total Quantity', 'lag_1', 'lag_7', 'rolling_mean_7']
                data_drift_detected, drifted_cols = check_data_drift(
                    ref_product_df, new_product_df,
                    numeric_cols=[col for col in numeric_cols if col in new_product_df.columns],
                    alpha=alpha_drift
                )

                if data_drift_detected:
                    logging.warning(f"Data drift detected for {product} in columns: {drifted_cols}")

                # Concept drift check
                prophet_ref_features = ref_product_df[['Date']].rename(columns={'Date': 'ds'})
                for col in ['lag_1', 'lag_7', 'rolling_mean_7']:
                    if col in ref_product_df.columns:
                        prophet_ref_features[col] = ref_product_df[col]

                forecast_ref = current_model.predict(prophet_ref_features)
                y_pred_ref = forecast_ref['yhat'].values
                y_true_ref = ref_product_df['Total Quantity'].values

                ref_errors = y_true_ref - y_pred_ref
                new_errors = y_true - y_pred

                concept_drift = check_concept_drift(ref_errors, new_errors, alpha=alpha_drift)
                if concept_drift:
                    logging.warning(f"Concept drift detected for {product} based on error distribution shift.")
            else:
                data_drift_detected = False
                concept_drift = False

            # Check if model needs retraining
            degraded = (rmse_new > rmse_threshold) or (mape_new > mape_threshold)
            if degraded:
                logging.warning(f"Model for {product} has degraded performance.")

            if degraded or data_drift_detected or concept_drift:
                logging.warning(f"Retraining model for {product}...")
                degraded_models.append(product)
                model_exists = False
            else:
                models[product] = current_model

        # Train new model if needed
        if not model_exists:
            logging.info(f"Training new model for {product}...")
            model, forecast, mae, rmse, r2 = train_prophet_model(
                product_data,
                product,
                use_hyperparams=False
            )

            if model is not None:
                models[product] = model
                forecasts[product] = forecast
                metrics[product] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}

                # Save the new model
                save_model(model, model_pickle_path)
                logging.info(f"New model saved for {product} with RMSE: {rmse:.4f}, R²: {r2:.4f}")

    # 6. Generate and save forecasts for all products with valid models
    consolidated_forecast = pd.DataFrame()
    for product in products:
        if product in models:
            # Generate future forecast for this product
            last_date = df[df['Product Name'] == product]['Date'].max()
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=30
            )
            future = pd.DataFrame({'ds': future_dates})

            # Add regressor values for future dates if needed
            product_data = df[df['Product Name'] == product].copy()
            regressors = []
            for col in ['lag_1', 'lag_7', 'rolling_mean_7']:
                if col in product_data.columns:
                    regressors.append(col)

            if regressors:
                for regressor in regressors:
                    if regressor == 'lag_1':
                        future[regressor] = np.append(
                            product_data['Total Quantity'].values[-1:],
                            [product_data['Total Quantity'].mean()] * (len(future) - 1)
                        )
                    elif regressor == 'lag_7':
                        last_values = product_data['Total Quantity'].values[-7:]
                        if len(last_values) < 7:
                            last_values = np.pad(
                                last_values,
                                (0, 7 - len(last_values)),
                                'constant',
                                constant_values=product_data['Total Quantity'].mean()
                            )
                        future[regressor] = np.append(
                            last_values,
                            [product_data['Total Quantity'].mean()] * (len(future) - 7)
                        )
                    elif regressor == 'rolling_mean_7':
                        future[regressor] = product_data['Total Quantity'].mean()

            # Generate the forecast
            product_forecast = models[product].predict(future)

            # Add to consolidated forecast
            temp_df = pd.DataFrame({
                'Date': product_forecast['ds'],
                'Product Name': product,
                'Forecasted_Quantity': product_forecast['yhat'],
                'Lower_Bound': product_forecast['yhat_lower'],
                'Upper_Bound': product_forecast['yhat_upper']
            })
            consolidated_forecast = pd.concat([consolidated_forecast, temp_df])

    # 7. Save consolidated forecast
    if not consolidated_forecast.empty:
        consolidated_forecast = consolidated_forecast.sort_values(['Date', 'Product Name'])
        consolidated_forecast.to_csv('Forecast.csv', index=False)
        logging.info("Forecast saved to 'Forecast.csv'")

    # 8. Report on model metrics
    if metrics:
        metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
        logging.info("\nModel Performance Metrics:")
        print(metrics_df)
        logging.info(f"Average MAE: {metrics_df['MAE'].mean():.2f}")
        logging.info(f"Average RMSE: {metrics_df['RMSE'].mean():.2f}")
        logging.info(f"Average R²: {metrics_df['R2'].mean():.4f}")

    # 9. Report on retraining
    if degraded_models:
        logging.info(f"Models retrained due to drift or degraded performance: {len(degraded_models)}")
        logging.info(f"Products with retrained models: {degraded_models}")

if __name__ == "__main__":
    main()