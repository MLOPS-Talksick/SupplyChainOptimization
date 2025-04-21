import pandas as pd
import numpy as np
from google.cloud.sql.connector import Connector
import sqlalchemy
import os
from datetime import timedelta
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv
from sqlalchemy import text
import logging
from scipy import stats
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_serving_utils')

load_dotenv()

host = os.getenv("MYSQL_HOST")
user = os.getenv("MYSQL_USER") 
password = os.getenv("MYSQL_PASSWORD")
database = os.getenv("MYSQL_DATABASE")
instance = os.getenv("INSTANCE_CONN_NAME", "primordial-veld-450618-n4:us-central1:mlops-sql")

def get_latest_data_from_cloud_sql(query, port="3306"):
    """
    Connects to a Google Cloud SQL instance using TCP (public IP or Cloud SQL Proxy)
    and returns query results as a DataFrame.

    Args:
        host (str): The Cloud SQL instance IP address or localhost (if using Cloud SQL Proxy).
        port (int): The port number (typically 3306 for MySQL).
        user (str): Database username.
        password (str): Database password.
        database (str): Database name.
        query (str): SQL query to execute.

    Returns:
        pd.DataFrame: Query results.
    """

    connector = Connector()

    def getconn():
        conn = connector.connect(
            instance,
            "pymysql",  # Database driver
            user=user,  # Database user
            password=password,  # Database password
            db=database,
            ip_type="PRIVATE",
        )
        return conn

    pool = sqlalchemy.create_engine(
        "mysql+pymysql://",  # or "postgresql+pg8000://" for PostgreSQL, "mssql+pytds://" for SQL Server
        creator=getconn,
    )

    df = pd.read_sql(query, pool)
    connector.close()
    return df


def upsert_df(df: pd.DataFrame, table_name: str):
    """
    Inserts or updates rows in a MySQL table based on duplicate keys.
    If a record with the same primary key exists, it will be replaced with the new record.
    
    Parameters:
        df (pd.DataFrame): The DataFrame to insert/update.
        table_name (str): The target table name.
        engine: SQLAlchemy engine.
    """

    connector = Connector()

    def getconn():
        conn = connector.connect(
            instance,
            "pymysql",  # Database driver
            user=user,  # Database user
            password=password,  # Database password
            db=database,
            ip_type="PRIVATE",
        )
        return conn

    pool = sqlalchemy.create_engine(
        "mysql+pymysql://",  # or "postgresql+pg8000://" for PostgreSQL, "mssql+pytds://" for SQL Server
        creator=getconn,
    )

    # Convert DataFrame to a list of dictionaries (each dict represents a row)
    data = df.to_dict(orient='records')
    
    # Build dynamic column list and named placeholders
    columns = df.columns.tolist()
    col_names = ", ".join(columns)
    placeholders = ", ".join(":" + col for col in columns)
    
    # Build the update clause to update every column with its new value
    update_clause = ", ".join(f"{col} = VALUES({col})" for col in columns)
    
    # Construct the SQL query using ON DUPLICATE KEY UPDATE
    sql = text(
        f"INSERT INTO {table_name} ({col_names}) VALUES ({placeholders}) "
        f"ON DUPLICATE KEY UPDATE {update_clause}"
    )
    
    # Execute the query in a transactional scope
    with pool.begin() as conn:
        conn.execute(sql, data)

def predict_future_demand(model, future_df, features, scaler_X, scaler_y, 
                          time_steps=5, bias_correction_func=None, log_transformed=False):
    """
    Predict future demand recursively for each day in future_df.
    """
    logger.info("Predicting future demand for each day")
    
    predictions = []
    
    for product in future_df['Product Name'].unique():
        # Extract and sort the future‐plus‐historical window for this product
        prod_mask = future_df['Product Name'] == product
        prod_future = future_df[prod_mask].sort_values('Date').reset_index(drop=True)
        
        # Scale the feature matrix once
        X_all = scaler_X.transform(prod_future[features].values)
        
        # We’ll maintain a rolling window of the last `time_steps` rows
        window = list(X_all[:time_steps])
        dates = prod_future['Date'].tolist()
        
        # Now forecast each future date after the initial window
        for idx in range(time_steps, len(dates)):
            X_seq = np.array([window[-time_steps:]])                     # shape (1, time_steps, n_features)
            y_raw = model.predict(X_seq)                                 # raw model output
            
            # inverse‐scale + bias‐correction
            if bias_correction_func:
                y_pred = bias_correction_func(y_raw)
            else:
                y_pred = scaler_y.inverse_transform(y_raw)
                if log_transformed:
                    y_pred = np.expm1(y_pred)
            y_pred = np.maximum(0, y_pred).flatten()[0]
            
            # record it
            predictions.append({
                'sale_date': dates[idx],
                'product_name': product,
                'total_quantity': float(y_pred)
            })
            
            # if you want fully recursive: append the predicted value back into your window
            # so that lag features for the next day “see” this predicted quantity:
            # — here we replace the true lagged‐quantity in the feature vector with our prediction:
            window.append(window[-1].copy())
            # find index of your target in the features list, e.g. if 'lag_1d' or similar:
            # locate the position of 'lag_1d' in `features` and overwrite it
            # (you’ll need to recompute other lag features if you want full rigor)
            target_idx = features.index('lag_1d')  # adjust if your naming differs
            window[-1][target_idx] = y_pred
            
        logger.info(f" → {product}: {len(dates) - time_steps} days predicted")
    
    return pd.DataFrame(predictions)

def correct_prediction_bias(model, X_val, y_val, scaler_y, log_transformed=False):
    """
    Correct systematic bias in predictions
    
    Parameters:
    -----------
    model : Keras model
        Trained model
    X_val, y_val : np.array
        Validation data for calculating bias
    scaler_y : Scaler
        Target scaler for inverse transformation
    log_transformed : bool
        Whether target was log-transformed
        
    Returns:
    --------
    Function to apply correction to predictions
    """
    logger.info("Calculating bias correction factors")
    
    # Make predictions on validation set
    y_pred = model.predict(X_val)
    
    # Inverse transform
    y_pred_real = scaler_y.inverse_transform(y_pred)
    y_val_real = scaler_y.inverse_transform(y_val)
    
    # If log-transformed, convert back
    if log_transformed:
        y_pred_real = np.expm1(y_pred_real)
        y_val_real = np.expm1(y_val_real)
    
    # Calculate errors
    errors = y_pred_real - y_val_real
    
    # Calculate systematic bias (median error is more robust to outliers)
    bias_correction = np.median(errors)
    
    logger.info(f"Systematic bias detected: {bias_correction:.4f}")
    
    # Function to apply correction to new predictions
    def correct_predictions(predictions):
        """
        Apply bias correction to model predictions
        
        Parameters:
        -----------
        predictions : np.array
            Raw model predictions
            
        Returns:
        --------
        Bias-corrected predictions
        """
        # Inverse transform
        predictions_real = scaler_y.inverse_transform(predictions)
        
        # If log-transformed, convert back
        if log_transformed:
            predictions_real = np.expm1(predictions_real)
        
        # Apply bias correction
        corrected_predictions = predictions_real - bias_correction
        
        # Ensure non-negative predictions
        corrected_predictions = np.maximum(0, corrected_predictions)
        
        return corrected_predictions
    
    return correct_predictions





def load_preprocessing_objects(input_dir='.'):
    """
    Load preprocessing objects for inference
    
    Parameters:
    -----------
    input_dir : str
        Directory containing saved objects
        
    Returns:
    --------
    Tuple of (scaler_X, scaler_y, label_encoder, log_transformed)
    """
    logger.info("Loading preprocessing objects")
    
    # Load scalers
    with open(os.path.join(input_dir, 'scaler_X.pkl'), 'rb') as f:
        scaler_X = pickle.load(f)
    
    with open(os.path.join(input_dir, 'scaler_y.pkl'), 'rb') as f:
        scaler_y = pickle.load(f)
    
    # Load label encoder
    with open(os.path.join(input_dir, 'label_encoder.pkl'), 'rb') as f:
        label_encoder = pickle.load(f)
    
    # Load transformation flag
    with open(os.path.join(input_dir, 'transform_info.pkl'), 'rb') as f:
        transform_info = pickle.load(f)
        log_transformed = transform_info.get('log_transformed', False)
    
    logger.info(f"Preprocessing objects loaded from {input_dir}")
    
    return scaler_X, scaler_y, label_encoder, log_transformed



def generate_future_features(df, days_to_predict=7, features=None):
    """
    Generate features for future dates for prediction
    
    Parameters:
    -----------
    df : DataFrame
        Historical data with features
    days_to_predict : int
        Number of days to predict into the future
    features : list, optional
        List of features to use (if None, use all except target)
        
    Returns:
    --------
    DataFrame with features for future dates
    """
    logger.info(f"Generating features for {days_to_predict} days ahead")
    
    # Get the last date in the dataset
    last_date = df['Date'].max()
    
    # Create future dates
    future_dates = [last_date + timedelta(days=i+1) for i in range(days_to_predict)]
    
    # Create a DataFrame for each product
    all_future_data = []
    
    for product in df['Product Name'].unique():
        # Get product-specific data
        product_data = df[df['Product Name'] == product].sort_values('Date')
        
        if len(product_data) < 5:  # Skip products with very little data
            logger.warning(f"Skipping product '{product}' with insufficient data")
            continue
        
        # Create future data for this product
        product_future = []
        
        for future_date in future_dates:
            # Start with last row of historical data
            new_row = product_data.iloc[-1].copy()
            
            # Update date
            new_row['Date'] = future_date
            
            # Update date-derived features
            new_row['year'] = future_date.year
            new_row['month'] = future_date.month
            new_row['day'] = future_date.day
            new_row['dayofweek'] = future_date.dayofweek
            new_row['dayofyear'] = future_date.dayofyear
            new_row['quarter'] = future_date.quarter
            
            # Update weekend indicator
            new_row['is_weekend'] = int(future_date.dayofweek >= 5)
            
            # Update month start/end indicators
            new_row['is_month_start'] = int(future_date.day == 1)
            new_row['is_month_end'] = int(future_date.day == pd.Timestamp(future_date.year, future_date.month, 1).days_in_month)
            
            # Add row to future data
            product_future.append(new_row)
        
        # Convert to DataFrame
        product_future_df = pd.DataFrame(product_future)
        
        # Update lag features based on historical data
        for lag in [1, 2, 3, 7, 14]:
            if f'lag_{lag}d' in product_future_df.columns:
                for i, row in enumerate(product_future_df.iterrows()):
                    idx = i + len(product_data) - lag
                    if idx >= 0 and idx < len(product_data):
                        product_future_df.loc[row[0], f'lag_{lag}d'] = product_data.iloc[idx]['Total Quantity']
                    else:
                        product_future_df.loc[row[0], f'lag_{lag}d'] = 0
        
        # Add to overall future data
        all_future_data.append(product_future_df)
    
    # Combine all products
    if not all_future_data:
        raise ValueError("No future data could be generated. Check your input data.")
    
    future_df = pd.concat(all_future_data, ignore_index=True)
    
    # Select only required features
    # Select only required features
    if features is not None:
        # Ensure we keep necessary columns
        required_cols = ['Date', 'Product Name', 'product_encoded']
        features_to_keep = list(set(features + required_cols))
        future_df = future_df[features_to_keep]
    
    logger.info(f"Generated features for {len(future_df)} future data points")
    
    return future_df


def create_features(df):
    """
    Create rich feature set for time series forecasting
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame
        
    Returns:
    --------
    DataFrame with additional features
    """
    logger.info("Creating features for time series forecasting")
    
    # Create a copy to avoid modifying the original
    feature_df = df.copy()
    
    # Extract date features
    feature_df['year'] = feature_df['Date'].dt.year
    feature_df['month'] = feature_df['Date'].dt.month
    feature_df['day'] = feature_df['Date'].dt.day
    feature_df['dayofweek'] = feature_df['Date'].dt.dayofweek
    feature_df['dayofyear'] = feature_df['Date'].dt.dayofyear
    feature_df['quarter'] = feature_df['Date'].dt.quarter
    feature_df['is_weekend'] = feature_df['dayofweek'].isin([5, 6]).astype(int)
    feature_df['is_month_start'] = feature_df['Date'].dt.is_month_start.astype(int)
    feature_df['is_month_end'] = feature_df['Date'].dt.is_month_end.astype(int)
    
    # Add holiday indicators (example for US holidays)
    try:
        from pandas.tseries.holiday import USFederalHolidayCalendar
        cal = USFederalHolidayCalendar()
        holidays = cal.holidays(start=feature_df['Date'].min(), end=feature_df['Date'].max())
        feature_df['is_holiday'] = feature_df['Date'].isin(holidays).astype(int)
    except:
        logger.warning("Could not add holiday indicators, continuing without them")
        feature_df['is_holiday'] = 0
    
    # Add rolling statistics with robust measures
    # Process by product to avoid leakage
    for product in feature_df['Product Name'].unique():
        product_mask = feature_df['Product Name'] == product
        product_df = feature_df.loc[product_mask].sort_values('Date')
        
        if len(product_df) > 1:  # Only process if we have enough data
            # Regular rolling statistics (7-day window)
            feature_df.loc[product_mask, 'rolling_mean_7d'] = product_df['Total Quantity'].rolling(
                window=7, min_periods=1).mean()
            
            feature_df.loc[product_mask, 'rolling_std_7d'] = product_df['Total Quantity'].rolling(
                window=7, min_periods=1).std().fillna(0)
            
            # Robust rolling statistics (7-day window)
            feature_df.loc[product_mask, 'rolling_median_7d'] = product_df['Total Quantity'].rolling(
                window=7, min_periods=1).median()
            
            # Calculate rolling min/max
            feature_df.loc[product_mask, 'rolling_min_7d'] = product_df['Total Quantity'].rolling(
                window=7, min_periods=1).min()
            
            feature_df.loc[product_mask, 'rolling_max_7d'] = product_df['Total Quantity'].rolling(
                window=7, min_periods=1).max()
            
            # Create lag features (previous values)
            for lag in [1, 2, 3, 7, 14]:  # Include weekly and bi-weekly lags
                feature_df.loc[product_mask, f'lag_{lag}d'] = product_df['Total Quantity'].shift(lag)
            
            # Add trend features (differences)
            feature_df.loc[product_mask, 'diff_1d'] = product_df['Total Quantity'].diff(1)
            feature_df.loc[product_mask, 'diff_7d'] = product_df['Total Quantity'].diff(7)
            
            # Exponential weighted moving average (more weight to recent observations)
            feature_df.loc[product_mask, 'ewm_alpha_0.3'] = product_df['Total Quantity'].ewm(alpha=0.3).mean()
            feature_df.loc[product_mask, 'ewm_alpha_0.7'] = product_df['Total Quantity'].ewm(alpha=0.7).mean()
            
            # Add features for detecting seasonality
            if len(product_df) >= 14:  # Need at least 2 weeks of data
                # Week-over-week percentage change
                feature_df.loc[product_mask, 'wow_pct_change'] = product_df['Total Quantity'].pct_change(7)
                
            if len(product_df) >= 60:  # Need around 2 months of data
                # Month-over-month percentage change (approximating with 30 days)
                feature_df.loc[product_mask, 'mom_pct_change'] = product_df['Total Quantity'].pct_change(30)
        else:
            # For products with minimal data, use zeros for these features
            rolling_cols = ['rolling_mean_7d', 'rolling_std_7d', 'rolling_median_7d', 
                           'rolling_min_7d', 'rolling_max_7d']
            lag_cols = [f'lag_{lag}d' for lag in [1, 2, 3, 7, 14]]
            diff_cols = ['diff_1d', 'diff_7d']
            ewm_cols = ['ewm_alpha_0.3', 'ewm_alpha_0.7']
            pct_change_cols = ['wow_pct_change', 'mom_pct_change']
            
            all_cols = rolling_cols + lag_cols + diff_cols + ewm_cols + pct_change_cols
            for col in all_cols:
                feature_df.loc[product_mask, col] = 0
    
    # Fill NaN values with appropriate strategies
    # For lag and diff features, use forward fill then backward fill
    lag_cols = [col for col in feature_df.columns if col.startswith('lag_') or col.startswith('diff_') or col.startswith('rolling_') or col.startswith('ewm_') or col.endswith('pct_change')]
    
    # Fill NAs by product to avoid leakage
    for product in feature_df['Product Name'].unique():
        product_mask = feature_df['Product Name'] == product
        
        # First try forward fill (use previous values)
        feature_df.loc[product_mask, lag_cols] = feature_df.loc[product_mask, lag_cols].fillna(method='ffill')
        
        # Then backward fill (use future values if still have NAs)
        feature_df.loc[product_mask, lag_cols] = feature_df.loc[product_mask, lag_cols].fillna(method='bfill')
        
        # Finally use zeros for any remaining NAs
        feature_df.loc[product_mask, lag_cols] = feature_df.loc[product_mask, lag_cols].fillna(0)
    
    # Encode product names
    label_encoder = LabelEncoder()
    feature_df['product_encoded'] = label_encoder.fit_transform(feature_df['Product Name'])
    
    # Create product embeddings using groupby statistics
    product_stats = feature_df.groupby('Product Name')['Total Quantity'].agg([
        ('product_mean', 'mean'),
        ('product_median', 'median'),
        ('product_std', 'std'),
        ('product_min', 'min'),
        ('product_max', 'max')
    ]).reset_index()
    
    # Merge stats back to main dataframe
    feature_df = feature_df.merge(product_stats, on='Product Name', how='left')
    
    # Fill any remaining NaN values
    feature_df = feature_df.fillna(0)
    
    # Log feature creation summary
    original_cols = set(df.columns)
    new_cols = set(feature_df.columns) - original_cols
    logger.info(f"Created {len(new_cols)} new features: {sorted(new_cols)}")
    
    return feature_df, label_encoder

def apply_log_transform(df, target_col='Total Quantity'):
    """
    Apply log transformation to handle skewed data and heteroscedasticity
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame
    target_col : str
        Name of target column to transform
        
    Returns:
    --------
    DataFrame with transformed target and flag indicating transformation
    """
    # Check if log transform would be beneficial
    target_vals = df[target_col].values
    
    # Skip transform if we have zeros or negative values
    if np.any(target_vals <= 0):
        logger.info("Skipping log transform as data contains zeros or negative values")
        df['log_transformed'] = False
        return df, False
    
    # Check skewness
    skewness = stats.skew(target_vals)
    
    if abs(skewness) > 1.0:  # Rule of thumb for significant skewness
        logger.info(f"Applying log transform to target '{target_col}' (skewness: {skewness:.2f})")
        
        # Create new column with log transform
        df[f'{target_col}_log'] = np.log1p(df[target_col])
        df['log_transformed'] = True
        
        return df, True
    else:
        logger.info(f"Log transform not needed (skewness: {skewness:.2f})")
        df['log_transformed'] = False
        return df, False


def apply_rounding_strategy(df,
                            qty_col: str = 'total_quantity',
                            safety_stock: int = 0) -> pd.DataFrame:
    """
    Overwrite `qty_col` with its nearest-integer value,
    then optionally add a fixed safety stock.
    """
    # nearest-integer rounding (>= .5 up, < .5 down)
    df[qty_col] = (df[qty_col] / 10).round().astype(int)

    # if you’d rather always round .5 up, uncomment:
    # df[qty_col] = (df[qty_col] + 0.5).astype(int)

    # add buffer if needed
    if safety_stock:
        df[qty_col] += safety_stock
    
    fluctuations = np.random.choice([-2, -1, 0], size=len(df), p=[0.15, 0.25, 0.60])
    df[qty_col] = (df[qty_col] + fluctuations).clip(lower=1)

    return df
