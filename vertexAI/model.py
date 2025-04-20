import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import losses
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from datetime import datetime, timedelta
import pickle
import random
from scipy import stats
from utils import analyze_data_distribution, plot_distribution_bias, get_latest_data_from_cloud_sql, send_email, get_connection, upload_to_gcs
from sklearn.metrics import mean_squared_error, mean_absolute_error
import shap
import optuna
from dotenv import load_dotenv


load_dotenv()

new_product_flag = os.getenv("FLAG")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('debiased_model')

class CustomError(Exception):
    """Base class for custom exceptions"""
    pass

class DataQualityError(CustomError):
    """Exception raised for data quality issues"""
    pass

def detect_data_quality_issues(df):
    """
    Detect data quality issues before proceeding
    """
    issues = []
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        issues.append(f"Found {missing_values.sum()} missing values in dataset")
    
    # Check for negative quantities
    negative_qty = (df['Total Quantity'] < 0).sum()
    if negative_qty > 0:
        issues.append(f"Found {negative_qty} negative quantity values")
    
    # Check for duplicate entries
    duplicates = df.duplicated(subset=['Date', 'Product Name']).sum()
    if duplicates > 0:
        issues.append(f"Found {duplicates} duplicate entries")
    
    # Check for products with very few samples
    product_counts = df['Product Name'].value_counts()
    min_samples_threshold = 7  # Absolute minimum to create sequences
    low_sample_products = product_counts[product_counts < min_samples_threshold]
    if len(low_sample_products) > 0:
        issues.append(f"Found {len(low_sample_products)} products with fewer than {min_samples_threshold} samples")
    
    # Check for large date gaps
    time_gaps = []
    for product in df['Product Name'].unique():
        product_df = df[df['Product Name'] == product].sort_values('Date')
        if len(product_df) > 1:
            date_diffs = product_df['Date'].diff().dt.days.iloc[1:]
            max_gap = date_diffs.max()
            if max_gap > 30:  # Flag gaps larger than 30 days
                time_gaps.append((product, max_gap))
    
    if time_gaps:
        issues.append(f"Found large time gaps (>30 days) in {len(time_gaps)} products")
    
    # Print summary of issues
    if issues:
        for issue in issues:
            logger.warning(issue)
    else:
        logger.info("No major data quality issues detected")
    
    return issues

def clean_data(df):
    """
    Clean the dataset to address basic quality issues
    """
    logger.info("Cleaning dataset")
    cleaned_df = df.copy()
    
    # Remove duplicate entries
    initial_count = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates(subset=['Date', 'Product Name'])
    duplicates_removed = initial_count - len(cleaned_df)
    if duplicates_removed > 0:
        logger.info(f"Removed {duplicates_removed} duplicate entries")
    
    # Ensure quantity is positive
    neg_qty_count = (cleaned_df['Total Quantity'] < 0).sum()
    if neg_qty_count > 0:
        logger.info(f"Converting {neg_qty_count} negative quantities to absolute values")
        cleaned_df['Total Quantity'] = cleaned_df['Total Quantity'].abs()
    
    # Handle missing values if any
    if cleaned_df.isnull().sum().sum() > 0:
        logger.info("Filling missing values")
        # For numeric columns, use median by product
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            cleaned_df[col] = cleaned_df.groupby('Product Name')[col].transform(
                lambda x: x.fillna(x.median() if not x.median() != x.median() else 0))
        
        # For remaining NA values, use 0
        cleaned_df = cleaned_df.fillna(0)
    
    return cleaned_df

def handle_outliers(df, method='winsorize', threshold=3):
    """
    Detect and handle outliers using robust statistics
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame
    method : str
        Method to handle outliers: 'winsorize', 'remove', or 'cap'
    threshold : float
        Threshold for outlier detection (in IQR units)
        
    Returns:
    --------
    DataFrame with outliers handled
    """
    logger.info(f"Handling outliers using {method} method")
    
    result_df = df.copy()
    outlier_stats = {}
    
    # Handle outliers by product
    for product in df['Product Name'].unique():
        product_mask = result_df['Product Name'] == product
        product_data = result_df[product_mask]
        
        if len(product_data) <= 10:  # Skip products with very few data points
            outlier_stats[product] = {'count': 0, 'percentage': 0}
            continue
        
        # Calculate median and IQR for each product
        median = product_data['Total Quantity'].median()
        q1 = product_data['Total Quantity'].quantile(0.25)
        q3 = product_data['Total Quantity'].quantile(0.75)
        iqr = q3 - q1
        
        # Define bounds
        lower_bound = max(0, q1 - threshold * iqr)  # Ensure non-negative
        upper_bound = q3 + threshold * iqr
        
        # Find outliers
        outlier_mask = (product_data['Total Quantity'] < lower_bound) | (product_data['Total Quantity'] > upper_bound)
        outliers = product_data[outlier_mask]
        
        outlier_count = len(outliers)
        outlier_pct = outlier_count / len(product_data) if len(product_data) > 0 else 0
        outlier_stats[product] = {'count': outlier_count, 'percentage': outlier_pct}
        
        if outlier_count > 0:
            logger.info(f"Found {outlier_count} outliers ({outlier_pct:.2%}) in product '{product}'")
            
            if method == 'winsorize':
                # Replace outliers with bounds (winsorization)
                for idx in outliers.index:
                    if result_df.loc[idx, 'Total Quantity'] < lower_bound:
                        result_df.loc[idx, 'Total Quantity'] = lower_bound
                    elif result_df.loc[idx, 'Total Quantity'] > upper_bound:
                        result_df.loc[idx, 'Total Quantity'] = upper_bound
            
            elif method == 'remove':
                # Remove outliers (not recommended for time series)
                logger.warning("Removing outliers from time series data may create gaps")
                result_df = result_df.drop(outliers.index)
            
            elif method == 'cap':
                # Cap at percentile values
                result_df.loc[product_mask & (result_df['Total Quantity'] < lower_bound), 'Total Quantity'] = lower_bound
                result_df.loc[product_mask & (result_df['Total Quantity'] > upper_bound), 'Total Quantity'] = upper_bound
    
    # Log summary statistics
    total_outliers = sum(stats['count'] for stats in outlier_stats.values())
    avg_outlier_pct = sum(stats['percentage'] for stats in outlier_stats.values()) / len(outlier_stats) if outlier_stats else 0
    logger.info(f"Total outliers handled: {total_outliers} ({avg_outlier_pct:.2%} average)")
    
    return result_df

def handle_data_imbalance(df, min_samples=30):
    """
    Handle data imbalance through smart augmentation
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame
    min_samples : int
        Minimum number of samples desired per product
        
    Returns:
    --------
    DataFrame with balanced product representation
    """
    logger.info("Handling data imbalance")
    
    product_counts = df['Product Name'].value_counts()
    products_to_augment = product_counts[product_counts < min_samples].index
    
    if len(products_to_augment) == 0:
        logger.info("No products require augmentation")
        return df
    
    logger.info(f"Augmenting {len(products_to_augment)} products with fewer than {min_samples} samples")
    
    augmented_df = df.copy()
    
    for product in products_to_augment:
        product_df = df[df['Product Name'] == product]
        current_count = len(product_df)
        
        if current_count < 5:
            logger.warning(f"Product '{product}' has only {current_count} samples, augmentation may not be reliable")
        
        if current_count < min_samples:
            num_to_generate = min_samples - current_count
            logger.info(f"Generating {num_to_generate} synthetic samples for product '{product}'")
            
            # Generate synthetic samples
            synthetic_samples = []
            
            for _ in range(num_to_generate):
                # Smart sampling: sample from nearby dates to preserve patterns
                if current_count > 1:
                    # Sample consecutive points when possible
                    sample_idx = np.random.randint(0, current_count - 1)
                    base_sample = product_df.iloc[sample_idx].copy()
                    next_sample = product_df.iloc[sample_idx + 1].copy()
                    
                    # Linear interpolation between points
                    alpha = np.random.random()  # Interpolation factor
                    new_sample = base_sample.copy()
                    
                    # Interpolate quantity (with some noise)
                    base_qty = base_sample['Total Quantity']
                    next_qty = next_sample['Total Quantity']
                    interp_qty = base_qty + alpha * (next_qty - base_qty)
                    
                    # Add controlled noise (±5-15%)
                    noise_factor = np.random.uniform(0.05, 0.15)
                    noise_direction = 1 if np.random.random() > 0.5 else -1
                    new_qty = max(1, int(interp_qty * (1 + noise_direction * noise_factor)))
                    new_sample['Total Quantity'] = new_qty
                    
                    # Interpolate date
                    days_diff = (next_sample['Date'] - base_sample['Date']).days
                    new_days = int(alpha * days_diff)
                    new_date = base_sample['Date'] + pd.Timedelta(days=new_days)
                    new_sample['Date'] = new_date
                    
                    # Update date-derived features if they exist
                    date_features = ['year', 'month', 'day', 'dayofweek', 'dayofyear', 'quarter']
                    for feature in date_features:
                        if feature in new_sample:
                            if feature == 'year':
                                new_sample[feature] = new_date.year
                            elif feature == 'month':
                                new_sample[feature] = new_date.month
                            elif feature == 'day':
                                new_sample[feature] = new_date.day
                            elif feature == 'dayofweek':
                                new_sample[feature] = new_date.dayofweek
                            elif feature == 'dayofyear':
                                new_sample[feature] = new_date.dayofyear
                            elif feature == 'quarter':
                                new_sample[feature] = new_date.quarter
                else:
                    # If only one sample, duplicate with noise
                    base_sample = product_df.iloc[0].copy()
                    new_sample = base_sample.copy()
                    
                    # Add noise to quantity (±10-20%)
                    noise_factor = np.random.uniform(0.1, 0.2)
                    noise_direction = 1 if np.random.random() > 0.5 else -1
                    current_qty = new_sample['Total Quantity']
                    new_qty = max(1, int(current_qty * (1 + noise_direction * noise_factor)))
                    new_sample['Total Quantity'] = new_qty
                    
                    # Shift date by random amount (±1-7 days)
                    days_shift = np.random.randint(-7, 8)  # -7 to +7 days
                    new_date = new_sample['Date'] + pd.Timedelta(days=days_shift)
                    new_sample['Date'] = new_date
                    
                    # Update date-derived features if they exist
                    date_features = ['year', 'month', 'day', 'dayofweek', 'dayofyear', 'quarter']
                    for feature in date_features:
                        if feature in new_sample:
                            if feature == 'year':
                                new_sample[feature] = new_date.year
                            elif feature == 'month':
                                new_sample[feature] = new_date.month
                            elif feature == 'day':
                                new_sample[feature] = new_date.day
                            elif feature == 'dayofweek':
                                new_sample[feature] = new_date.dayofweek
                            elif feature == 'dayofyear':
                                new_sample[feature] = new_date.dayofyear
                            elif feature == 'quarter':
                                new_sample[feature] = new_date.quarter
                
                synthetic_samples.append(new_sample)
            
            # Add synthetic samples to augmented dataframe
            if synthetic_samples:
                synthetic_df = pd.DataFrame(synthetic_samples)
                augmented_df = pd.concat([augmented_df, synthetic_df], ignore_index=True)
    
    # Sort by date and product
    augmented_df = augmented_df.sort_values(['Product Name', 'Date'])
    
    # Log augmentation results
    original_counts = df['Product Name'].value_counts()
    new_counts = augmented_df['Product Name'].value_counts()
    
    logger.info(f"Original dataset: {len(df)} samples across {df['Product Name'].nunique()} products")
    logger.info(f"Augmented dataset: {len(augmented_df)} samples across {augmented_df['Product Name'].nunique()} products")
    
    return augmented_df

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

class FairnessAwareCallback(tf.keras.callbacks.Callback):
    """
    Custom callback to monitor and balance model performance across different products
    """
    def __init__(self, X_val, y_val, product_indices, product_names, scaler_y, log_transformed=False, patience=5):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.product_indices = product_indices
        self.product_names = product_names
        self.scaler_y = scaler_y
        self.log_transformed = log_transformed
        self.patience = patience
        self.best_disparities = float('inf')
        self.wait = 0
        self.best_weights = None
        self.product_metrics_history = []
        
    def on_epoch_end(self, epoch, logs=None):
        # Get predictions
        y_pred = self.model.predict(self.X_val, verbose=0)
        
        # Calculate metrics by product
        product_metrics = {}
        product_idx_to_name = dict(zip(self.product_indices, self.product_names))
        
        for product_idx in self.product_indices:
            product_name = product_idx_to_name.get(product_idx, f"Product_{product_idx}")
            
            # Get indices for this product in the validation set
            product_mask = self.X_val[:, 0, 6] == product_idx  # Assuming product_encoded is at index 6
            
            if np.sum(product_mask) > 0:
                # Get predictions and actual values for this product
                product_y_pred = y_pred[product_mask]
                product_y_val = self.y_val[product_mask]
                
                # Inverse transform if needed
                product_y_pred_real = self.scaler_y.inverse_transform(product_y_pred)
                product_y_val_real = self.scaler_y.inverse_transform(product_y_val)
                
                # If log-transformed, convert back
                if self.log_transformed:
                    product_y_pred_real = np.expm1(product_y_pred_real)
                    product_y_val_real = np.expm1(product_y_val_real)
                
                # Calculate metrics
                mse = np.mean((product_y_pred_real - product_y_val_real) ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(product_y_pred_real - product_y_val_real))
                
                # Calculate MAPE for non-zero values to avoid division by zero
                non_zero_mask = product_y_val_real > 0
                if np.sum(non_zero_mask) > 0:
                    mape = np.mean(np.abs((product_y_val_real[non_zero_mask] - product_y_pred_real[non_zero_mask]) / 
                                          product_y_val_real[non_zero_mask])) * 100
                else:
                    mape = np.nan
                
                product_metrics[product_name] = {
                    'rmse': rmse,
                    'mae': mae,
                    'mape': mape,
                    'sample_count': np.sum(product_mask)
                }
        
        # Store metrics history
        self.product_metrics_history.append({
            'epoch': epoch,
            'metrics': product_metrics
        })
        
        # Calculate disparity
        if product_metrics:
            rmse_values = [m['rmse'] for m in product_metrics.values() if not np.isnan(m['rmse'])]
            
            if rmse_values:
                max_rmse = max(rmse_values)
                min_rmse = min(rmse_values)
                disparity = max_rmse / (min_rmse + 1e-6)  # Avoid division by zero
                
                # Find worst and best products
                worst_product = None
                best_product = None
                max_val = -float('inf')
                min_val = float('inf')
                
                if disparity < self.best_disparities:
                    self.best_disparities = disparity
                    self.wait = 0
                    # Save weights
                    self.best_weights = self.model.get_weights()
                    logger.info(f"Epoch {epoch+1}: Improved fairness disparity to {disparity:.4f}")
                else:
                    self.wait += 1
                    if self.wait >= self.patience:
                        # Restore best weights
                        if self.best_weights is not None:
                            self.model.set_weights(self.best_weights)
                            logger.info(f"Restoring best weights from epoch {epoch+1-self.wait}")
                        self.model.stop_training = True
                        logger.info(f"Early stopping due to fairness disparity not improving for {self.patience} epochs")
                
                # Log information
                if logs is not None:
                    logs['fairness_disparity'] = disparity
                logger.info(f"Epoch {epoch+1}: Fairness disparity = {disparity:.4f}")

def create_weighted_sequences(df, features, target_col='Total Quantity', log_transformed=False,
                             time_steps=5, test_size=0.2):
    """
    Create sequences for LSTM with sample weights based on product frequency
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame with features
    features : list
        List of feature column names
    target_col : str
        Name of target column
    log_transformed : bool
        Whether target has been log-transformed
    time_steps : int
        Number of time steps for LSTM sequences
    test_size : float
        Proportion of data to use for testing
        
    Returns:
    --------
    Tuple containing:
    - X_train, X_val, X_test: Training, validation, and test features
    - y_train, y_val, y_test: Training, validation, and test targets
    - weights_train, weights_val, weights_test: Sample weights
    - scaler_X, scaler_y: Feature and target scalers
    - product_indices, product_names: Product ID mappings
    """
    logger.info(f"Creating weighted sequences with {time_steps} time steps")
    
    # Use target_col_log if log transformed
    if log_transformed and f'{target_col}_log' in df.columns:
        actual_target = f'{target_col}_log'
        logger.info(f"Using log-transformed target: {actual_target}")
    else:
        actual_target = target_col
    
    # Initialize robust scalers
    scaler_X = RobustScaler()
    scaler_y = RobustScaler()
    
    # Scale features and target
    X_values = df[features].values
    X_scaled = scaler_X.fit_transform(X_values)
    
    y_values = df[[actual_target]].values
    y_scaled = scaler_y.fit_transform(y_values)
    
    # Function to create sequences
    def create_sequences(X, y, time_steps=time_steps):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            Xs.append(X[i:(i + time_steps)])
            ys.append(y[i + time_steps])
        return np.array(Xs), np.array(ys)
    
    # Initialize containers for sequences and weights
    all_X_seq = []
    all_y_seq = []
    all_weights = []
    all_product_indices = []
    sequence_products = []  # Track which product each sequence belongs to
    
    # Get unique products
    products = df['Product Name'].unique()
    product_indices = []
    product_names = []
    
    for product in products:
        product_idx = df[df['Product Name'] == product]['product_encoded'].iloc[0]
        product_indices.append(product_idx)
        product_names.append(product)
    
    # Calculate inverse frequency weights
    product_counts = df['Product Name'].value_counts()
    max_count = product_counts.max()
    product_weights = {product: max_count / count for product, count in product_counts.items()}
    
    # Scale weights to avoid extreme values
    max_weight = max(product_weights.values())
    if max_weight > 10:
        scale_factor = 10 / max_weight
        product_weights = {p: w * scale_factor for p, w in product_weights.items()}
    
    logger.info(f"Product weights range: {min(product_weights.values()):.2f} to {max(product_weights.values()):.2f}")
    
    # Create sequences and weights by product
    for product, product_idx in zip(products, product_indices):
        product_df_indices = df[df['Product Name'] == product].index
        product_X = X_scaled[product_df_indices]
        product_y = y_scaled[product_df_indices]
        
        if len(product_X) > time_steps:
            X_seq, y_seq = create_sequences(product_X, product_y, time_steps)
            
            # Skip if no sequences were created
            if len(X_seq) == 0:
                continue
            
            # Assign weights based on product frequency
            weights = np.ones(len(X_seq)) * product_weights[product]
            
            all_X_seq.append(X_seq)
            all_y_seq.append(y_seq)
            all_weights.append(weights)
            
            # Track which product each sequence belongs to
            sequence_products.extend([product_idx] * len(X_seq))
    
    # Combine all sequences and weights
    if not all_X_seq:
        raise DataQualityError("No sequences could be created. Check your data and time_steps parameter.")
    
    X_seq_combined = np.vstack(all_X_seq)
    y_seq_combined = np.vstack(all_y_seq)
    weights_combined = np.concatenate(all_weights)
    
    # Split to train, validation and test sets while preserving product distribution
    X_train_val, X_test, y_train_val, y_test, weights_train_val, weights_test, products_train_val, products_test = train_test_split(
        X_seq_combined, y_seq_combined, weights_combined, sequence_products,
        test_size=test_size, random_state=42, stratify=sequence_products)
    
    X_train, X_val, y_train, y_val, weights_train, weights_val, products_train, products_val = train_test_split(
        X_train_val, y_train_val, weights_train_val, products_train_val,
        test_size=0.2, random_state=42, stratify=products_train_val)
    
    logger.info(f"Created {len(X_train)} training, {len(X_val)} validation, and {len(X_test)} test sequences")
    
    return (X_train, X_val, X_test, y_train, y_val, y_test, 
            weights_train, weights_val, weights_test,
            scaler_X, scaler_y, product_indices, product_names)

def build_model(input_shape, hyperparams=None):
    """
    Build LSTM model for demand forecasting with bias mitigation
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data (time_steps, features)
    hyperparams : dict, optional
        Hyperparameters for model architecture
        
    Returns:
    --------
    Compiled Keras model
    """
    if hyperparams is None:
        # Default hyperparameters
        hyperparams = {
            'units_1': 64,
            'dropout_1': 0.2,
            'activation_1': 'tanh',
            'units_2': 32,
            'dropout_2': 0.2,
            'activation_2': 'tanh',
            'dense_units': 16,
            'dense_activation': 'relu',
            'learning_rate': 0.001,
            'optimizer': 'adam'
        }
    
    logger.info(f"Building LSTM model with hyperparameters: {hyperparams}")
    
    model = Sequential()
    
    # First LSTM layer
    model.add(LSTM(units=hyperparams['units_1'],
                   activation=hyperparams['activation_1'],
                   return_sequences=True,
                   input_shape=input_shape))
    model.add(Dropout(hyperparams['dropout_1']))
    
    # Second LSTM layer
    model.add(LSTM(units=hyperparams['units_2'],
                   activation=hyperparams['activation_2']))
    model.add(Dropout(hyperparams['dropout_2']))
    
    # Dense hidden layer
    model.add(Dense(units=hyperparams['dense_units'],
                    activation=hyperparams['dense_activation']))
    
    # Output layer
    model.add(Dense(1))
    
    # Configure optimizer
    if hyperparams['optimizer'] == 'adam':
        optimizer = Adam(learning_rate=hyperparams['learning_rate'])
    else:
        optimizer = RMSprop(learning_rate=hyperparams['learning_rate'])
    
    # Compile model
    model.compile(optimizer=optimizer, loss='mse')
    
    return model



def objective(trial, X_train, y_train, X_val, y_val, input_shape, weights_train, weights_val):
    """
    Objective function for Optuna hyperparameter tuning
    
    Parameters:
    -----------
    trial : optuna.trial.Trial
        Current trial object
    X_train, y_train : np.array
        Training data
    X_val, y_val : np.array
        Validation data
    input_shape : tuple
        Shape of input data (time_steps, features)
    weights_train, weights_val : np.array
        Sample weights
        
    Returns:
    --------
    Validation loss value
    """
    # Define hyperparameters to tune
    hyperparams = {
        'units_1': trial.suggest_int('units_1', 32, 128, step=16),
        'dropout_1': trial.suggest_float('dropout_1', 0.1, 0.5, step=0.1),
        'activation_1': trial.suggest_categorical('activation_1', ['tanh', 'relu']),
        'units_2': trial.suggest_int('units_2', 16, 64, step=8),
        'dropout_2': trial.suggest_float('dropout_2', 0.1, 0.5, step=0.1),
        'activation_2': trial.suggest_categorical('activation_2', ['tanh', 'relu']),
        'dense_units': trial.suggest_int('dense_units', 8, 32, step=8),
        'dense_activation': trial.suggest_categorical('dense_activation', ['relu', 'tanh']),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'optimizer': trial.suggest_categorical('optimizer', ['adam', 'rmsprop'])
    }
    
    # Create model with trial hyperparameters
    model = Sequential()
    
    # First LSTM layer
    model.add(LSTM(units=hyperparams['units_1'],
                  activation=hyperparams['activation_1'],
                  return_sequences=True,
                  input_shape=input_shape))
    model.add(Dropout(hyperparams['dropout_1']))
    
    # Second LSTM layer
    model.add(LSTM(units=hyperparams['units_2'],
                  activation=hyperparams['activation_2']))
    model.add(Dropout(hyperparams['dropout_2']))
    
    # Dense hidden layer
    model.add(Dense(units=hyperparams['dense_units'],
                   activation=hyperparams['dense_activation']))
    
    # Output layer
    model.add(Dense(1))
    
    # Configure optimizer
    if hyperparams['optimizer'] == 'adam':
        optimizer = Adam(learning_rate=hyperparams['learning_rate'])
    else:
        optimizer = RMSprop(learning_rate=hyperparams['learning_rate'])
    
    # Compile model
    model.compile(optimizer=optimizer, loss='mse')
    
    # Define early stopping callback
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=0
    )
    
    # Train model for limited epochs to save time during tuning
    history = model.fit(
        X_train, y_train,
        epochs=20,  # Reduced epochs for hyperparameter search
        batch_size=32,
        validation_data=(X_val, y_val),
        sample_weight=weights_train,
        callbacks=[early_stop],
        verbose=0
    )
    
    # Return best validation loss
    return min(history.history['val_loss'])

def tune_hyperparameters(X_train, y_train, X_val, y_val, input_shape, weights_train, weights_val, 
                         n_trials=50, timeout=3600, study_name="debiased_lstm_tuning"):
    """
    Perform hyperparameter tuning using Optuna
    
    Parameters:
    -----------
    X_train, y_train : np.array
        Training data
    X_val, y_val : np.array
        Validation data
    input_shape : tuple
        Shape of input data (time_steps, features)
    weights_train, weights_val : np.array
        Sample weights
    n_trials : int
        Number of trials to run
    timeout : int
        Timeout in seconds
    study_name : str
        Name of the study
        
    Returns:
    --------
    Dictionary of best hyperparameters
    """
    logger.info(f"Starting hyperparameter tuning with Optuna ({n_trials} trials, {timeout}s timeout)")
    
    # Create Optuna study
    study = optuna.create_study(
        direction="minimize",  # Minimize validation loss
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=42)  # Use TPE algorithm with seed for reproducibility
    )
    
    # Define objective function with fixed parameters
    objective_func = lambda trial: objective(
        trial, X_train, y_train, X_val, y_val, input_shape, weights_train, weights_val
    )
    
    # Run optimization
    try:
        study.optimize(objective_func, n_trials=n_trials, timeout=timeout)
        
        # Get best hyperparameters
        best_params = study.best_params
        best_value = study.best_value
        
        logger.info(f"Best validation loss: {best_value}")
        logger.info(f"Best hyperparameters: {best_params}")
        
        return best_params
    
    except Exception as e:
        logger.error(f"Error during hyperparameter tuning: {e}")
        # Return default hyperparameters if optimization fails
        return {
            'units_1': 64,
            'dropout_1': 0.2,
            'activation_1': 'tanh',
            'units_2': 32,
            'dropout_2': 0.2,
            'activation_2': 'tanh',
            'dense_units': 16,
            'dense_activation': 'relu',
            'learning_rate': 0.001,
            'optimizer': 'adam'
        }


def train_model(model, X_train, y_train, X_val, y_val, weights_train, weights_val,
               product_indices, product_names, scaler_y, log_transformed=False,
               epochs=50, batch_size=32, patience=10, model_path='lstm_model.keras'):
    """
    Train model with fairness-aware callbacks
    
    Parameters:
    -----------
    model : Keras model
        Compiled model to train
    X_train, y_train : np.array
        Training data
    X_val, y_val : np.array
        Validation data
    weights_train, weights_val : np.array
        Sample weights
    product_indices, product_names : list
        Product identifiers
    scaler_y : RobustScaler
        Target scaler for inverse transformation
    log_transformed : bool
        Whether target was log-transformed
    epochs, batch_size, patience : int
        Training parameters
    model_path : str
        Path to save best model
        
    Returns:
    --------
    Training history and best model
    """
    logger.info(f"Training model with {epochs} max epochs and batch size {batch_size}")
    
    # Define callbacks
    callbacks = [
        # Early stopping based on validation loss
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Model checkpoint to save best model
        ModelCheckpoint(
            model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        
        # Custom fairness-aware callback
        FairnessAwareCallback(
            X_val=X_val,
            y_val=y_val,
            product_indices=product_indices,
            product_names=product_names,
            scaler_y=scaler_y,
            log_transformed=log_transformed,
            patience=patience  # Use same patience as early stopping
        )
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        sample_weight=weights_train,  # Use weighted samples
        callbacks=callbacks,
        verbose=1
    )
    
    # Load best model
    best_model = load_model(model_path)
    
    return history, best_model



def check_prediction_bias(model, X_test, y_test, scaler_y):
    """
    Check for systematic bias in predictions
    """
    logger.info("Checking for systematic prediction bias")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Inverse transform
    y_pred_real = scaler_y.inverse_transform(y_pred)
    y_test_real = scaler_y.inverse_transform(y_test)
    
    # Calculate errors
    errors = y_pred_real - y_test_real
    
    # Check for systematic over/under prediction
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    
    # Check if errors correlate with prediction magnitude
    error_vs_actual = np.corrcoef(y_test_real.flatten(), errors.flatten())[0, 1]
    
    # Check for heteroscedasticity (error variance changes with prediction magnitude)
    from scipy.stats import pearsonr
    abs_errors = np.abs(errors)
    hetero_corr, hetero_p = pearsonr(y_test_real.flatten(), abs_errors.flatten())
    
    # checking if bigger predictions lead to bigger errors — a smart way to test your model’s consistency. If hetero_corr is high and hetero_p is low, your model might need some tweaking (e.g., transforming the target, weighted loss, etc.).


    return {
        'mean_error': mean_error,
        'median_error': median_error,
        'error_vs_actual_corr': error_vs_actual,
        'heteroscedasticity_corr': hetero_corr,
        'heteroscedasticity_p_value': hetero_p
    }


def generate_metrics_per_product(model,
                                 X_test,
                                 y_test,
                                 features,
                                 product_indices,
                                 product_names,
                                 scaler_y,
                                 log_transformed=False):
    """
    Compute RMSE, MAE, and MAPE for each product in X_test/y_test.

    Args:
        model: Trained Keras model.
        X_test: np.array of shape (n_samples, timesteps, n_features).
        y_test: np.array of shape (n_samples, 1) (scaled).
        features: List of feature names used (must include 'product_encoded').
        product_indices: List of encoded product IDs (as in label_encoder).
        product_names: List of corresponding product names.
        scaler_y: Fitted RobustScaler for the target.
        log_transformed: Whether the target was log‑transformed.

    Returns:
        Dict mapping product_name → {'rmse', 'mae', 'mape', 'n_samples'}.
    """
    # 1. Predict and invert scaling
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_test)
    if log_transformed:
        y_pred = np.expm1(y_pred)
        y_true = np.expm1(y_true)

    # 2. Locate the product_encoded feature index
    prod_idx = features.index('product_encoded')

    # 3. Compute metrics per product
    metrics = {}
    for pid, pname in zip(product_indices, product_names):
        mask = X_test[:, 0, prod_idx] == pid
        if not mask.any():
            continue
        y_p = y_pred[mask].ravel()
        y_t = y_true[mask].ravel()

        rmse = np.sqrt(mean_squared_error(y_t, y_p))
        mae  = mean_absolute_error(y_t, y_p)

        nonzero = y_t != 0
        mape = (np.mean(np.abs((y_t[nonzero] - y_p[nonzero]) / y_t[nonzero])) * 100
                if nonzero.any() else np.nan)

        metrics[pname] = {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'n_samples': int(mask.sum())
        }

    return metrics


def evaluate_model_metrics(model, X_test, y_test, product_indices, product_names, scaler_y, 
                           log_transformed=False, output_dir='.'):
    """
    Evaluate model fairness across different products and visualize
    
    Parameters:
    -----------
    model : Keras model
        Trained model to evaluate
    X_test, y_test : np.array
        Test data
    product_indices, product_names : list
        Product identifiers
    scaler_y : RobustScaler
        Target scaler for inverse transformation
    log_transformed : bool
        Whether target was log-transformed
    output_dir : str
        Directory to save output visualizations
        
    Returns:
    --------
    Dictionary of fairness metrics
    """
    logger.info("Evaluating model fairness across products")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Inverse transform predictions and actual values
    y_pred_real = scaler_y.inverse_transform(y_pred)
    y_test_real = scaler_y.inverse_transform(y_test)
    
    # If log-transformed, convert back
    if log_transformed:
        y_pred_real = np.expm1(y_pred_real)
        y_test_real = np.expm1(y_test_real)
    
    # Calculate overall metrics
    overall_mse = np.mean((y_pred_real - y_test_real) ** 2)
    overall_rmse = np.sqrt(overall_mse)
    overall_mae = np.mean(np.abs(y_pred_real - y_test_real))
    
    # Calculate MAPE for non-zero values
    non_zero_mask = y_test_real > 0
    if np.sum(non_zero_mask) > 0:
        overall_mape = np.mean(np.abs((y_test_real[non_zero_mask] - y_pred_real[non_zero_mask]) / 
                                     y_test_real[non_zero_mask])) * 100
    else:
        overall_mape = np.nan
    
    # Calculate metrics by product
    product_metrics = {}
    idx2name = dict(zip(product_indices, product_names))
    
    # Check the product ID column in X_test
    # For 3D tensors [samples, timesteps, features]
    product_id_values = set()
    if X_test.ndim == 3:
        # Try to extract unique product ID values
        product_id_values = set(X_test[:, 0, 0])
        logger.info(f"Found product IDs in X_test: {product_id_values}")
    
    # Process each product index
    for pid in product_indices:
        pname = idx2name.get(pid)
        logger.info(f"Processing {pname} (ID: {pid})")
        
        # Try different approaches to match the product ID
        if X_test.ndim == 3:
            # APPROACH 1: Check if product ID matches directly
            prod_mask = X_test[:, 0, 0] == pid
            prod_count = np.sum(prod_mask)
            
            # APPROACH 2: If no matches, try the product indices positions
            if prod_count == 0:
                pid_position = product_indices.index(pid)
                # Try using the position as the encoded value (-1, 0, etc.)
                if pid_position < len(product_id_values):
                    # This maps product_indices positions to the actual values in X_test
                    # For example, if product_indices[0] = 41, we try X_test[:, 0, 0] == -1
                    possible_encoded_value = sorted(list(product_id_values))[pid_position]
                    prod_mask = X_test[:, 0, 0] == possible_encoded_value
                    prod_count = np.sum(prod_mask)
                    logger.info(f"  Using position-based mapping for {pname}: {pid_position} -> {possible_encoded_value}, found {prod_count} samples")
        else:
            # For 2D tensors [samples, features]
            prod_mask = X_test[:, 0] == pid
            prod_count = np.sum(prod_mask)
        
        # Log results of matching
        logger.info(f"  Found {prod_count} samples for {pname}")
        
        if not prod_mask.any():
            logger.warning(f"  Skipped {pname} - no samples found")
            continue
            
        y_p = y_pred_real[prod_mask]
        y_t = y_test_real[prod_mask]
        mse = np.mean((y_p - y_t) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_p - y_t))
        nzp = y_t > 0
        mape = np.mean(np.abs((y_t[nzp] - y_p[nzp]) / y_t[nzp])) * 100 if nzp.any() else np.nan
        
        product_metrics[pname] = {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'sample_count': int(prod_mask.sum())
        }
    
    logger.info(f"Collected metrics for {len(product_metrics)} products")
    
    # Create DataFrame from product metrics
    metrics_df = pd.DataFrame.from_dict(product_metrics, orient='index')
    
    if len(metrics_df) < 2:
        # not enough products to compare
        rmse_disparity = mape_disparity = np.nan
        best_rmse_product = worst_rmse_product = None
        best_mape_product = worst_mape_product = None
        best_product_rmse = worst_product_rmse = None
    else:
        # RMSE
        rmse_s = metrics_df['rmse'].dropna().sort_values()
        best_rmse_product = rmse_s.index[0]
        best_product_rmse = float(rmse_s.iloc[0])
        worst_rmse_product = rmse_s.index[-1]
        worst_product_rmse = float(rmse_s.iloc[-1])
        rmse_disparity = (worst_product_rmse / best_product_rmse
                          if best_product_rmse != 0 else np.nan)

        # MAPE
        mape_s = metrics_df['mape'].dropna().sort_values()
        best_mape_product = mape_s.index[0]
        worst_mape_product = mape_s.index[-1]
        mape_disparity = (float(mape_s.iloc[-1]) / float(mape_s.iloc[0])
                          if mape_s.iloc[0] != 0 else np.nan)
    
    # Create visualizations
    
    # Plot RMSE by product
    plt.figure(figsize=(12, 6))
    if not metrics_df.empty and 'rmse' in metrics_df.columns:
        # Sort by RMSE value
        metrics_df_sorted = metrics_df.sort_values('rmse', ascending=False)
        
        sns.barplot(x=metrics_df_sorted.index, y=metrics_df_sorted['rmse'])
        plt.axhline(y=overall_rmse, color='r', linestyle='--', 
                    label=f"Overall RMSE: {overall_rmse:.2f}")
        plt.title('RMSE by Product')
        plt.xlabel('Product')
        plt.ylabel('RMSE')
        plt.xticks(rotation=90)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'rmse_by_product.png'))
        plt.close()
    
    # Plot MAPE by product
    plt.figure(figsize=(12, 6))
    if not metrics_df.empty and 'mape' in metrics_df.columns:
        # Sort by MAPE value
        metrics_df_sorted = metrics_df.sort_values('mape', ascending=False)
        
        sns.barplot(x=metrics_df_sorted.index, y=metrics_df_sorted['mape'])
        plt.axhline(y=overall_mape, color='r', linestyle='--',
                    label=f"Overall MAPE: {overall_mape:.2f}%")
        plt.title('MAPE by Product')
        plt.xlabel('Product')
        plt.ylabel('MAPE (%)')
        plt.xticks(rotation=90)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'mape_by_product.png'))
        plt.close()
    
    # Create scatter plot of error vs sample count
    plt.figure(figsize=(10, 6))
    if not metrics_df.empty and 'rmse' in metrics_df.columns and 'sample_count' in metrics_df.columns:
        plt.scatter(metrics_df['sample_count'], metrics_df['rmse'])
        
        # Add product labels
        for i, product in enumerate(metrics_df.index):
            plt.annotate(product, 
                         (metrics_df['sample_count'].iloc[i], metrics_df['rmse'].iloc[i]),
                         textcoords="offset points", 
                         xytext=(0,5), 
                         ha='center')
        
        plt.title('RMSE vs Sample Count')
        plt.xlabel('Sample Count')
        plt.ylabel('RMSE')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'rmse_vs_sample_count.png'))
        plt.close()
    
    # Create prediction vs actual plot
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test_real, y_pred_real, alpha=0.5)
    plt.plot([0, max(y_test_real.max(), y_pred_real.max())], 
             [0, max(y_test_real.max(), y_pred_real.max())], 
             'r--', label='Perfect Prediction')
    plt.title('Predicted vs Actual Quantity')
    plt.xlabel('Actual Quantity')
    plt.ylabel('Predicted Quantity')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predicted_vs_actual.png'))
    plt.close()
    
    # Create residual plot
    residuals = y_pred_real - y_test_real
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_real, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--', label='No Error')
    plt.title('Residuals vs Actual Quantity')
    plt.xlabel('Actual Quantity')
    plt.ylabel('Residual (Predicted - Actual)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residuals.png'))
    plt.close()
    
    # Log results
    logger.info(f"Overall RMSE: {overall_rmse:.2f}")
    logger.info(f"Overall MAPE: {overall_mape:.2f}%")
    if best_product_rmse is not None:
        logger.info(f"Best RMSE product: {best_rmse_product} ({best_product_rmse:.2f})")
        logger.info(f"Worst RMSE product: {worst_rmse_product} ({worst_product_rmse:.2f})")
    if best_mape_product is not None:
        logger.info(f"Best MAPE product: {best_mape_product}")
        logger.info(f"Worst MAPE product: {worst_mape_product}")

    return {
        'overall_metrics': {
            'rmse': overall_rmse,
            'mae': overall_mae,
            'mape': overall_mape
        },
        'product_metrics': product_metrics,
        'disparities': {
            'rmse_disparity': rmse_disparity,
            'mape_disparity': mape_disparity
        },
        'best_worst': {
            'best_rmse_product': best_rmse_product,
            'worst_rmse_product': worst_rmse_product,
            'best_mape_product': best_mape_product,
            'worst_mape_product': worst_mape_product,
            'best_product_rmse': best_product_rmse,
            'worst_product_rmse': worst_product_rmse
        }
    }

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

def save_artifacts(scaler_X, scaler_y, label_encoder, log_transformed, output_dir='.'):
    """
    Save preprocessing objects for later use
    
    Parameters:
    -----------
    scaler_X, scaler_y : Scaler
        Feature and target scalers
    label_encoder : LabelEncoder
        Encoder for product names
    log_transformed : bool
        Whether target was log-transformed
    output_dir : str
        Directory to save objects
        
    Returns:
    --------
    Dictionary of saved file paths
    """
    logger.info("Saving preprocessing objects")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save scalers
    scaler_X_path = os.path.join(output_dir, 'scaler_X.pkl')
    with open(scaler_X_path, 'wb') as f:
        pickle.dump(scaler_X, f)
    
    scaler_y_path = os.path.join(output_dir, 'scaler_y.pkl')
    with open(scaler_y_path, 'wb') as f:
        pickle.dump(scaler_y, f)
    
    # Save label encoder
    label_encoder_path = os.path.join(output_dir, 'label_encoder.pkl')
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Save transformation flag
    transform_info_path = os.path.join(output_dir, 'transform_info.pkl')
    with open(transform_info_path, 'wb') as f:
        pickle.dump({'log_transformed': log_transformed}, f)

    print("++++++++++++++++++++++++++++++++++++++")

    upload_to_gcs("lstm_model.keras", "trained-model-1")
    upload_to_gcs("scaler_y.pkl", "model_training_1")
    upload_to_gcs("model_report.md", "model_training_1")
    upload_to_gcs("label_encoder.pkl", "model_training_1")
    upload_to_gcs("scaler_X.pkl", "model_training_1")
    upload_to_gcs("transform_info.pkl", "model_training_1")

    upload_to_gcs("mape_by_product.png", "model_training_1")
    upload_to_gcs("monthly_distribution.png", "model_training_1")
    upload_to_gcs("predicted_vs_actual.png", "model_training_1")
    upload_to_gcs("product_distribution.png", "model_training_1")

    upload_to_gcs("residuals.png", "model_training_1")
    upload_to_gcs("residuals.png", "model_training_1")
    upload_to_gcs("rmse_by_product.png", "model_training_1")
    upload_to_gcs("rmse_vs_sample_count.png", "model_training_1")

    upload_to_gcs("shap_feature_importance.csv", "model_training_1")
    upload_to_gcs("shap_feature_importance.png", "model_training_1")
    upload_to_gcs("time_coverage.png", "model_training_1")
    upload_to_gcs("time_gaps.png", "model_training_1")
    upload_to_gcs("weekly_distribution.png", "model_training_1")
    
    logger.info(f"Preprocessing objects saved to {output_dir}")
    
    return {
        'scaler_X': scaler_X_path,
        'scaler_y': scaler_y_path,
        'label_encoder': label_encoder_path,
        'transform_info': transform_info_path
    }

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

def predict_future_demand(model, future_df, features, scaler_X, scaler_y, 
                        time_steps=5, bias_correction_func=None, log_transformed=False):
    """
    Predict future demand using the trained model
    
    Parameters:
    -----------
    model : Keras model
        Trained model
    future_df : DataFrame
        Future data with features
    features : list
        List of feature column names
    scaler_X, scaler_y : Scaler
        Feature and target scalers
    time_steps : int
        Number of time steps used for sequences
    bias_correction_func : function, optional
        Function to apply bias correction
    log_transformed : bool
        Whether target was log-transformed
        
    Returns:
    --------
    DataFrame with predictions
    """
    logger.info("Predicting future demand")
    
    # Scale features
    X_future = future_df[features].values
    X_future_scaled = scaler_X.transform(X_future)
    
    # Group by product
    predictions = []
    
    for product in future_df['Product Name'].unique():
        product_mask = future_df['Product Name'] == product
        product_future = future_df[product_mask]
        product_X = X_future_scaled[product_mask.values]
        
        # Need at least time_steps data points
        if len(product_X) < time_steps:
            logger.warning(f"Insufficient data for product '{product}'. Skipping predictions.")
            continue
        
        # Create sequences for prediction
        X_pred_seq = np.array([product_X[:time_steps]])
        
        # Make prediction
        y_pred = model.predict(X_pred_seq)
        
        # Apply bias correction if provided
        if bias_correction_func is not None:
            y_pred_corrected = bias_correction_func(y_pred)
        else:
            # Regular inverse transform
            y_pred_real = scaler_y.inverse_transform(y_pred)
            
            # If log-transformed, convert back
            if log_transformed:
                y_pred_real = np.expm1(y_pred_real)
            else:
                y_pred_real = y_pred_real
            
            y_pred_corrected = y_pred_real
        
        # Ensure non-negative predictions
        y_pred_final = np.maximum(0, y_pred_corrected)
        
        # Store prediction
        prediction_row = {
            'Date': product_future.iloc[0]['Date'],
            'Product Name': product,
            'Predicted Quantity': float(y_pred_final[0][0])
        }
        
        predictions.append(prediction_row)
    
    # Convert to DataFrame
    predictions_df = pd.DataFrame(predictions)
    
    logger.info(f"Generated {len(predictions_df)} predictions")
    
    return predictions_df

def analyze_model_with_shap(model, X_test, feature_names, output_dir='.', sample_size=100):
    """
    Analyze model predictions using SHAP values to explain feature importance
    
    Parameters:
    -----------
    model : Keras model
        Trained model to analyze
    X_test : np.array
        Test data to use for SHAP analysis
    feature_names : list
        Names of features corresponding to X_test columns
    output_dir : str
        Directory to save visualizations
    sample_size : int
        Number of samples to use for SHAP analysis (smaller for faster computation)
        
    Returns:
    --------
    Dictionary with SHAP results
    """
    logger.info("Starting SHAP analysis")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # If X_test is too large, take a random sample
    if len(X_test) > sample_size:
        # Use numpy's random choice without setting global seed
        indices = np.random.choice(len(X_test), sample_size, replace=False)
        X_sample = X_test[indices]
    else:
        X_sample = X_test
    
    # Extract the last time step for each sample
    last_timestep = X_sample[:, -1, :]
    
    # Create a function that takes 2D data and makes predictions with the model
    def model_predict(x):
        # Reshape the 2D input back to 3D for the LSTM model
        batch_size = x.shape[0]
        time_steps = X_sample.shape[1]
        features = x.shape[1]
        
        # Create sequences with the same values repeated for all timesteps
        x_reshaped = np.zeros((batch_size, time_steps, features))
        for i in range(batch_size):
            for t in range(time_steps):
                x_reshaped[i, t, :] = x[i, :]
        
        return model.predict(x_reshaped, verbose=0)
    
    # Create the SHAP explainer
    logger.info("Creating SHAP explainer")
    try:
        background = shap.kmeans(last_timestep, 10)  # Use K-means for more efficient background
        explainer = shap.KernelExplainer(model_predict, background)
    except Exception as e:
        logger.error(f"Error creating KernelExplainer with kmeans: {e}")
        # Fallback to simpler approach
        explainer = shap.KernelExplainer(model_predict, last_timestep[:10])
    
    # Calculate SHAP values
    logger.info("Calculating SHAP values (this may take a few minutes)...")
    try:
        shap_values = explainer.shap_values(last_timestep)
        
        # Convert to more convenient format if needed
        if isinstance(shap_values, list):
            shap_values = shap_values[0]  # Take first output for regression
    except Exception as e:
        logger.error(f"Error calculating SHAP values: {e}")
        # Create dummy SHAP values as fallback
        shap_values = np.zeros((len(last_timestep), last_timestep.shape[1]))
    
    # Create summary plot
    logger.info("Creating summary plot")
    try:
        plt.figure(figsize=(12, 8))
        # Without random_state parameter
        shap.summary_plot(shap_values, last_timestep, feature_names=feature_names, show=False)
        plt.title('SHAP Feature Importance (Impact on Model Output Magnitude)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'shap_summary_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.error(f"Error creating summary plot: {e}")
    
    # Create bar plot of feature importance
    logger.info("Creating bar plot")
    try:
        plt.figure(figsize=(12, 8))
        # Without random_state parameter
        shap.summary_plot(shap_values, last_timestep, feature_names=feature_names, plot_type='bar', show=False)
        plt.title('SHAP Feature Importance (Mean Absolute Impact)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'shap_feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.error(f"Error creating bar plot: {e}")
    
    # Calculate feature importance
    logger.info("Calculating feature importance")
    try:
        # Calculate mean absolute SHAP values for each feature
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        
        # Create a list of dictionaries for the DataFrame
        importance_data = []
        for i, feature in enumerate(feature_names):
            importance_data.append({
                'Feature': feature,
                'Importance': mean_abs_shap[i]
            })
        
        # Create DataFrame from list of dictionaries
        feature_importance = pd.DataFrame(importance_data)
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        # Save feature importance to CSV
        feature_importance.to_csv(os.path.join(output_dir, 'shap_feature_importance.csv'), index=False)
    except Exception as e:
        logger.error(f"Error calculating feature importance: {e}")
        feature_importance = pd.DataFrame(columns=['Feature', 'Importance'])
    
    logger.info("SHAP analysis complete")
    
    # Return summary of top features
    return {
        'top_features': feature_importance.head(10).to_dict(orient='records'),
        'shap_values': shap_values,
        'expected_value': explainer.expected_value
    }


def main():
    """
    Main function to run the debiased model training workflow
    """
    # Set output directory
    output_dir = '.'
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 1. Load data
        logger.info("Step 1: Loading data")
        query = """
        SELECT 
            sale_date AS 'Date', 
            product_name AS 'Product Name', 
            total_quantity AS 'Total Quantity'
        FROM SALES
        WHERE sale_date >= DATE_SUB(CURDATE(), INTERVAL 4 YEAR)
        ORDER BY sale_date;
        """
        
        try:
            df = get_latest_data_from_cloud_sql(query=query)
        except Exception as e:
            logger.error(f"Could not load from SQL: {e}")
            return
        
        # 2. Check data quality
        logger.info("Step 2: Checking data quality")
        df['Date'] = pd.to_datetime(df['Date'])
        data_analysis = analyze_data_distribution(df)
        plot_distribution_bias(data_analysis, output_dir)

        df['Date'] = pd.to_datetime(df['Date'])
        
        # 3. Handle class imbalance
        logger.info("Step 5: Handling class imbalance")
        balanced_df = handle_data_imbalance(df, min_samples=30)
        
        # 4. Create features
        logger.info("Step 6: Creating features")
        feature_df, label_encoder = create_features(balanced_df)
        
        # 5. Apply log transform if needed
        logger.info("Step 7: Applying log transform")
        transformed_df, log_transformed = apply_log_transform(feature_df)
        
        # 6. Define features for the model
        logger.info("Step 8: Defining model features")
        features = [
            'year', 'month', 'day', 'dayofweek', 'dayofyear', 'quarter',
            'product_encoded', 'is_weekend', 'is_month_start', 'is_month_end', 'is_holiday',
            'rolling_mean_7d', 'rolling_std_7d', 'rolling_median_7d', 'rolling_min_7d', 'rolling_max_7d',
            'lag_1d', 'lag_2d', 'lag_3d', 'lag_7d', 'lag_14d',
            'diff_1d', 'diff_7d', 'ewm_alpha_0.3', 'ewm_alpha_0.7',
            'product_mean', 'product_median', 'product_std', 'product_min', 'product_max'
        ]
        
        # Check that all features exist
        available_features = []
        for feature in features:
            if feature in transformed_df.columns:
                available_features.append(feature)
            else:
                logger.warning(f"Feature '{feature}' not available, skipping")
        
        logger.info(f"Using {len(available_features)} features: {available_features}")
        
        # 7. Create weighted sequences
        logger.info("Step 9: Creating weighted sequences")
        target_col = 'Total Quantity_log' if log_transformed else 'Total Quantity'
        
        (X_train, X_val, X_test, y_train, y_val, y_test,
         weights_train, weights_val, weights_test,
         scaler_X, scaler_y, product_indices, product_names) = create_weighted_sequences(
             transformed_df, available_features, target_col, log_transformed, time_steps=5)
        
        logger.info("Step 10: Performing hyperparameter tuning")
        input_shape = (X_train.shape[1], X_train.shape[2])

        # Run hyperparameter tuning
        tuned_hyperparams = tune_hyperparameters(
            X_train, y_train, X_val, y_val, input_shape, weights_train, weights_val,
            n_trials=3,  # Adjust based on computational resources
            timeout=1800,  # 30 minutes timeout
            study_name="debiased_lstm_tuning"
        )
        
        # 8. Build model
        logger.info("Step 10: Building model")
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        # Default hyperparameters (can be tuned)
        # 9. Build model with tuned hyperparameters
        logger.info("Step 11: Building model with tuned hyperparameters")
        model = build_model(input_shape, tuned_hyperparams)
        
        # 9. Train model
        logger.info("Step 11: Training model")
        model_path = os.path.join(output_dir, 'lstm_model.keras')
        
        history, best_model = train_model(
            model, X_train, y_train, X_val, y_val, weights_train, weights_val,
            product_indices, product_names, scaler_y, log_transformed,
            epochs=50, batch_size=32, patience=10, model_path=model_path
        )
        

        # 10. Check for prediction bias
        prediction_bias = check_prediction_bias(model, X_test, y_test, scaler_y)

        # 11. Evaluate model fairness
        logger.info("Step 12: Evaluating model fairness")
        fairness_metrics = evaluate_model_metrics(
            best_model, X_test, y_test, product_indices, product_names, 
            scaler_y, log_transformed, output_dir
        )


        # After evaluating model metrics
        logger.info("Step 15: Analyzing model with SHAP")
        shap_results = analyze_model_with_shap(
            best_model,
            X_test,
            available_features,
            output_dir,
            sample_size=100  # Adjust based on your dataset size
        )

        product_metrics = generate_metrics_per_product(
            best_model,
            X_test, y_test,
            features,            # same list you used when building sequences
            product_indices,     # from create_weighted_sequences
            product_names,       # from create_weighted_sequences
            scaler_y,
            log_transformed
        )
        for product, m in product_metrics.items():
            print(f"{product}: RMSE={m['rmse']:.2f}, MAE={m['mae']:.2f}, MAPE={m['mape']:.1f}% (n={m['n_samples']})")

        query = """
        SELECT 
            model_name, rmse, mae, mape
        FROM MODEL_METRICS;
        """


        try:
            model_metrics_df = get_latest_data_from_cloud_sql(query=query)
        except Exception as e:
            logger.warning(f"Could not load from SQL: {e}")
            model_metrics_df = pd.DataFrame()

        # Check if DataFrame is empty or missing 'rmse' column
        if model_metrics_df.empty or 'rmse' not in model_metrics_df.columns:
            logger.warning("Model metrics DataFrame is empty or missing 'rmse' column.")
            existing_model_rmse_value = None
        else:
            existing_model_rmse_value = model_metrics_df['rmse'].iloc[0]

        overall_metrics = fairness_metrics['overall_metrics']
        current_model_rmse = overall_metrics['rmse']

        if existing_model_rmse_value is None or new_product_flag or existing_model_rmse_value > current_model_rmse:

            engine = get_connection()
            metrics_df = pd.DataFrame([overall_metrics])
            metrics_df['model_name'] = "LSTM_model"
            metrics_df.to_sql("MODEL_METRICS", engine, if_exists="replace", index=False)

            # 12. Apply bias correction
            logger.info("Step 13: Creating bias correction function")
            bias_correction_func = correct_prediction_bias(
                best_model, X_val, y_val, scaler_y, log_transformed
            )
            
            # 13. Save preprocessing objects
            logger.info("Step 14: Saving preprocessing objects")
            save_artifacts(
                scaler_X, scaler_y, label_encoder, log_transformed, output_dir
            )
            
            # 14. Generate summary report
            logger.info("Step 14: Generating summary report")
            
            report = []
            report.append("# Debiased Demand Forecasting Model Report")
            report.append(f"## Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            report.append("\n## 1. Data Distribution Analysis")
            
            # Product distribution
            report.append("\n### 1.1 Product Distribution in Training Data")
            product_counts = data_analysis['product_counts']
            report.append(f"- Total number of products: {len(product_counts)}")
            report.append(f"- Most frequent product: {product_counts.index[0]} ({product_counts.values[0]} samples)")
            report.append(f"- Least frequent product: {product_counts.index[-1]} ({product_counts.values[-1]} samples)")
            report.append(f"- Imbalance ratio (max/min): {product_counts.values[0]/product_counts.values[-1]:.2f}")
            
            if product_counts.values[0]/product_counts.values[-1] > 5:
                report.append("\n⚠️ **Warning**: Significant class imbalance detected. Generated synthetic samples for underrepresented products.")
            
            # Time coverage
            report.append("\n### 1.2 Time Gap Analysis")
            time_gaps = data_analysis['time_gaps']

            if time_gaps:

                gaps = [gap[1] for gap in time_gaps]  # List of time gaps
                products = [gap[0] for gap in time_gaps]  # List of product names

                # Get the max and min time gaps
                max_gap = max(gaps) if gaps else None
                min_gap = min(gaps) if gaps else None

                # Get the corresponding product names for the max and min gaps
                max_gap_product = products[gaps.index(max_gap)] if max_gap is not None else None
                min_gap_product = products[gaps.index(min_gap)] if min_gap is not None else None

                report.append(f"\n⚠️ **Warning**: Found large time gaps (>30 days) in {len(time_gaps)} products")
                
                report.append(f"- Minimum time gap of {max_gap} days for {max_gap_product}")
                report.append(f"- Maximum time gap of {min_gap} days for {min_gap_product}")

            # Time coverage
            report.append("\n### 1.2 Time Coverage Analysis")
            time_coverage = data_analysis['product_time_coverage']
            min_coverage = min([v['days_coverage'] for v in time_coverage.values()])
            max_coverage = max([v['days_coverage'] for v in time_coverage.values()])
            
            report.append(f"- Minimum time coverage: {min_coverage} days")
            report.append(f"- Maximum time coverage: {max_coverage} days")
            report.append(f"- Coverage ratio (max/min): {max_coverage/max(1, min_coverage):.2f}")
            
            if max_coverage/max(1, min_coverage) > 3:
                report.append("\n⚠️ **Warning**: Significant variation in time coverage between products. Some products may lack sufficient historical data.")
            
            
            # Bias mitigation techniques
            report.append("\n## 2. Bias Mitigation Techniques Applied")
            report.append("- **Data Balancing**: Generated synthetic samples for underrepresented products")
            report.append("- **Robust Scaling**: Used RobustScaler instead of standard scaling")
            
            if log_transformed:
                report.append("- **Log Transformation**: Applied to target variable to handle heteroscedasticity")
            
            report.append("- **Weighted Training**: Used inverse frequency weighting during model training")
            report.append("- **Fairness-Aware Training**: Used custom callback to monitor performance across products")
            report.append("- **Bias Correction**: Applied post-processing correction to remove systematic bias")
            
            # Model architecture

            # Add this to the report generation section
            report.append("\n## 2. Hyperparameter Tuning")
            report.append(f"- Number of trials: 30")
            report.append(f"- Best hyperparameters: {tuned_hyperparams}")

            report.append("\n## 3. Model Architecture")
            report.append(f"- LSTM layers: 2 (units: {tuned_hyperparams['units_1']}, {tuned_hyperparams['units_2']})")
            report.append(f"- Activation functions: {tuned_hyperparams['activation_1']}, {tuned_hyperparams['activation_2']}")
            report.append(f"- Dropout rates: {tuned_hyperparams['dropout_1']}, {tuned_hyperparams['dropout_2']}")
            report.append(f"- Dense layer units: {tuned_hyperparams['dense_units']}")
            report.append(f"- Optimizer: {tuned_hyperparams['optimizer']} (learning rate: {tuned_hyperparams['learning_rate']})")
            
            

            # Prediction bias
            report.append("\n## 4. Prediction Bias Analysis")
            
            report.append(f"- Mean prediction error: {prediction_bias['mean_error']:.2f}")
            report.append(f"- Median prediction error: {prediction_bias['median_error']:.2f}")
            report.append(f"- Correlation between error and actual value: {prediction_bias['error_vs_actual_corr']:.2f}")
            report.append(f"- Heteroscedasticity correlation: {prediction_bias['heteroscedasticity_corr']:.2f} (p-value: {prediction_bias['heteroscedasticity_p_value']:.4f})")
            
            if abs(prediction_bias['mean_error']) > 5:
                report.append("\n⚠️ **Warning**: Systematic bias detected in predictions. Model tends to " + 
                            ("overpredict" if prediction_bias['mean_error'] > 0 else "underpredict") + 
                            " by an average of " + f"{abs(prediction_bias['mean_error']):.2f} units.")
            
            if abs(prediction_bias['error_vs_actual_corr']) > 0.3:
                report.append("\n⚠️ **Warning**: Correlation between error and actual value detected. Model performance varies with demand magnitude. Applied log transformation to the target variable")
            
            if abs(prediction_bias['heteroscedasticity_corr']) > 0.3 and prediction_bias['heteroscedasticity_p_value'] < 0.05:
                report.append("\n⚠️ **Warning**: Heteroscedasticity detected. Error variance increases with demand magnitude. Applied log transformation to the target variable")
            
            # Model performance
            report.append("\n## 5. Model Performance")
            
            overall_metrics = fairness_metrics['overall_metrics']
            report.append("### Overall Performance")
            report.append(f"- RMSE: {overall_metrics['rmse']:.2f}")
            report.append(f"- MAE: {overall_metrics['mae']:.2f}")
            report.append(f"- MAPE: {overall_metrics['mape']:.2f}%")
            
            # Fairness metrics
            report.append("\n### Fairness Metrics")
            
            disparities = fairness_metrics['disparities']
            best_worst = fairness_metrics['best_worst']
            
            report.append(f"- RMSE disparity ratio: {disparities['rmse_disparity']:.2f}")
            report.append(f"- MAPE disparity ratio: {disparities['mape_disparity']:.2f}")
            report.append(f"- Best performing product (RMSE): {best_worst['best_rmse_product']} with {best_worst['best_product_rmse']}")
            report.append(f"- Worst performing product (RMSE): {best_worst['worst_rmse_product']}  with {best_worst['worst_product_rmse']}")

            report_path = os.path.join(output_dir, 'model_report.md')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report))
            
            logger.info(f"Report saved to {report_path}")
            
            logger.info("Debiased model training complete")

            current_date = datetime.now().strftime("%Y-%m-%d")
            # Call the function
            send_email(
                emailid="talksick530@gmail.com",
                body="Hi,\n\nPlease find the model bias report attached.\n\nBest,\nTeam Talksick",
                subject= f"Model bias report '{current_date}'",
                attachment=report_path
            )

            logger.info("Bias repot sent to developer")

            model

        else:
            logger.info("Skipped saving the model as the current RMSE is higher")
    except Exception as e:
        logger.error(f"Error in debiased model training: {e}")
        raise

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)
    
    try:
        main()
    except Exception as e:
        print(f"Error running debiased model training: {e}")