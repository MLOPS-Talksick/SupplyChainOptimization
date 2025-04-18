import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import tensorflow as tf
import logging
import pickle
import os
from datetime import datetime, timedelta
from google.cloud.sql.connector import Connector
import sqlalchemy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('bias_detection')

def load_dataset(query=None, file_path=None):
    """
    Load dataset either from Cloud SQL or local file
    """
    try:
        if query:
            # Use your existing function to get data from Cloud SQL
            df = get_latest_data_from_cloud_sql(query=query)
        elif file_path:
            # Load from local CSV file
            df = pd.read_csv(file_path)
        else:
            raise ValueError("Either query or file_path must be provided")
        
        # Convert data types
        df['Total Quantity'] = df['Total Quantity'].astype(int)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Sort data by date
        df = df.sort_values('Date')
        
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

def analyze_data_distribution(df):
    """
    Analyze data distribution to check for potential biases
    """
    logger.info("Analyzing data distribution for potential biases")
    
    # Check for class imbalance in products
    product_counts = df['Product Name'].value_counts()
    
    # Calculate time coverage for each product
    product_time_coverage = {}
    for product in df['Product Name'].unique():
        product_df = df[df['Product Name'] == product]
        min_date = product_df['Date'].min()
        max_date = product_df['Date'].max()
        date_range = (max_date - min_date).days
        product_time_coverage[product] = {
            'min_date': min_date,
            'max_date': max_date,
            'days_coverage': date_range,
            'data_points': len(product_df)
        }
    
    # Check for seasonality bias
    df['month'] = df['Date'].dt.month
    df['dayofweek'] = df['Date'].dt.dayofweek
    
    monthly_distribution = df.groupby(['Product Name', 'month'])['Total Quantity'].mean().unstack()
    weekly_distribution = df.groupby(['Product Name', 'dayofweek'])['Total Quantity'].mean().unstack()
    
    return {
        'product_counts': product_counts,
        'product_time_coverage': product_time_coverage,
        'monthly_distribution': monthly_distribution,
        'weekly_distribution': weekly_distribution
    }

def plot_distribution_bias(analysis_results, output_dir='.'):
    """
    Create visualizations to highlight potential biases
    """
    logger.info("Creating bias visualization plots")
    
    # Plot product frequency distribution
    plt.figure(figsize=(12, 6))
    sns.barplot(x=analysis_results['product_counts'].index, 
                y=analysis_results['product_counts'].values)
    plt.title('Product Distribution in Training Data')
    plt.xlabel('Product')
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'product_distribution.png'))
    plt.close()
    
    # Plot time coverage
    coverage_df = pd.DataFrame.from_dict(analysis_results['product_time_coverage'], 
                                         orient='index')
    plt.figure(figsize=(12, 6))
    sns.barplot(x=coverage_df.index, y=coverage_df['days_coverage'])
    plt.title('Time Coverage by Product (Days)')
    plt.xlabel('Product')
    plt.ylabel('Days')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_coverage.png'))
    plt.close()
    
    # Plot monthly distribution heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(analysis_results['monthly_distribution'], cmap='viridis', annot=True)
    plt.title('Monthly Demand Distribution by Product')
    plt.xlabel('Month')
    plt.ylabel('Product')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'monthly_distribution.png'))
    plt.close()
    
    # Plot weekly distribution heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(analysis_results['weekly_distribution'], cmap='viridis', annot=True)
    plt.title('Weekly Demand Distribution by Product')
    plt.xlabel('Day of Week')
    plt.ylabel('Product')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'weekly_distribution.png'))
    plt.close()
    
    # # Plot outlier percentages
    # plt.figure(figsize=(12, 6))
    # outlier_df = pd.Series(analysis_results['outlier_percentage'])
    # sns.barplot(x=outlier_df.index, y=outlier_df.values)
    # plt.title('Percentage of Outliers by Product')
    # plt.xlabel('Product')
    # plt.ylabel('Outlier Percentage')
    # plt.xticks(rotation=90)
    # plt.tight_layout()
    # plt.savefig(os.path.join(output_dir, 'outlier_percentage.png'))
    # plt.close()

def detect_temporal_bias(df):
    """
    Detect temporal bias in the dataset
    """
    logger.info("Detecting temporal bias")
    
    # Calculate autocorrelation for each product
    autocorrelation = {}
    for product in df['Product Name'].unique():
        product_df = df[df['Product Name'] == product]
        if len(product_df) > 30:  # Ensure sufficient data
            acf_values = pd.Series(product_df['Total Quantity']).autocorr(lag=7)  # Weekly lag
            autocorrelation[product] = acf_values
    
    # Check for gaps in time series
    gaps = {}
    for product in df['Product Name'].unique():
        product_df = df[df['Product Name'] == product].sort_values('Date')
        if len(product_df) > 1:
            date_diffs = product_df['Date'].diff().dt.days.iloc[1:]
            gaps[product] = {
                'max_gap': date_diffs.max(),
                'avg_gap': date_diffs.mean(),
                'gap_days': date_diffs[date_diffs > 1].sum()
            }
    
    # Check for recency bias (more recent data)
    current_date = df['Date'].max()
    one_month_ago = current_date - timedelta(days=30)
    three_months_ago = current_date - timedelta(days=90)
    
    recency_bias = {}
    for product in df['Product Name'].unique():
        product_df = df[df['Product Name'] == product]
        total_points = len(product_df)
        
        if total_points > 0:
            last_month_points = len(product_df[product_df['Date'] >= one_month_ago])
            last_three_months_points = len(product_df[product_df['Date'] >= three_months_ago])
            
            recency_bias[product] = {
                'last_month_ratio': last_month_points / total_points,
                'last_three_months_ratio': last_three_months_points / total_points
            }
    
    return {
        'autocorrelation': autocorrelation,
        'time_gaps': gaps,
        'recency_bias': recency_bias
    }

def calculate_fairness_metrics(model, X_test, y_test, product_indices, product_names, scaler_y):
    """
    Calculate fairness metrics across different products
    """
    logger.info("Calculating fairness metrics across products")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Inverse transform the predictions and actual values
    y_pred_real = scaler_y.inverse_transform(y_pred)
    y_test_real = scaler_y.inverse_transform(y_test)
    
    # Calculate error metrics by product
    product_metrics = {}
    
    for product_idx, product_name in zip(product_indices, product_names):
        # Get indices for this product in the test set
        product_mask = X_test[:, 0, 6] == product_idx  # Assuming product_encoded is at index 6
        
        if np.sum(product_mask) > 0:
            # Calculate metrics for this product
            product_y_pred = y_pred_real[product_mask]
            product_y_test = y_test_real[product_mask]
            
            mae = np.mean(np.abs(product_y_pred - product_y_test))
            rmse = np.sqrt(np.mean((product_y_pred - product_y_test) ** 2))
            mape = np.mean(np.abs((product_y_test - product_y_pred) / np.maximum(1, product_y_test))) * 100
            
            product_metrics[product_name] = {
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'sample_count': np.sum(product_mask)
            }
    
    # Calculate overall metrics
    overall_mae = np.mean(np.abs(y_pred_real - y_test_real))
    overall_rmse = np.sqrt(np.mean((y_pred_real - y_test_real) ** 2))
    overall_mape = np.mean(np.abs((y_test_real - y_pred_real) / np.maximum(1, y_test_real))) * 100
    
    return {
        'product_metrics': product_metrics,
        'overall_metrics': {
            'mae': overall_mae,
            'rmse': overall_rmse,
            'mape': overall_mape
        }
    }

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

# def handle_data_imbalance(df, min_samples=30):
#     """
#     Handle data imbalance through augmentation or weighted sampling
#     """
#     logger.info("Handling data imbalance")
    
#     product_counts = df['Product Name'].value_counts()
#     products_to_augment = product_counts[product_counts < min_samples].index
    
#     augmented_df = df.copy()
    
#     for product in products_to_augment:
#         product_df = df[df['Product Name'] == product]
        
#         # If very few samples, use SMOTE-like approach by adding noise to existing samples
#         if len(product_df) < 10:
#             num_to_generate = min_samples - len(product_df)
            
#             # Generate synthetic samples by adding noise to quantity
#             for _ in range(num_to_generate):
#                 # Randomly select a row to duplicate with noise
#                 sample_idx = np.random.randint(0, len(product_df))
#                 new_sample = product_df.iloc[sample_idx].copy()
                
#                 # Add noise to quantity (±10%)
#                 noise_factor = 0.1
#                 quantity = new_sample['Total Quantity']
#                 new_quantity = max(1, int(quantity * (1 + np.random.uniform(-noise_factor, noise_factor))))
#                 new_sample['Total Quantity'] = new_quantity
                
#                 # Add to the augmented dataframe
#                 augmented_df = pd.concat([augmented_df, pd.DataFrame([new_sample])], ignore_index=True)
    
#     return augmented_df

def create_balanced_sequences(df, features, target_col, time_steps=5, test_size=0.2):
    """
    Create balanced sequences for LSTM model
    """
    logger.info("Creating balanced sequences")
    
    # Initialize scalers
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    # Encode product names
    label_encoder = LabelEncoder()
    df['product_encoded'] = label_encoder.fit_transform(df['Product Name'])
    
    # Scale features and target
    X_scaled = scaler_X.fit_transform(df[features])
    y_scaled = scaler_y.fit_transform(df[[target_col]])
    
    # Function to create sequences
    def create_sequences(X, y, time_steps=time_steps):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            Xs.append(X[i:(i + time_steps)])
            ys.append(y[i + time_steps])
        return np.array(Xs), np.array(ys)
    
    # Initialize containers for sequences
    all_X_seq = []
    all_y_seq = []
    product_indices = []
    product_names = []
    
    # Balance sequences across products
    products = df['Product Name'].unique()
    max_sequences_per_product = 0
    
    # First pass to determine the maximum number of sequences
    for product in products:
        product_indices.append(df[df['Product Name'] == product]['product_encoded'].iloc[0])
        product_names.append(product)
        
        product_df_indices = df[df['Product Name'] == product].index
        product_X = X_scaled[product_df_indices]
        product_y = y_scaled[product_df_indices]
        
        if len(product_X) > time_steps + 1:
            X_seq, y_seq = create_sequences(product_X, product_y, time_steps)
            max_sequences_per_product = max(max_sequences_per_product, len(X_seq))
    
    # Second pass to create balanced sequences
    for product, product_idx in zip(products, product_indices):
        product_df_indices = df[df['Product Name'] == product].index
        product_X = X_scaled[product_df_indices]
        product_y = y_scaled[product_df_indices]
        
        if len(product_X) > time_steps + 1:
            X_seq, y_seq = create_sequences(product_X, product_y, time_steps)
            
            # Balance by oversampling if needed
            if len(X_seq) < max_sequences_per_product:
                # Calculate how many additional sequences needed
                num_to_add = max_sequences_per_product - len(X_seq)
                
                # Randomly sample with replacement
                indices_to_duplicate = np.random.choice(len(X_seq), size=num_to_add, replace=True)
                
                # Add noise to the duplicated sequences
                for idx in indices_to_duplicate:
                    new_X = X_seq[idx].copy()
                    new_y = y_seq[idx].copy()
                    
                    # Add small noise to features (except categorical ones like product_encoded)
                    for j in range(new_X.shape[0]):  # For each time step
                        for k in range(new_X.shape[1]):  # For each feature
                            if k != 6:  # Skip product_encoded feature
                                noise = np.random.normal(0, 0.01)  # Small noise
                                new_X[j, k] += noise
                    
                    # Add small noise to target
                    new_y += np.random.normal(0, 0.01)
                    
                    # Append to sequences
                    X_seq = np.vstack([X_seq, [new_X]])
                    y_seq = np.vstack([y_seq, [new_y]])
            
            all_X_seq.append(X_seq)
            all_y_seq.append(y_seq)
    
    # Combine all sequences
    X_seq_combined = np.vstack(all_X_seq)
    y_seq_combined = np.vstack(all_y_seq)
    
    # Split to train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq_combined, y_seq_combined, test_size=test_size, random_state=42)
    
    return X_train, X_test, y_train, y_test, scaler_X, scaler_y, label_encoder, product_indices, product_names

def correct_prediction_bias(model, X_validation, y_validation, scaler_y):
    """
    Correct systematic bias in predictions
    """
    logger.info("Correcting prediction bias")
    
    # Make predictions on validation set
    y_pred = model.predict(X_validation)
    
    # Inverse transform
    y_pred_real = scaler_y.inverse_transform(y_pred)
    y_val_real = scaler_y.inverse_transform(y_validation)
    
    # Calculate errors
    errors = y_pred_real - y_val_real
    
    # Calculate systematic bias (median error is more robust to outliers)
    bias_correction = np.median(errors)
    
    logger.info(f"Systematic bias detected: {bias_correction}")
    
    # Function to apply correction to new predictions
    def correct_predictions(y_pred):
        y_pred_real = scaler_y.inverse_transform(y_pred)
        y_pred_corrected = y_pred_real - bias_correction
        return y_pred_corrected
    
    return correct_predictions

def evaluate_model_fairness(model, X_test, y_test, product_indices, product_names, scaler_y, output_dir='.'):
    """
    Evaluate model fairness across different products and visualize
    """
    logger.info("Evaluating model fairness")
    
    # Calculate fairness metrics
    fairness_metrics = calculate_fairness_metrics(
        model, X_test, y_test, product_indices, product_names, scaler_y)
    
    # Create dataframe from product metrics
    metrics_df = pd.DataFrame.from_dict(fairness_metrics['product_metrics'], orient='index')
    
    # Plot RMSE by product
    plt.figure(figsize=(12, 6))
    sns.barplot(x=metrics_df.index, y=metrics_df['rmse'])
    plt.axhline(y=fairness_metrics['overall_metrics']['rmse'], color='r', linestyle='--', 
                label=f"Overall RMSE: {fairness_metrics['overall_metrics']['rmse']:.2f}")
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
    sns.barplot(x=metrics_df.index, y=metrics_df['mape'])
    plt.axhline(y=fairness_metrics['overall_metrics']['mape'], color='r', linestyle='--',
                label=f"Overall MAPE: {fairness_metrics['overall_metrics']['mape']:.2f}%")
    plt.title('MAPE by Product')
    plt.xlabel('Product')
    plt.ylabel('MAPE (%)')
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mape_by_product.png'))
    plt.close()
    
    # Plot sample count by product
    plt.figure(figsize=(12, 6))
    sns.barplot(x=metrics_df.index, y=metrics_df['sample_count'])
    plt.title('Test Sample Count by Product')
    plt.xlabel('Product')
    plt.ylabel('Sample Count')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_count_by_product.png'))
    plt.close()
    
    # Create scatter plot of error vs sample count
    plt.figure(figsize=(10, 6))
    plt.scatter(metrics_df['sample_count'], metrics_df['rmse'])
    for i, product in enumerate(metrics_df.index):
        plt.annotate(product, 
                     (metrics_df['sample_count'].iloc[i], metrics_df['rmse'].iloc[i]),
                     textcoords="offset points", 
                     xytext=(0,10), 
                     ha='center')
    plt.title('RMSE vs Sample Count')
    plt.xlabel('Sample Count')
    plt.ylabel('RMSE')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rmse_vs_sample_count.png'))
    plt.close()
    
    return fairness_metrics


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

    MYSQL_HOST = "34.56.21.150"
    MYSQL_USER = "shrey"
    MYSQL_PASSWORD = "shrey"
    MYSQL_DATABASE = "combined_transaction_data"

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/svaru/Downloads/cloud_run.json"

    host = MYSQL_HOST
    user = MYSQL_USER
    password = MYSQL_PASSWORD
    database = MYSQL_DATABASE
    connector = Connector()

    def getconn():
        conn = connector.connect(
            "primordial-veld-450618-n4:us-central1:mlops-sql",  # Cloud SQL instance connection name
            "pymysql",  # Database driver
            user=user,  # Database user
            password=password,  # Database password
            db=database,
        )
        return conn

    pool = sqlalchemy.create_engine(
        "mysql+pymysql://",  # or "postgresql+pg8000://" for PostgreSQL, "mssql+pytds://" for SQL Server
        creator=getconn,
    )
    df = pd.read_sql(query, pool)
    connector.close()
    return df

def generate_bias_report(data_analysis, temporal_bias, prediction_bias, fairness_metrics, output_dir='.'):
    """
    Generate comprehensive bias report
    """
    logger.info("Generating bias report")
    
    report = []
    report.append("# Demand Forecasting Model Bias Assessment Report")
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
        report.append("\n⚠️ **Warning**: Significant class imbalance detected. Consider data augmentation or weighted training.")
    
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
    
    # # Outliers
    # report.append("\n### 1.3 Outlier Analysis")
    # outlier_pct = data_analysis['outlier_percentage']
    # max_outlier_pct = max(outlier_pct.values())
    
    # report.append(f"- Maximum outlier percentage: {max_outlier_pct:.2%}")
    # report.append(f"- Average outlier percentage: {sum(outlier_pct.values())/len(outlier_pct):.2%}")
    
    # if max_outlier_pct > 0.1:
    #     report.append("\n⚠️ **Warning**: High percentage of outliers detected. Consider robust scaling or outlier removal.")
    
    # Temporal bias
    report.append("\n## 2. Temporal Bias Analysis")
    
    # Autocorrelation
    report.append("\n### 2.1 Autocorrelation Analysis")
    autocorr = temporal_bias['autocorrelation']
    if autocorr:
        max_autocorr = max(autocorr.values())
        min_autocorr = min(autocorr.values())
        
        report.append(f"- Maximum weekly autocorrelation: {max_autocorr:.2f}")
        report.append(f"- Minimum weekly autocorrelation: {min_autocorr:.2f}")
        
        if max_autocorr > 0.7:
            report.append("\n⚠️ **Warning**: Strong weekly seasonality detected. Ensure model captures this pattern.")
    
    # Time gaps
    report.append("\n### 2.2 Time Gaps Analysis")
    gaps = temporal_bias['time_gaps']
    if gaps:
        max_gap = max([g['max_gap'] for g in gaps.values()])
        
        report.append(f"- Maximum gap in data: {max_gap} days")
        
        if max_gap > 7:
            report.append("\n⚠️ **Warning**: Significant gaps in time series data detected. Consider imputation for missing periods.")
    
    # Recency bias
    report.append("\n### 2.3 Recency Bias Analysis")
    recency = temporal_bias['recency_bias']
    if recency:
        avg_last_month = sum([r['last_month_ratio'] for r in recency.values()])/len(recency)
        
        report.append(f"- Average ratio of data from last month: {avg_last_month:.2f}")
        
        if avg_last_month > 0.5:
            report.append("\n⚠️ **Warning**: Potential recency bias detected. Data is heavily weighted towards recent periods.")
    
    # Prediction bias
    report.append("\n## 3. Prediction Bias Analysis")
    
    report.append(f"- Mean prediction error: {prediction_bias['mean_error']:.2f}")
    report.append(f"- Median prediction error: {prediction_bias['median_error']:.2f}")
    report.append(f"- Correlation between error and actual value: {prediction_bias['error_vs_actual_corr']:.2f}")
    report.append(f"- Heteroscedasticity correlation: {prediction_bias['heteroscedasticity_corr']:.2f} (p-value: {prediction_bias['heteroscedasticity_p_value']:.4f})")
    
    if abs(prediction_bias['mean_error']) > 5:
        report.append("\n⚠️ **Warning**: Systematic bias detected in predictions. Model tends to " + 
                     ("overpredict" if prediction_bias['mean_error'] > 0 else "underpredict") + 
                     " by an average of " + f"{abs(prediction_bias['mean_error']):.2f} units.")
    
    if abs(prediction_bias['error_vs_actual_corr']) > 0.3:
        report.append("\n⚠️ **Warning**: Correlation between error and actual value detected. Model performance varies with demand magnitude.")
    
    if abs(prediction_bias['heteroscedasticity_corr']) > 0.3 and prediction_bias['heteroscedasticity_p_value'] < 0.05:
        report.append("\n⚠️ **Warning**: Heteroscedasticity detected. Error variance increases with demand magnitude.")
    
    # Fairness metrics
    report.append("\n## 4. Model Fairness Analysis")
    
    # Overall metrics
    overall = fairness_metrics['overall_metrics']
    report.append("\n### 4.1 Overall Model Performance")
    report.append(f"- Overall RMSE: {overall['rmse']:.2f}")
    report.append(f"- Overall MAE: {overall['mae']:.2f}")
    report.append(f"- Overall MAPE: {overall['mape']:.2f}%")
    
    # Product-specific metrics
    report.append("\n### 4.2 Performance Across Products")
    product_metrics = fairness_metrics['product_metrics']
    
    # Calculate fairness disparities
    rmse_values = [m['rmse'] for m in product_metrics.values()]
    mape_values = [m['mape'] for m in product_metrics.values()]
    
    max_rmse = max(rmse_values)
    min_rmse = min(rmse_values)
    rmse_disparity = max_rmse / min_rmse
    
    max_mape = max(mape_values)
    min_mape = min(mape_values)
    mape_disparity = max_mape / min_mape
    
    report.append(f"- RMSE disparity ratio (max/min): {rmse_disparity:.2f}")
    report.append(f"- MAPE disparity ratio (max/min): {mape_disparity:.2f}")
    
    # Find worst performing products
    worst_rmse_product = list(product_metrics.keys())[rmse_values.index(max_rmse)]
    worst_mape_product = list(product_metrics.keys())[mape_values.index(max_mape)]
    
    report.append(f"- Worst RMSE performance: {worst_rmse_product} (RMSE: {max_rmse:.2f})")
    report.append(f"- Worst MAPE performance: {worst_mape_product} (MAPE: {max_mape:.2f}%)")
    
    if rmse_disparity > 3:
        report.append("\n⚠️ **Warning**: Significant performance disparity across products. Some products have much higher error rates than others.")
    
    # Recommendations
    report.append("\n## 5. Recommendations for Bias Mitigation")
    
    # Build recommendations based on findings
    recommendations = []
    
    # Data imbalance recommendations
    if product_counts.values[0]/product_counts.values[-1] > 5:
        recommendations.append("- **Data Augmentation**: Apply data augmentation techniques for products with low sample counts.")
        recommendations.append("- **Weighted Loss Function**: Implement weighted loss function to give more importance to underrepresented products.")
    
    # Time coverage recommendations
    if max_coverage/max(1, min_coverage) > 3:
        recommendations.append("- **Historical Data Collection**: Collect more historical data for products with limited time coverage.")
        recommendations.append("- **Transfer Learning**: Use knowledge from products with extensive history to inform predictions for newer products.")
    
    # # Outlier recommendations
    # if max_outlier_pct > 0.1:
    #     recommendations.append("- **Robust Scaling**: Use robust scaling methods (like RobustScaler) that are less sensitive to outliers.")
    #     recommendations.append("- **Anomaly Detection**: Implement anomaly detection to identify and handle extreme values appropriately.")
    
    # Temporal bias recommendations
    if 'autocorr' in locals() and max_autocorr > 0.7:
        recommendations.append("- **Seasonal Features**: Ensure model captures seasonality by including appropriate features (day of week, month, etc.).")
    
    if 'gaps' in locals() and max_gap > 7:
        recommendations.append("- **Time Series Imputation**: Apply time series imputation techniques to fill gaps in the data.")
    
    # Prediction bias recommendations
    if abs(prediction_bias['mean_error']) > 5:
        recommendations.append("- **Bias Correction**: Implement post-processing bias correction to adjust predictions.")
    
    if abs(prediction_bias['error_vs_actual_corr']) > 0.3:
        recommendations.append("- **Log Transformation**: Apply log transformation to the target variable if errors increase with demand magnitude.")
    
    # Fairness recommendations
    if rmse_disparity > 3:
        recommendations.append("- **Product-Specific Models**: Consider training separate models for products with significantly different patterns.")
        recommendations.append("- **Feature Engineering**: Develop product-specific features that capture the unique characteristics of each product.")
    
    # Add recommendations to report
    for rec in recommendations:
        report.append(rec)
    
    # Write report to file
    report_path = os.path.join(output_dir, 'bias_assessment_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    logger.info(f"Bias report generated and saved to {report_path}")
    
    return report_path

def main():
    """
    Main function to run the bias detection and mitigation workflow
    """
    # Set output directory
    output_dir = 'bias_analysis_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load your dataset (replace with actual data source)
    query = """
    SELECT 
        sale_date AS 'Date', 
        product_name AS 'Product Name', 
        total_quantity AS 'Total Quantity'
    FROM SALES
    ORDER BY sale_date;
    """
    
    try:
        # Load data from SQL (if available) or from CSV file
        
        df = get_latest_data_from_cloud_sql(query=query)

        df['Date'] = pd.to_datetime(df['Date'])
        
        # Analyze data distribution
        data_analysis = analyze_data_distribution(df)
        plot_distribution_bias(data_analysis, output_dir)
        
        # Detect temporal bias
        temporal_bias = detect_temporal_bias(df)
        
        # Handle data imbalance
        # balanced_df = handle_data_imbalance(df)
        
        # Extract date features
        df['year'] = df['Date'].dt.year
        df['month'] = df['Date'].dt.month
        df['day'] = df['Date'].dt.day
        df['dayofweek'] = df['Date'].dt.dayofweek
        df['dayofyear'] = df['Date'].dt.dayofyear
        df['quarter'] = df['Date'].dt.quarter
        
        # Add rolling statistics
        df = df.sort_values(['Product Name', 'Date'])
        df['rolling_mean_7d'] = df.groupby('Product Name')['Total Quantity'].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean())
        df['rolling_std_7d'] = df.groupby('Product Name')['Total Quantity'].transform(
            lambda x: x.rolling(window=7, min_periods=1).std().fillna(0))
        
        # Add lag features
        for lag in [1, 2, 3, 7]:
            df[f'lag_{lag}d'] = df.groupby('Product Name')['Total Quantity'].shift(lag)
        
        # Fill NaN values
        df = df.fillna(0)
        
        # Define features
        features = ['year', 'month', 'day', 'dayofweek', 'dayofyear', 'quarter', 
                    'product_encoded', 'rolling_mean_7d', 'rolling_std_7d',
                    'lag_1d', 'lag_2d', 'lag_3d', 'lag_7d']
        
        # Create balanced sequences
        X_train, X_test, y_train, y_test, scaler_X, scaler_y, label_encoder, product_indices, product_names = create_balanced_sequences(
            df, features, 'Total Quantity', time_steps=5, test_size=0.2)
        
        # Load or train your model
        try:
            model = load_model('lstm_model.keras')
            logger.info("Loaded existing model")
        except:
            logger.info("Couldn't load model, please train a new one")
            # Add code to train a new model here if needed
            raise ValueError("No model available for evaluation")
        
        # Check for prediction bias
        prediction_bias = check_prediction_bias(model, X_test, y_test, scaler_y)
        
        # Evaluate model fairness
        fairness_metrics = evaluate_model_fairness(
            model, X_test, y_test, product_indices, product_names, scaler_y, output_dir)
        
        # Correct prediction bias if needed
        if abs(prediction_bias['mean_error']) > 5:
            # Create a validation set from test set (just for example)
            val_size = len(X_test) // 2
            X_val, X_test = X_test[:val_size], X_test[val_size:]
            y_val, y_test = y_test[:val_size], y_test[val_size:]
            
            correction_func = correct_prediction_bias(model, X_val, y_val, scaler_y)
            
            # Apply correction to test set predictions
            y_pred = model.predict(X_test)
            y_pred_corrected = correction_func(y_pred)
            
            # Calculate corrected metrics
            y_test_real = scaler_y.inverse_transform(y_test)
            corrected_rmse = np.sqrt(np.mean((y_pred_corrected - y_test_real) ** 2))
            original_rmse = np.sqrt(np.mean((scaler_y.inverse_transform(y_pred) - y_test_real) ** 2))
            
            logger.info(f"Original RMSE: {original_rmse:.4f}, Corrected RMSE: {corrected_rmse:.4f}")
            logger.info(f"Improvement: {((original_rmse - corrected_rmse) / original_rmse) * 100:.2f}%")
        
        # Generate comprehensive bias report
        report_path = generate_bias_report(
            data_analysis, temporal_bias, prediction_bias, fairness_metrics, output_dir)
        
        logger.info(f"Bias analysis complete. Report saved to {report_path}")
        
    except Exception as e:
        logger.error(f"Error in bias analysis: {e}")
        raise

if __name__ == "__main__":
    main()