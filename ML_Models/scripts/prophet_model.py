import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import pickle
import os
from sqlalchemy import text
from google.cloud import storage
from Data_Pipeline.scripts.logger import logger
from Data_Pipeline.scripts.utils import send_email, setup_gcp_credentials
from ML_Models.scripts.utils import get_latest_data_from_cloud_sql, get_cloud_sql_connection
from dotenv import load_dotenv

logger.info("Starting Prophet model training script")
# Set random seed for reproducibility
np.random.seed(42)

# email = "talksick530@gmail.com"
email = "svarunanusheel@gmail.com"
setup_gcp_credentials()

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
    df['Date'] = pd.to_datetime(df['Date'])
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

# Save preprocessing objects
try:
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

# Prophet requires specific column names: 'ds' for date and 'y' for target
products = df['Product Name'].unique()
logger.info(f"Found {len(products)} unique products")

# Dictionary to store models and metrics
prophet_models = {}
forecasts = {}
metrics_dict = {}

# Function to plot forecast vs actual
def plot_forecast_vs_actual(forecast, actuals, product_name, save=True):
    plt.figure(figsize=(12, 6))
    plt.plot(forecast['ds'], forecast['yhat'], label='Forecast')
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], 
                     color='gray', alpha=0.2, label='Confidence Interval')
    plt.scatter(actuals['ds'], actuals['y'], color='red', label='Actual', alpha=0.5)
    plt.title(f'Forecast vs Actual for {product_name}')
    plt.xlabel('Date')
    plt.ylabel('Quantity')
    plt.legend()
    plt.tight_layout()
    
    if save:
        file_name = f'forecast_{product_name.replace(" ", "_")}_{current_datetime}.png'
        plt.savefig(file_name, bbox_inches='tight', dpi=300)
        plt.close()
        return file_name
    else:
        plt.show()
        plt.close()

# Train a Prophet model for each product
overall_rmse = []
hyperparameters = {}

for product in products:
    logger.info(f"Training Prophet model for product: {product}")
    try:
        # Filter data for this product
        product_data = df[df['Product Name'] == product].copy()
        
        if len(product_data) < 30:
            logger.warning(f"Not enough data for product {product} (only {len(product_data)} records). Skipping.")
            continue
            
        # Prepare data for Prophet
        prophet_data = product_data[['Date', 'Total Quantity']].rename(
            columns={'Date': 'ds', 'Total Quantity': 'y'})
        
        # Split data into train and test
        train_size = int(len(prophet_data) * 0.8)
        train_data = prophet_data[:train_size]
        test_data = prophet_data[train_size:]
        
        if len(test_data) < 5:
            logger.warning(f"Test set too small for product {product}. Using cross-validation instead.")
            use_test_set = False
        else:
            use_test_set = True
            
        # Initialize and fit model with hyperparameters
        model = Prophet(
            changepoint_prior_scale=0.05,  # Flexibility of the trend
            seasonality_prior_scale=10.0,  # Flexibility of the seasonality
            holidays_prior_scale=10.0,     # Flexibility of the holiday effects
            seasonality_mode='multiplicative',  # Multiplicative seasonality
            weekly_seasonality=True,       # Weekly seasonality
            yearly_seasonality=True,       # Yearly seasonality
            daily_seasonality=False        # No daily seasonality
        )
        
        # Add monthly seasonality
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        
        # Fit the model
        model.fit(train_data)
        
        # Store hyperparameters
        hyperparameters[product] = {
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10.0,
            'holidays_prior_scale': 10.0,
            'seasonality_mode': 'multiplicative'
        }
        
        # Make predictions
        if use_test_set:
            # Forecast on test set
            future = model.make_future_dataframe(periods=len(test_data))
            forecast = model.predict(future)
            
            # Calculate metrics on test set
            forecast_test = forecast.iloc[-len(test_data):]
            y_pred = forecast_test['yhat'].values
            y_true = test_data['y'].values
            
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            
            logger.info(f"RMSE for {product}: {rmse:.4f}")
            logger.info(f"MAPE for {product}: {mape:.4f}%")
            
            # Plot forecast vs actual
            file_name = plot_forecast_vs_actual(forecast_test, test_data, product)
            
            # Store metrics
            metrics_dict[product] = {
                'rmse': rmse,
                'mape': mape,
                'plot_file': file_name
            }
            
            overall_rmse.append(rmse)
        else:
            # Perform cross-validation when test set is too small
            df_cv = cross_validation(model, initial='240 days', period='30 days', horizon='60 days')
            df_p = performance_metrics(df_cv)
            
            rmse = df_p['rmse'].mean()
            mape = df_p['mape'].mean() * 100
            
            logger.info(f"CV RMSE for {product}: {rmse:.4f}")
            logger.info(f"CV MAPE for {product}: {mape:.4f}%")
            
            # Forecast on all data
            future = model.make_future_dataframe(periods=30)  # 30 days into future
            forecast = model.predict(future)
            
            # Store metrics
            metrics_dict[product] = {
                'rmse': rmse,
                'mape': mape,
                'cv_metrics': df_p
            }
            
            overall_rmse.append(rmse)
        
        # Store model and forecast
        prophet_models[product] = model
        forecasts[product] = forecast
        
        # Plot model components
        fig = model.plot_components(forecast)
        plt.tight_layout()
        components_file = f'components_{product.replace(" ", "_")}_{current_datetime}.png'
        fig.savefig(components_file, bbox_inches='tight', dpi=300)
        plt.close(fig)
        
        # Upload plot to GCS
        upload_to_gcs(components_file, "model_training_1")
        
    except Exception as e:
        logger.error(f"Error training Prophet model for {product}: {e}")
        send_email(
            emailid=email,
            subject=f"Error training Prophet model for {product}",
            body=f"An error occurred while training Prophet model for {product}: {str(e)}",
        )
        continue

# Calculate overall mean RMSE
if overall_rmse:
    mean_rmse = np.mean(overall_rmse)
    logger.info(f"Mean RMSE across all products: {mean_rmse:.4f}")
else:
    mean_rmse = None
    logger.warning("No RMSE values calculated")

# Plot training results summary
try:
    logger.info("Plotting training results summary")
    plt.figure(figsize=(12, 6))
    
    # Sort products by RMSE
    sorted_metrics = {k: v for k, v in sorted(metrics_dict.items(), 
                                              key=lambda item: item[1]['rmse'])}
    
    products_list = list(sorted_metrics.keys())
    rmse_values = [sorted_metrics[p]['rmse'] for p in products_list]
    
    # Create bar chart of RMSE by product
    plt.bar(range(len(products_list)), rmse_values)
    plt.xticks(range(len(products_list)), products_list, rotation=90)
    plt.title('RMSE by Product')
    plt.ylabel('RMSE')
    plt.xlabel('Product')
    plt.axhline(y=mean_rmse, color='r', linestyle='--', label=f'Mean RMSE: {mean_rmse:.4f}')
    plt.legend()
    plt.tight_layout()
    
    summary_file = f'rmse_summary_{current_datetime}.png'
    plt.savefig(summary_file, bbox_inches='tight', dpi=300)
    plt.close()
    
    # Upload summary to GCS
    upload_to_gcs(summary_file, "model_training_1")
    
except Exception as e:
    logger.error(f"Error plotting training results summary: {e}")
    send_email(
        emailid=email,
        subject="Error plotting training results summary",
        body=f"An error occurred while plotting training results summary: {str(e)}",
    )

# Function to upload files to Google Cloud Storage
def upload_to_gcs(file_name, gcs_bucket_name):
    # Initialize Google Cloud Storage client
    storage_client = storage.Client()

    # Upload file
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

# Save models and artifacts
def save_artifacts():
    """
    Save model locally and to GCS bucket.
    Returns True if successful, False otherwise.
    """
    try:
        logger.info("Saving model artifacts")
        
        # Save all models in one pickle file
        with open('prophet_models.pkl', 'wb') as f:
            pickle.dump(prophet_models, f)
        
        # Save metrics
        with open('prophet_metrics.pkl', 'wb') as f:
            pickle.dump(metrics_dict, f)
        
        # Upload to GCS
        upload_to_gcs('prophet_models.pkl', "trained-model-1")
        upload_to_gcs('prophet_metrics.pkl', "model_training_1")
        upload_to_gcs('label_encoder.pkl', "model_training_1")
        
        # Upload all plot files
        for product, metrics in metrics_dict.items():
            if 'plot_file' in metrics:
                upload_to_gcs(metrics['plot_file'], "model_training_1")
        
        return True
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        send_email(
            emailid=email,
            subject="Error saving model artifacts",
            body=f"An error occurred while saving model artifacts: {str(e)}",
        )
        return False

save_artifacts()
print("Prophet models saved as 'prophet_models.pkl'")

# Save the model configuration results to SQL database
try:
    logger.info("Saving model configuration to database")
    
    # Prepare data for insertion
    data = {
        "model_type": "Prophet",
        "model_training_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "units_1": None,  # Prophet doesn't use these LSTM parameters
        "dropout_1": None,
        "activation_1": None,
        "units_2": None,
        "dropout_2": None,
        "activation_2": None,
        "dense_units": None,
        "dense_activation": None,
        "learning_rate": None,
        "optimizer": None,
        "train_loss": None,
        "test_loss": mean_rmse if mean_rmse is not None else float('nan'),
        "rmse": mean_rmse if mean_rmse is not None else float('nan')
    }

    pool = get_cloud_sql_connection()

    # Insert into table
    with pool.connect() as conn:
        insert_stmt = text("""
            INSERT INTO MODEL_CONFIG (
                model_type, model_training_date, units_1, dropout_1, activation_1,
                units_2, dropout_2, activation_2,
                dense_units, dense_activation,
                learning_rate, optimizer,
                train_loss, test_loss, rmse
            ) VALUES (
                :model_type, :model_training_date, :units_1, :dropout_1, :activation_1,
                :units_2, :dropout_2, :activation_2,
                :dense_units, :dense_activation,
                :learning_rate, :optimizer,
                :train_loss, :test_loss, :rmse
            )
        """)
        conn.execute(insert_stmt, data)
        conn.commit()
except Exception as e:
    print(f"Error saving model configuration to database: {e}")
    send_email(
        emailid=email,
        subject="Error saving model configuration to database",
        body=f"An error occurred while saving model configuration to database: {str(e)}",
    )

# Delete temporary files
try:
    logger.info("Deleting temporary files")
    file_paths = ["prophet_models.pkl", "prophet_metrics.pkl", "label_encoder.pkl"]
    
    # Add plot files
    for product, metrics in metrics_dict.items():
        if 'plot_file' in metrics:
            file_paths.append(metrics['plot_file'])
    
    # Add summary file
    file_paths.append(f'rmse_summary_{current_datetime}.png')
    
    # Add component files
    for product in prophet_models.keys():
        file_paths.append(f'components_{product.replace(" ", "_")}_{current_datetime}.png')

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

# Send success email
send_email(
    emailid=email,
    subject="Prophet model run successful",
    body=f"Successfully trained and saved the Prophet models. \n\n"
    f"Mean RMSE across all products: {mean_rmse:.4f if mean_rmse else 'N/A'}\n",
)

###################################
# Feature Importance Analysis for Prophet
###################################

# Prophet provides built-in feature importance via component analysis
def analyze_prophet_feature_importance():
    """
    Analyze and visualize the importance of different components in Prophet models.
    """
    logger.info("Analyzing Prophet feature importance")
    
    # Create summary dataframe for component contributions
    component_summary = []
    
    for product, model in prophet_models.items():
        try:
            # Get forecast for this product
            forecast = forecasts[product]
            
            # Calculate component contributions
            trend_contribution = forecast['trend'].max() - forecast['trend'].min()
            
            # Seasonality contributions
            yearly_contribution = 0
            weekly_contribution = 0
            monthly_contribution = 0
            
            if 'yearly' in forecast.columns:
                yearly_contribution = forecast['yearly'].max() - forecast['yearly'].min()
                
            if 'weekly' in forecast.columns:
                weekly_contribution = forecast['weekly'].max() - forecast['weekly'].min()
                
            if 'monthly' in forecast.columns:
                monthly_contribution = forecast['monthly'].max() - forecast['monthly'].min()
            
            # Add to summary
            component_summary.append({
                'Product': product,
                'Trend': trend_contribution,
                'Yearly Seasonality': yearly_contribution,
                'Weekly Seasonality': weekly_contribution,
                'Monthly Seasonality': monthly_contribution
            })
            
        except Exception as e:
            logger.error(f"Error analyzing importance for {product}: {e}")
            continue
    
    # Create dataframe
    if component_summary:
        component_df = pd.DataFrame(component_summary)
        
        # Plot component importance
        plt.figure(figsize=(14, 8))
        
        # Melt dataframe for easier plotting
        melted_df = pd.melt(component_df, id_vars=['Product'], 
                           value_vars=['Trend', 'Yearly Seasonality', 
                                       'Weekly Seasonality', 'Monthly Seasonality'],
                           var_name='Component', value_name='Contribution')
        
        # Calculate average importance by component
        avg_by_component = melted_df.groupby('Component')['Contribution'].mean().reset_index()
        avg_by_component = avg_by_component.sort_values('Contribution', ascending=False)
        
        # Plot
        plt.bar(avg_by_component['Component'], avg_by_component['Contribution'])
        plt.title('Average Component Contribution Across All Products')
        plt.ylabel('Contribution Magnitude')
        plt.xlabel('Component')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        importance_file = f'component_importance_{current_datetime}.png'
        plt.savefig(importance_file, bbox_inches='tight', dpi=300)
        plt.close()
        
        # Upload to GCS
        upload_to_gcs(importance_file, "model_training_1")
        
        # Clean up
        if os.path.exists(importance_file):
            os.remove(importance_file)
        
        return component_df
    else:
        return None

# Analyze feature importance
importance_df = analyze_prophet_feature_importance()
if importance_df is not None:
    print("\nComponent importance summary:")
    print(importance_df.describe())

###################################
# Product-Specific Analysis
###################################

# Function to analyze specific product forecast
def analyze_product_forecast(product_name):
    """
    Analyze forecast details for a specific product.
    """
    if product_name not in prophet_models:
        return f"No model found for product {product_name}"
    
    try:
        model = prophet_models[product_name]
        forecast = forecasts[product_name]
        
        # Get product data
        product_data = df[df['Product Name'] == product_name].copy()
        prophet_data = product_data[['Date', 'Total Quantity']].rename(
            columns={'Date': 'ds', 'Total Quantity': 'y'})
        
        # Plot forecast with components
        fig1 = model.plot(forecast, figsize=(12, 6))
        plt.title(f'Forecast for {product_name}')
        plt.tight_layout()
        
        fig2 = model.plot_components(forecast, figsize=(12, 10))
        plt.tight_layout()
        
        # Check for changepoints
        if hasattr(model, 'changepoints') and model.changepoints is not None:
            plt.figure(figsize=(12, 6))
            plt.plot(prophet_data['ds'], prophet_data['y'], 'k.')
            for cp in model.changepoints:
                plt.axvline(cp, color='r', linestyle='--', alpha=0.5)
            plt.title(f'Changepoints for {product_name}')
            plt.xlabel('Date')
            plt.ylabel('Quantity')
            plt.tight_layout()
        
        plt.show()
        
        # Get metrics for this product
        if product_name in metrics_dict:
            metrics = metrics_dict[product_name]
            print(f"\nMetrics for {product_name}:")
            print(f"RMSE: {metrics['rmse']:.4f}")
            print(f"MAPE: {metrics['mape']:.4f}%")
        
        return "Analysis complete"
    
    except Exception as e:
        print(f"Error analyzing forecast for {product_name}: {e}")
        return f"Error in analysis: {str(e)}"

# Analyze a specific product (for demonstration)
if products.size > 0:
    sample_product = products[0]
    analysis_result = analyze_product_forecast(sample_product)
