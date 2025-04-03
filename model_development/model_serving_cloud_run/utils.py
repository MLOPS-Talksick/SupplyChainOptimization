import pandas as pd
import random

def generate_prediction_data(num_days, product_name, last_date):
    date_range = pd.date_range(last_date, periods=num_days, freq='D')

    predicted_quantities = [random.randint(1, 15) for _ in range(num_days)]

    df = pd.DataFrame({
        'date': date_range,
        'predicted_quantity': predicted_quantities
    })

    return df

# =================================PREDICTIONS PLACEHOLDER CODE========================================================

# def make_predictions(model, data, product_name=None, feature_columns=None):
#     """
#     Make predictions using either the global XGBoost model or the hybrid model.
    
#     Parameters:
#     -----------
#     model : object
#         Either an XGBoost model or a HybridTimeSeriesModel instance
#     data : pandas.DataFrame
#         The data to make predictions on, must contain all required feature columns
#     product_name : str, optional
#         The name of the product for which predictions are being made.
#         Required if model is a HybridTimeSeriesModel.
#     feature_columns : list, optional
#         List of feature column names. If not provided, will try to use model's feature_columns
#         attribute if available.
        
#     Returns:
#     --------
#     numpy.ndarray
#         The predicted values
#     """
#     try:
#         # Check if we have a hybrid model or a regular XGBoost model
#         if hasattr(model, "__class__") and model.__class__.__name__ == "HybridTimeSeriesModel":
#             if product_name is None:
#                 raise ValueError("Product name must be provided when using a hybrid model")
            
#             # For hybrid model, we need to pass both the product name and features
#             return model.predict(product_name, data[feature_columns])
        
#         # For regular XGBoost model, just use the standard predict method
#         else:
#             if feature_columns is None:
#                 # Try to get feature columns from model if available
#                 if hasattr(model, "feature_columns"):
#                     feature_columns = model.feature_columns
#                 else:
#                     raise ValueError("Feature columns must be provided when not using a hybrid model")
            
#             return model.predict(data[feature_columns])
            
#     except Exception as e:
#         logger.error(f"Error making predictions: {str(e)}")
#         raise

# def predict_new_data(model, new_data, product_name=None, feature_columns=None):
#     """
#     Prepare new data and make predictions using the provided model.
    
#     Parameters:
#     -----------
#     model : object
#         Either an XGBoost model or a HybridTimeSeriesModel instance
#     new_data : pandas.DataFrame
#         New data to make predictions on. Should contain at minimum 'Date' and 'Product_Name' columns
#         (if product_name is not provided separately)
#     product_name : str, optional
#         The product name to use for predictions. If not provided, will try to extract from new_data
#     feature_columns : list, optional
#         List of feature column names. If not provided, will try to use model's feature_columns attribute
        
#     Returns:
#     --------
#     pandas.DataFrame
#         DataFrame with original data plus predictions
#     """
#     # Make a copy to avoid modifying the original data
#     df = new_data.copy()
    
#     # Extract product_name from data if not provided
#     if product_name is None and 'Product_Name' in df.columns:
#         # Use the first product name if multiple exist
#         product_name = df['Product_Name'].iloc[0]
    
#     # Extract features
#     logger.info("Extracting features for prediction data...")
#     df = extract_features(df)
    
#     # One-hot encode product names if needed
#     if hasattr(model, "product_dummy_columns") or any(col.startswith("prod_") for col in feature_columns):
#         logger.info("One-hot encoding 'Product_Name' column...")
#         df = pd.get_dummies(df, columns=["Product_Name"], prefix="prod")
    
#     # Make predictions
#     logger.info(f"Making predictions for product: {product_name}")
#     predictions = make_predictions(model, df, product_name, feature_columns)
    
#     # Add predictions to the dataframe
#     df['predicted_quantity'] = predictions
    
#     return df

# # Example usage:
# def example_usage():
#     # Load the model
#     model_path = "path/to/model.pkl"
#     model = load_model("trained-model-1", "model.pkl")
    
#     # Get new data
#     query = """
#     SELECT 
#         sale_date AS 'Date', 
#         product_name AS 'Product_Name', 
#         total_quantity AS 'Total_Quantity'
#     FROM SALES
#     WHERE sale_date > CURDATE() - INTERVAL 30 DAY
#     ORDER BY sale_date;
#     """
#     new_data = get_latest_data_from_cloud_sql(query=query)
    
#     # Define feature columns
#     feature_columns = [
#         "lag_1", "lag_7", "lag_14", "lag_30",
#         "rolling_mean_3", "rolling_mean_21", "rolling_mean_14", "rolling_std_7",
#         "day_of_week", "is_weekend", "day_of_month", "day_of_year",
#         "month", "week_of_year"
#     ] + [col for col in new_data.columns if col.startswith("prod_")]
    
#     # Make predictions
#     results = predict_new_data(
#         model=model,
#         new_data=new_data,
#         feature_columns=feature_columns
#     )
    
#     # Display or save results
#     print(results[['Date', 'Product_Name', 'Total_Quantity', 'predicted_quantity']])
#     results.to_csv("predictions.csv", index=False)