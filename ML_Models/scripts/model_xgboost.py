import pandas as pd
import numpy as np
from math import sqrt
import optuna
import pickle
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import shap
import os
import uuid
from google.cloud import aiplatform
from google.cloud import storage
from Data_Pipeline.scripts.logger import logger
from Data_Pipeline.scripts.utils import send_email, setup_gcp_credentials

# from logger import logger
# from utils import send_email
from google.cloud import storage
from ML_Models.scripts.utils import get_latest_data_from_cloud_sql
from dotenv import load_dotenv

load_dotenv()


def save_pkl_to_gcs(
    data, bucket_name="trained-model-1", destination_blob_name="model.pkl"
):
    """
    Saves a Python object as a pickle file to Google Cloud Storage.
    Will create the bucket if it doesn't exist.
    """
    if not setup_gcp_credentials():
        logger.error("Failed to set up GCP credentials. Cannot save model.")
        return False

    try:
        storage_client = storage.Client()

        # Check if bucket exists, create if it doesn't
        try:
            bucket = storage_client.get_bucket(bucket_name)
        except Exception:
            logger.info(f"Bucket {bucket_name} does not exist. Creating it...")
            bucket = storage_client.create_bucket(bucket_name)

        pkl_data = pickle.dumps(data)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_string(pkl_data)

        logger.info(
            f"File saved to gs://{bucket_name}/{destination_blob_name}"
        )
        return True
    except Exception as e:
        logger.error(f"Error saving model to GCP: {str(e)}")
        return False


# email = "talksick530@gmail.com"
email = "svarunanusheel@gmail.com"


def compute_rmse(y_true, y_pred):
    """Computes Root Mean Squared Error."""
    return sqrt(mean_squared_error(y_true, y_pred))


# =======================
# 1. Feature Engineering
# =======================


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-series features for each row.
    Adds date features, lag features (lag_1, lag_7, lag_14, lag_30),
    and rolling statistics (rolling_mean_3, rolling_mean_21, rolling_mean_14, rolling_std_7).
    You can add additional long-window features (e.g., 60-day rolling mean) as needed.
    """
    try:
        logger.info("Starting feature extraction.")
        df["Date"] = pd.to_datetime(df["Date"])# , format='%d-%m-%Y')
        df = df.sort_values(by=["Product_Name", "Date"]).reset_index(drop=True)
        logger.info("Converted Date column to datetime and sorted data.")

        # Date-based features
        df["day_of_week"] = df["Date"].dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df["day_of_month"] = df["Date"].dt.day
        df["day_of_year"] = df["Date"].dt.dayofyear
        df["month"] = df["Date"].dt.month
        df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
        logger.info("Extracted date-based features.")

        # Lag features
        for lag in [1, 7, 14, 30]:
            df[f"lag_{lag}"] = df.groupby("Product_Name")[
                "Total_Quantity"
            ].shift(lag)
        logger.info("Created lag features.")

        # Rolling window features
        for window in [3, 14, 21]:
            df[f"rolling_mean_{window}"] = df.groupby("Product_Name")[
                "Total_Quantity"
            ].transform(
                lambda x: x.shift(1)
                .rolling(window=window, min_periods=1)
                .mean()
            )
        df["rolling_std_7"] = df.groupby("Product_Name")[
            "Total_Quantity"
        ].transform(
            lambda x: x.shift(1).rolling(window=14, min_periods=1).std()
        )
        logger.info("Computed rolling statistics.")

        logger.info("Feature extraction completed.")
        return df

    except Exception as e:
        # Log the error and send an error email
        logger.error(f"An error occurred: {str(e)}")
        send_email(
            emailid=email,
            subject="Feature extraction Failure",
            body=f"An error occurred while feature extraction: {str(e)}",
        )


def create_target(df: pd.DataFrame, horizon: int = 7) -> pd.DataFrame:
    """
    Create an aggregated target as the average of the next `horizon` days' 'Total_Quantity'.
    This means each record's target is the average demand over the next 7 days.
    """
    try:
        logger.info("Starting target creation with horizon=%d.", horizon)
        df = df.sort_values(by=["Product_Name", "Date"]).reset_index(drop=True)
        df["target"] = df.groupby("Product_Name")["Total_Quantity"].transform(
            lambda x: x.shift(-1).rolling(window=horizon, min_periods=1).mean()
        )
        logger.info("Target variable created.")
        return df

    except Exception as e:
        # Log the error and send an error email
        logger.error(f"An error occurred: {str(e)}")
        send_email(
            emailid=email,
            subject="Creating target Failure",
            body=f"An error occurred while creating target: {str(e)}",
        )


# ==========================================
# 2. Automatic Train/Validation/Test Split
# ==========================================


def get_train_valid_test_split(
    df: pd.DataFrame, train_frac: float = 0.7, valid_frac: float = 0.1
):
    """
    Automatically splits the DataFrame into train, validation, and test sets based on the date range.
    """
    try:
        logger.info("Starting train-validation-test split")
        unique_dates = df["Date"].drop_duplicates().sort_values()
        n = len(unique_dates)

        train_cutoff = unique_dates.iloc[int(n * train_frac)]
        valid_cutoff = unique_dates.iloc[int(n * (train_frac + valid_frac))]

        logger.info(f"Train cutoff date: {train_cutoff}")
        logger.info(f"Validation cutoff date: {valid_cutoff}")

        train_df = df[df["Date"] < train_cutoff].copy()
        valid_df = df[
            (df["Date"] >= train_cutoff) & (df["Date"] < valid_cutoff)
        ].copy()
        test_df = df[df["Date"] >= valid_cutoff].copy()

        logger.info("Data split completed")
        return train_df, valid_df, test_df

    except Exception as e:
        # Log the error and send an error email
        logger.error(f"An error occurred: {str(e)}")
        send_email(
            emailid=email,
            subject="Getting Data Split Failure",
            body=f"An error occurred while splitting data: {str(e)}",
        )


# ==========================================
# 3. Hyperparameter Tuning using Optuna
# ==========================================


def objective(trial, train_df, valid_df, feature_columns, target_column):
    """
    Hyperparameter tuning objective function for XGBoost.
    """
    try:
        logger.info(f"Starting trial {trial.number}")
        param = {
            "objective": "reg:squarederror",
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.005, 0.2, log=True
            ),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.6, 1.0
            ),
            "gamma": trial.suggest_float("gamma", 0, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 1),
            "random_state": 42,
        }

        logger.info(f"Trial {trial.number}: Parameters selected {param}")
        model = XGBRegressor(**param)
        model.fit(train_df[feature_columns], train_df[target_column])

        valid_pred = model.predict(valid_df[feature_columns])
        rmse = compute_rmse(valid_df[target_column], valid_pred)
        logger.info(f"Trial {trial.number}: RMSE = {rmse}")
        return rmse

    except Exception as e:
        # Log the error and send an error email
        logger.error(f"An error occurred: {str(e)}")
        send_email(
            emailid=email,
            subject="Objective Failure",
            body=f"An error occurred while objective: {str(e)}",
        )


def hyperparameter_tuning(
    train_df, valid_df, feature_columns, target_column, n_trials: int = 100
):
    try:
        logger.info("Starting hyperparameter tuning")
        study = optuna.create_study(
            direction="minimize", pruner=optuna.pruners.MedianPruner()
        )
        study.optimize(
            lambda trial: objective(
                trial, train_df, valid_df, feature_columns, target_column
            ),
            n_trials=n_trials,
        )
        best_params = study.best_trial.params
        logger.info(f"Best parameters: {best_params}")
        return best_params
    except Exception as e:
        # Log the error and send an error email
        logger.error(f"An error occurred: {str(e)}")
        send_email(
            emailid=email,
            subject="Hyperparameter tuning Failure",
            body=f"An error occurred while hyperparameter tuning: {str(e)}",
        )


# ==========================================
# 4. Model Training Function (without early_stopping_rounds)
# ==========================================


def model_training(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_columns: list,
    target_column: str,
    params: dict = None,
):
    """
    Trains an XGBoost one-step forecasting model using the specified features and target.
    Uses an eval_set to display loss during training.
    """
    try:
        logger.info("Starting model training")
        if params is None:
            params = {
                "objective": "reg:squarederror",
                "learning_rate": 0.1,
                "max_depth": 6,
                "n_estimators": 200,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
            }
        model = XGBRegressor(**params)
        eval_set = [
            (train_df[feature_columns], train_df[target_column]),
            (valid_df[feature_columns], valid_df[target_column]),
        ]
        model.fit(
            train_df[feature_columns],
            train_df[target_column],
            eval_set=eval_set,
            verbose=True,
        )
        logger.info("Model training completed")
        return model

    except Exception as e:
        # Log the error and send an error email
        logger.error(f"An error occurred: {str(e)}")
        send_email(
            emailid=email,
            subject="Model Training Failure",
            body=f"An error occurred while model training: {str(e)}",
        )


# ==========================================
# 5. Iterative Forecasting Function (Updated to include dummy features)
# ==========================================


def iterative_forecast(
    model,
    df_product: pd.DataFrame,
    forecast_days: int = 7,
    product_columns: list = None,
) -> pd.DataFrame:
    """
    Iteratively forecast the next `forecast_days` for a given product using a trained XGBoost model.
    Ensures that missing product columns are handled correctly.

    Parameters:
    - model: Trained XGBoost model
    - df_product: DataFrame containing historical data for a single product
    - forecast_days: Number of days to forecast
    - product_columns: List of all product dummy feature names from training

    Returns:
    - DataFrame with forecasted values
    """
    try:
        history = (
            df_product.copy().sort_values(by="Date").reset_index(drop=True)
        )
        last_date = history["Date"].iloc[-1]
        forecasts = []

        base_feature_columns = [
            "lag_1",
            "lag_7",
            "lag_14",
            "lag_30",
            "rolling_mean_3",
            "rolling_mean_21",
            "rolling_mean_14",
            "rolling_std_7",
            "day_of_week",
            "is_weekend",
            "day_of_month",
            "day_of_year",
            "month",
            "week_of_year",
        ]

        logger.info(
            f"Starting forecast for {forecast_days} days for product {df_product['Product_Name'].iloc[0]}"
        )

        for day in range(forecast_days):
            next_date = last_date + pd.Timedelta(days=1)

            # Log feature generation for the next forecast day
            logger.debug(
                f"Generating features for forecast day {day + 1} (next_date: {next_date})"
            )

            # Build feature row as a dictionary (avoiding fragmented DataFrame warning)
            feature_row = {
                "day_of_week": next_date.dayofweek,
                "is_weekend": int(next_date.dayofweek in [5, 6]),
                "day_of_month": next_date.day,
                "day_of_year": next_date.timetuple().tm_yday,
                "month": next_date.month,
                "week_of_year": next_date.isocalendar().week,
            }

            # Lag features
            last_qty = history["Total_Quantity"].iloc[-1]
            feature_row["lag_1"] = last_qty
            feature_row["lag_7"] = (
                history.iloc[-7]["Total_Quantity"]
                if len(history) >= 7
                else last_qty
            )
            feature_row["lag_14"] = (
                history.iloc[-14]["Total_Quantity"]
                if len(history) >= 14
                else last_qty
            )
            feature_row["lag_30"] = (
                history.iloc[-30]["Total_Quantity"]
                if len(history) >= 30
                else last_qty
            )

            # Rolling statistics
            qty_list = history["Total_Quantity"].tolist()
            feature_row["rolling_mean_3"] = (
                np.mean(qty_list[-3:])
                if len(qty_list) >= 3
                else np.mean(qty_list)
            )
            feature_row["rolling_mean_21"] = (
                np.mean(qty_list[-21:])
                if len(qty_list) >= 21
                else np.mean(qty_list)
            )
            feature_row["rolling_mean_14"] = (
                np.mean(qty_list[-14:])
                if len(qty_list) >= 14
                else np.mean(qty_list)
            )
            feature_row["rolling_std_7"] = (
                np.std(qty_list[-7:])
                if len(qty_list) >= 7
                else np.std(qty_list)
            )

            # Log feature values
            logger.debug(f"Features generated: {feature_row}")

            # Convert dictionary to DataFrame
            X_pred = pd.DataFrame([feature_row])

            # Handle product encoding (Ensure all product columns exist)
            if product_columns is not None:
                current_product = df_product["Product_Name"].iloc[0]
                dummy_features = {
                    col: 0 for col in product_columns
                }  # Initialize all product columns to 0
                product_dummy = f"prod_{current_product}"

                if product_dummy in dummy_features:
                    dummy_features[product_dummy] = (
                        1  # Set the correct product column
                    )

                # Merge product dummies with X_pred
                X_pred = pd.concat(
                    [X_pred, pd.DataFrame([dummy_features])], axis=1
                )

            # Ensure correct column order (same as training)
            X_pred = X_pred[base_feature_columns + product_columns]

            # Make prediction
            next_qty = model.predict(X_pred)[0]
            next_qty = np.round(next_qty)  # Round predicted quantity

            # Store forecasted value
            forecasts.append(
                {"Date": next_date, "Predicted Quantity": next_qty}
            )

            # Update history with new prediction for next iteration
            new_row = feature_row.copy()
            new_row["Date"] = next_date
            new_row["Product_Name"] = df_product["Product_Name"].iloc[0]
            new_row["Total_Quantity"] = next_qty
            history = pd.concat(
                [history, pd.DataFrame([new_row])], ignore_index=True
            )

            last_date = next_date  # Move to the next day

        logger.info("Forecasting completed successfully.")
        return pd.DataFrame(forecasts)

    except Exception as e:
        # Log the error and send an error email
        logger.error(f"An error occurred: {str(e)}")
        send_email(
            emailid=email,
            subject="Iterative Forecast Failure",
            body=f"An error occurred while iterative forecast: {str(e)}",
        )
def save_model_to_model_registry(model, model_name="model.pkl"):
    try:
        project_id = os.environ.get("PROJECT_ID", "primordial-veld-450618-n4")
        location = os.environ.get("REGION", "us-central1")
        aiplatform.init(project=project_id, location=location)
        artifact_uri = os.environ.get("TRAINED_MODEL_BUCKET_URI", "gs://trained-model-1")
        serving_container_image_uri = os.environ.get("SERVING_CONTAINER_IMAGE_URI", "gcr.io/cloud-aiplatform/prediction/xgboost-cpu.1-1:latest")
        
        # List existing models
        existing_models = aiplatform.Model.list(
            filter=f'display_name="{model_name}"',
            order_by="create_time desc",
        )

        if existing_models:
            parent_model = existing_models[0]  # Get latest model version
            version_id = len(existing_models) + 1  # Increment version number
            logger.info(f"Updating {model_name} with version {version_id}...")
            vertex_model = aiplatform.Model.upload(
                parent_model=parent_model.resource_name,
                display_name=model_name,
                artifact_uri=artifact_uri,
                serving_container_image_uri=serving_container_image_uri,  # Change based on your ML framework
            )
        else:
            version_id = 1  # First version
            vertex_model = aiplatform.Model.upload(
                display_name=model_name,
                artifact_uri=artifact_uri,
                serving_container_image_uri=serving_container_image_uri,
                labels={"source": "gcs", "framework": "xg_boost"}
            )

            logger.info(f"Model {vertex_model.display_name} registered to Vertex AI with ID: {vertex_model.name}")
            logger.info(f"Model resource path: {vertex_model.resource_name}")

        # Uncomment below to enable deployment if needed
        # endpoint = aiplatform.Endpoint.create(
        #     display_name=model_name,
        #     project=project_id,
        #     location=location,
        # )

        # endpoint.deploy(
        #     model=vertex_model,
        #     deployed_model_display_name=model_name,
        #     machine_type="n1-standard-4",
        #     traffic_percentage=100,  # Ensures only the new version serves traffic
        # )
        # logger.info(f"Uploaded and Deployed {model_name} with version # {version_id}.")

        return vertex_model

    except Exception as e:
        logger.error(f"An error occurred while saving the model to the registry: {str(e)}")
        raise

def save_model(model, filename="model.pkl"):
    """
    Save model locally and to GCS bucket.
    Returns True if successful, False otherwise.
    """
    try:
        with open(filename, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Model saved locally as {filename}")

        # Upload to GCS
        success = save_pkl_to_gcs(model, destination_blob_name=filename)
        model_resource_name = save_model_to_model_registry(model, filename)
        
        print(f"Model uploaded successfully. Model name: {model_resource_name}")

        return success

    except Exception as e:
        # Log the error and send an error email
        logger.error(f"An error occurred: {str(e)}")
        send_email(
            emailid=email,
            subject="Save Model Failure",
            body=f"An error occurred while saving model: {str(e)}",
        )
        return False


# ==========================================
# 6. SHAP
# ==========================================
def shap_analysis(model, valid_df, feature_columns):
    try:
        # Initialize SHAP explainer (TreeExplainer is used for tree-based models like XGBoost)
        explainer = shap.TreeExplainer(model)

        # Calculate SHAP values for the validation set
        shap_values = explainer.shap_values(valid_df[feature_columns])

        # Plot summary plot (global feature importance)
        shap.summary_plot(shap_values, valid_df[feature_columns])

        # Plot a SHAP value for a single prediction (local explanation)

        shap.save_html(
            "shap_force_plot.html",
            shap.force_plot(
                explainer.expected_value,
                shap_values[0],
                valid_df[feature_columns].iloc[0],
            ),
        )

        # shap.initjs()
        # shap.force_plot(explainer.expected_value, shap_values[0], valid_df[feature_columns].iloc[0])
    except Exception as e:
        # Log the error and send an error email
        logger.error(f"An error occurred: {str(e)}")
        send_email(
            emailid=email,
            subject="SHAP Analysis Failure",
            body=f"An error occurred during SHAP analysis: {str(e)}",
        )


def setup_gcp_credentials():
    """
    Sets up the GCP credentials by setting the GOOGLE_APPLICATION_CREDENTIALS environment variable
    to point to the correct location of the GCP key file.
    """
    # The GCP key is always in the mounted secret directory
    gcp_key_path = "/app/secret/gcp-key.json"  # Path inside Docker container
    local_key_path = "secret/gcp-key.json"  # Path for local development

    # Try the Docker container path first, fall back to local path
    if os.path.exists(gcp_key_path):
        credentials_path = gcp_key_path
    elif os.path.exists(local_key_path):
        credentials_path = local_key_path
    else:
        logger.warning(
            f"GCP credentials file not found at {gcp_key_path} or {local_key_path}"
        )
        return False

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
    logger.info(f"Set GCP credentials path to: {credentials_path}")
    return True


def load_model(bucket_name: str, file_name: str):
    """
    Loads a pickle file (typically a model) from a GCP bucket and returns the loaded object.

    Args:
        bucket_name (str): The name of the Google Cloud Storage bucket.
        file_name (str): The name of the pickle file in the bucket.

    Returns:
        The Python object loaded from the pickle file, or None if loading fails.
    """
    if not setup_gcp_credentials():
        logger.error("Failed to set up GCP credentials. Cannot load model.")
        send_email(
            emailid=email,
            subject="Load model Failure",
            body=f"Failed to set up GCP credentials. Cannot load model from {bucket_name}/{file_name}",
        )
        return None

    try:
        import io  # Add missing import for BytesIO

        # Check if bucket exists
        storage_client = storage.Client()
        try:
            bucket = storage_client.get_bucket(bucket_name)
        except Exception as e:
            logger.info(
                f"Bucket {bucket_name} does not exist or is not accessible: {str(e)}"
            )
            return None

        # Check if blob exists
        blob = bucket.blob(file_name)
        if not blob.exists():
            logger.info(
                f"Model file {file_name} does not exist in bucket {bucket_name}"
            )
            return None

        blob_content = blob.download_as_string()

        file_extension = file_name.split(".")[-1].lower()
        if file_extension not in ["pkl", "pickle"]:
            logger.error(f"Unsupported file type for pickle: {file_extension}")
            return None

        model = pickle.loads(
            blob_content
        )  # Use loads instead of load with BytesIO
        logger.info(
            f"'{file_name}' from bucket '{bucket_name}' successfully loaded as pickle."
        )
        return model
    except Exception as e:
        logger.error(f"Error loading model from GCP: {str(e)}")
        send_email(
            emailid=email,
            subject="Load model Failure",
            body=f"An error occurred while Loading {file_name} model: {str(e)}",
        )
        return None


# ---------------------------
# 7. Hybrid Model Wrapper Class
# ---------------------------
class HybridTimeSeriesModel:
    """
    Wrapper class that contains a global model and product-specific models.
    The predict method uses a product-specific model if available,
    otherwise it falls back to the global model.
    """

    def __init__(
        self,
        global_model,
        product_models: dict,
        feature_columns: list,
        product_dummy_columns: list,
    ):
        self.global_model = global_model
        self.product_models = product_models  # dict: product -> model
        self.feature_columns = feature_columns
        self.product_dummy_columns = product_dummy_columns

    def predict(self, product_name, X: pd.DataFrame):
        """
        Predict using the product-specific model if available; otherwise use the global model.
        """
        logger.info(f"Prediction requested for product: {product_name}")
        # Prepare X: assume X already has base features and product dummy columns in the correct order.
        if product_name in self.product_models:
            model = self.product_models[product_name]
            logger.info(f"Using product-specific model for {product_name}")
        else:
            model = self.global_model
            logger.info(
                "No product-specific model found. Falling back to global model."
            )

        prediction = model.predict(X)
        logger.info(f"Prediction completed for {product_name}")
        return prediction


# ==========================================
# 8. Main Pipeline
# ==========================================


def main():
    try:
        # 1. Load data
        query = """
        SELECT 
            sale_date AS 'Date', 
            product_name AS 'Product_Name', 
            total_quantity AS 'Total_Quantity'
        FROM SALES
        ORDER BY sale_date;
    """

        df = get_latest_data_from_cloud_sql(query=query)


        # print(new_df.head())
        # df = pd.read_csv("E:/MLOps/SupplyChainOptimization/filename.csv")

        # 2. Create features and target on original data (keep for forecasting)
        logger.info(
            "Extracting features and creating target for the original dataset..."
        )
        original_df = extract_features(df.copy())
        original_df = create_target(original_df, horizon=7)
        original_df = original_df.dropna().reset_index(drop=True)

        # For training, work on a copy and then one-hot encode "Product_Name"
        logger.info(
            "Preparing training data by extracting features and creating target..."
        )
        df_train = extract_features(df.copy())
        df_train = create_target(df_train, horizon=7)
        df_train = df_train.dropna().reset_index(drop=True)

        # -------------------------------
        # New step: Preserve original product name for evaluation
        logger.info("Preserving original product name for evaluation...")
        df_train["Product"] = df_train["Product_Name"]
        # -------------------------------

        logger.info("One-hot encoding 'Product_Name' column...")
        df_train = pd.get_dummies(
            df_train, columns=["Product_Name"], prefix="prod"
        )

        # Update feature_columns to include one-hot encoded product columns.
        product_columns = [
            col for col in df_train.columns if col.startswith("prod_")
        ]
        other_features = [
            "lag_1",
            "lag_7",
            "lag_14",
            "lag_30",
            "rolling_mean_3",
            "rolling_mean_21",
            "rolling_mean_14",
            "rolling_std_7",
            "day_of_week",
            "is_weekend",
            "day_of_month",
            "day_of_year",
            "month",
            "week_of_year",
        ]
        feature_columns = other_features + product_columns
        target_column = "target"

        # ----- Step 2: Automatic Time-Based Train/Validation/Test Split -----
        logger.info(
            "Splitting the dataset into train, validation, and test sets..."
        )
        train_df, valid_df, test_df = get_train_valid_test_split(
            df_train, train_frac=0.7, valid_frac=0.1
        )

        # ----- Step 3: Hyperparameter Tuning using Optuna -----
        logger.info("Starting hyperparameter tuning with Optuna...")
        best_params = hyperparameter_tuning(
            train_df, valid_df, feature_columns, target_column, n_trials=5
        )
        logger.info(f"Best parameters from tuning: {best_params}")

        # ----- Step 4: Train Final Model on Train+Validation -----
        logger.info(
            "Training the final model on combined train and validation datasets..."
        )
        train_valid_df = pd.concat([train_df, valid_df], ignore_index=True)
        final_model = model_training(
            train_valid_df,
            valid_df,
            feature_columns,
            target_column,
            params=best_params,
        )

        # Compute and print training, validation, and test RMSE.
        logger.info("Evaluating the model performance...")
        train_rmse = compute_rmse(
            train_df[target_column],
            final_model.predict(train_df[feature_columns]),
        )
        valid_rmse = compute_rmse(
            valid_df[target_column],
            final_model.predict(valid_df[feature_columns]),
        )
        test_pred = final_model.predict(test_df[feature_columns])
        test_rmse = compute_rmse(test_df[target_column], test_pred)
        logger.info("Training RMSE:", train_rmse)
        logger.info("Validation RMSE:", valid_rmse)
        logger.info("Test RMSE:", test_rmse)

        # -------------------------------
        # New step: Calculate RMSE for each product in the test set.
        logger.info("Calculating RMSE per product in the test set...")
        test_df = test_df.copy()
        test_df["predicted"] = test_pred
        rmse_per_product = test_df.groupby("Product").apply(
            lambda d: compute_rmse(d[target_column], d["predicted"])
        )
        logger.info("Individual RMSE per product:")
        logger.info(rmse_per_product)

        # Identify biased products (RMSE more than 2 standard deviations from the mean)
        mean_rmse = rmse_per_product.mean()
        std_rmse = rmse_per_product.std()
        threshold = mean_rmse + 2 * std_rmse
        biased_products = rmse_per_product[
            rmse_per_product > threshold
        ].index.tolist()
        logger.info(
            "Biased products (to receive product-specific models):",
            biased_products,
        )
        send_email(
            body=f"Biased products (to receive product-specific models):{biased_products}",
            emailid=email,
            subject="BIAS report",
        )

        # Train product-specific models for biased products
        product_specific_models = {}
        for prod in biased_products:
            prod_train_df = train_valid_df[
                train_valid_df["Product"] == prod
            ].copy()
            # Ensure there is enough data to train a product-specific model
            if len(prod_train_df) < 30:
                logger.info(
                    f"Not enough data to train a product-specific model for {prod}. Skipping."
                )
                continue
            # Train using the same hyperparameters (or retune if desired)
            prod_model = model_training(
                prod_train_df,
                valid_df,
                feature_columns,
                target_column,
                params=best_params,
            )
            product_specific_models[prod] = prod_model
            logger.info(f"Trained product-specific model for {prod}.")

        # Create the hybrid model wrapper instance
        hybrid_model = HybridTimeSeriesModel(
            global_model=final_model,
            product_models=product_specific_models,
            feature_columns=feature_columns,
            product_dummy_columns=product_columns,
        )

        # Compute RMSE for the hybrid model on the test set
        logger.info(
            "Evaluating the hybrid model performance on the test set..."
        )
        # Get a product name for testing purposes - using the first unique product
        test_product = original_df["Product_Name"].unique()[0]
        hybrid_test_pred = hybrid_model.predict(
            test_product, test_df[feature_columns]
        )
        hybrid_rmse = compute_rmse(test_df[target_column], hybrid_test_pred)
        logger.info(f"Hybrid Model RMSE: {hybrid_rmse}")

        # Compute RMSE for the old model
        old_model = load_model("trained-model-1", "model.pkl")
        old_rmse = None
        if old_model is not None:
            logger.info(
                "Evaluating the old model performance on the test set..."
            )
            try:
                # Check if old_model has the predict method directly
                if hasattr(old_model, "predict"):
                    old_test_pred = old_model.predict(test_df[feature_columns])
                    old_rmse = compute_rmse(
                        test_df[target_column], old_test_pred
                    )
                    logger.info(f"Old Model RMSE: {old_rmse}")
                # Check if it's a HybridTimeSeriesModel
                elif (
                    hasattr(old_model, "__class__")
                    and old_model.__class__.__name__ == "HybridTimeSeriesModel"
                ):
                    old_test_pred = old_model.predict(
                        test_product, test_df[feature_columns]
                    )
                    old_rmse = compute_rmse(
                        test_df[target_column], old_test_pred
                    )
                    logger.info(f"Old Model (Hybrid) RMSE: {old_rmse}")
                else:
                    logger.error(
                        f"Loaded model is of unknown type: {type(old_model)}"
                    )
            except Exception as e:
                logger.error(f"Error using old model: {str(e)}")
                old_rmse = None
        else:
            logger.info("No previous model found. This is the first run.")

        # Decide whether to save the new model
        if old_rmse is not None and old_rmse < hybrid_rmse:
            logger.info(
                "Older model performed better, we won't save the new one..."
            )
        else:
            if old_rmse is None:
                logger.info(
                    "First time uploading a model. Saving the new model..."
                )
            else:
                logger.info(
                    "New model performs better. Saving the new model..."
                )

            save_success = save_model(final_model)
        if save_success:
            logger.info("Model saved successfully to GCS bucket.")
        else:
            logger.error("Failed to save model to GCS bucket.")

        # ----- Step 5: Iterative Forecasting for Each Product -----
        # For iterative forecasting, use the original_df (which still contains the original "Product_Name").
        logger.info("Starting iterative forecasting for each product...")
        products = original_df["Product_Name"].unique()
        all_forecasts = []

        for product in products:
            df_product = original_df[
                original_df["Product_Name"] == product
            ].copy()
            if len(df_product) >= 60:
                logger.info(f"Forecasting for product: {product}")
                fc = iterative_forecast(
                    final_model,
                    df_product,
                    forecast_days=7,
                    product_columns=product_columns,
                )
                fc["Product_Name"] = product
                all_forecasts.append(fc)

        all_forecasts_df = pd.concat(all_forecasts, ignore_index=True)
        logger.info("7-day forecasts for each product:")
        logger.info(all_forecasts_df)

        all_forecasts_df.to_csv("7_day_forecasts.csv", index=False)

        logger.info("Performing SHAP analysis on the model...")
        shap_analysis(final_model, valid_df, feature_columns)

        # Send a success email
        send_email(
            emailid=email,
            subject="Model Run Success",
            body="The model run has completed successfully and forecasts have been saved.",
        )

    except Exception as e:
        # Log the error and send an error email
        logger.error(f"An error occurred: {str(e)}")
        send_email(
            emailid=email,
            subject="Model Run Failure",
            body=f"An error occurred during the model run: {str(e)}",
        )


if __name__ == "__main__":
    main()
