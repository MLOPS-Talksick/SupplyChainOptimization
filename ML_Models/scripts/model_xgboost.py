import pandas as pd
import numpy as np
from math import sqrt
import optuna
import pickle
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import shap
from Data_Pipeline.scripts.logger import logger
from Data_Pipeline.scripts.utils import send_email, setup_gcp_credentials

# from logger import logger
# from utils import send_email
from google.cloud import storage
from ML_Models.scripts.utils import get_latest_data_from_cloud_sql


def save_pkl_to_gcs(
    data, bucket_name="trained-model-1", destination_blob_name="model.pkl"
):
    setup_gcp_credentials()

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    pkl_data = pickle.dumps(data)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_string(pkl_data)

    print(f"File saved to gs://{bucket_name}/{destination_blob_name}")


email = "svarunanusheel@gmail.com"
file_path = "C:/Users/svaru/Downloads/test.csv"


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
    and rolling statistics (rolling_mean_7, rolling_mean_14, rolling_std_7).
    You can add additional long-window features (e.g., 60-day rolling mean) as needed.
    """
    try:
        logger.info("Starting feature extraction.")
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values(by=["Product Name", "Date"]).reset_index(drop=True)
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
            df[f"lag_{lag}"] = df.groupby("Product Name")[
                "Total Quantity"
            ].shift(lag)
        logger.info("Created lag features.")

        # Rolling window features
        for window in [7, 14]:
            df[f"rolling_mean_{window}"] = df.groupby("Product Name")[
                "Total Quantity"
            ].transform(
                lambda x: x.shift(1)
                .rolling(window=window, min_periods=1)
                .mean()
            )
        df["rolling_std_7"] = df.groupby("Product Name")[
            "Total Quantity"
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
    Create an aggregated target as the average of the next `horizon` days' 'Total Quantity'.
    This means each record's target is the average demand over the next 7 days.
    """
    try:
        logger.info("Starting target creation with horizon=%d.", horizon)
        df = df.sort_values(by=["Product Name", "Date"]).reset_index(drop=True)
        df["target"] = df.groupby("Product Name")["Total Quantity"].transform(
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
            "rolling_mean_7",
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
            f"Starting forecast for {forecast_days} days for product {df_product['Product Name'].iloc[0]}"
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
            last_qty = history["Total Quantity"].iloc[-1]
            feature_row["lag_1"] = last_qty
            feature_row["lag_7"] = (
                history.iloc[-7]["Total Quantity"]
                if len(history) >= 7
                else last_qty
            )
            feature_row["lag_14"] = (
                history.iloc[-14]["Total Quantity"]
                if len(history) >= 14
                else last_qty
            )
            feature_row["lag_30"] = (
                history.iloc[-30]["Total Quantity"]
                if len(history) >= 30
                else last_qty
            )

            # Rolling statistics
            qty_list = history["Total Quantity"].tolist()
            feature_row["rolling_mean_7"] = (
                np.mean(qty_list[-7:])
                if len(qty_list) >= 7
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
                current_product = df_product["Product Name"].iloc[0]
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
            new_row["Product Name"] = df_product["Product Name"].iloc[0]
            new_row["Total Quantity"] = next_qty
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


def save_model(model, filename="final_model.pkl"):
    try:
        with open(filename, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Model saved as {filename}")

        save_pkl_to_gcs(model, destination_blob_name=filename)

    except Exception as e:
        # Log the error and send an error email
        logger.error(f"An error occurred: {str(e)}")
        send_email(
            emailid=email,
            subject="Save Model Failure",
            body=f"An error occurred while saving model: {str(e)}",
        )


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
            product_name AS 'Product Name', 
            total_quantity AS 'Total Quantity'
        FROM SALES
        ORDER BY sale_date;
    """

        df = get_latest_data_from_cloud_sql(query=query)

        # print(new_df.head())
        # df = pd.read_csv(file_path)

        # 2. Create features and target on original data (keep for forecasting)
        logger.info(
            "Extracting features and creating target for the original dataset..."
        )
        original_df = extract_features(df.copy())
        original_df = create_target(original_df, horizon=7)
        original_df = original_df.dropna().reset_index(drop=True)

        # For training, work on a copy and then one-hot encode "Product Name"
        logger.info(
            "Preparing training data by extracting features and creating target..."
        )
        df_train = extract_features(df.copy())
        df_train = create_target(df_train, horizon=7)
        df_train = df_train.dropna().reset_index(drop=True)

        # -------------------------------
        # New step: Preserve original product name for evaluation
        logger.info("Preserving original product name for evaluation...")
        df_train["Product"] = df_train["Product Name"]
        # -------------------------------

        logger.info("One-hot encoding 'Product Name' column...")
        df_train = pd.get_dummies(
            df_train, columns=["Product Name"], prefix="prod"
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
            "rolling_mean_7",
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
            train_df, valid_df, feature_columns, target_column, n_trials=15
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

        # (Optional) Save the hybrid model to a pickle file
        save_model(hybrid_model, filename="hybrid_model.pkl")
        logger.info("Hybrid model saved as 'hybrid_model.pkl'.")

        # ----- Step 5: Iterative Forecasting for Each Product -----
        # For iterative forecasting, use the original_df (which still contains the original "Product Name").
        logger.info("Starting iterative forecasting for each product...")
        products = original_df["Product Name"].unique()
        all_forecasts = []

        for product in products:
            df_product = original_df[
                original_df["Product Name"] == product
            ].copy()
            if len(df_product) >= 60:
                logger.info(f"Forecasting for product: {product}")
                fc = iterative_forecast(
                    final_model,
                    df_product,
                    forecast_days=7,
                    product_columns=product_columns,
                )
                fc["Product Name"] = product
                all_forecasts.append(fc)

        all_forecasts_df = pd.concat(all_forecasts, ignore_index=True)
        logger.info("7-day forecasts for each product:")
        logger.info(all_forecasts_df)

        all_forecasts_df.to_csv("7_day_forecasts.csv", index=False)
        logger.info("Saving final model...")
        save_model(final_model)

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
