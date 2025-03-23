!pip install mlflow
!pip install pyngrok
!pip install pygtok
!pip install optuna

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
subprocess.Popen(["mlflow", "ui", "--backend-store-uri", MLFLOW_TRACKING_URI])

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("LSTM_vs_XGBoost_Comparison")

print("Enter your authtoken, which can be copied from https://dashboard.ngrok.com/auth")
conf.get_default().auth_token = getpass.getpass()
port=5000
public_url = ngrok.connect(port).public_url
print(f' * ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:{port}\"')

#xgboost
import pandas as pd
import numpy as np
from math import sqrt
import optuna
import pickle
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import shap
import mlflow
import mlflow.xgboost
import matplotlib.pyplot as plt
import seaborn as sns

# =======================
# Utility Functions
# =======================

def compute_rmse(y_true, y_pred):
    """Computes Root Mean Squared Error."""
    return sqrt(mean_squared_error(y_true, y_pred))

def compute_mape(y_true, y_pred):
    """Computes Mean Absolute Percentage Error."""
    # Avoid division by zero
    mask = y_true != 0
    return mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100

# =======================
# MLflow Logging Functions
# =======================

def log_feature_importance(model, feature_columns):
    """Log feature importance plot to MLflow"""
    # Create feature importance dataframe
    feature_importance = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
    plt.title('Feature Importance')
    plt.tight_layout()

    # Save the figure and log it to MLflow
    plt.savefig('feature_importance.png')
    mlflow.log_artifact('feature_importance.png')
    plt.close()

def log_prediction_actual_scatter(y_true, y_pred, set_name):
    """Log scatter plot of actual vs predicted values"""
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5)

    # Add perfect prediction line
    max_val = max(y_true.max(), y_pred.max())
    min_val = min(y_true.min(), y_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')

    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Predicted vs Actual Values ({set_name})')
    plt.grid(True)
    plt.tight_layout()

    # Save the figure and log it to MLflow
    filename = f'actual_vs_predicted_{set_name}.png'
    plt.savefig(filename)
    mlflow.log_artifact(filename)
    plt.close()

def log_metrics_comparison(train_metrics, valid_metrics, test_metrics):
    """Log comparison of RMSE and MAPE across dataset splits"""
    metrics = ['RMSE', 'MAPE']
    datasets = ['Train', 'Validation', 'Test']

    # Create a dataframe for plotting
    data = {
        'Dataset': ['Train', 'Validation', 'Test', 'Train', 'Validation', 'Test'],
        'Metric': ['RMSE', 'RMSE', 'RMSE', 'MAPE', 'MAPE', 'MAPE'],
        'Value': [
            train_metrics['RMSE'], valid_metrics['RMSE'], test_metrics['RMSE'],
            train_metrics['MAPE'], valid_metrics['MAPE'], test_metrics['MAPE']
        ]
    }
    df = pd.DataFrame(data)

    # Create the grouped bar chart
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Metric', y='Value', hue='Dataset', data=df)
    plt.title('Performance Metrics Comparison')
    plt.tight_layout()

    # Save the figure and log it to MLflow
    plt.savefig('metrics_comparison.png')
    mlflow.log_artifact('metrics_comparison.png')
    plt.close()

def log_rmse_per_product(rmse_per_product):
    """Log RMSE per product as a bar chart"""
    plt.figure(figsize=(12, 6))
    rmse_per_product.sort_values().plot(kind='bar')
    plt.title('RMSE per Product')
    plt.ylabel('RMSE')
    plt.xlabel('Product')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the figure and log it to MLflow
    plt.savefig('rmse_per_product.png')
    mlflow.log_artifact('rmse_per_product.png')
    plt.close()

def log_hyperparameter_importances(study):
    """Log hyperparameter importance from Optuna study"""
    param_importances = optuna.importance.get_param_importances(study)

    # Convert to dataframe for plotting
    importance_df = pd.DataFrame({
        'Parameter': list(param_importances.keys()),
        'Importance': list(param_importances.values())
    }).sort_values('Importance', ascending=False)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Parameter', data=importance_df)
    plt.title('Hyperparameter Importance')
    plt.tight_layout()

    # Save the figure and log it to MLflow
    plt.savefig('hyperparameter_importance.png')
    mlflow.log_artifact('hyperparameter_importance.png')
    plt.close()

def log_hyperparameter_optimization_history(study):
    """Log the optimization history from Optuna study"""
    # Plot optimization history
    plt.figure(figsize=(10, 6))
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.tight_layout()

    # Save the figure and log it to MLflow
    plt.savefig('optimization_history.png')
    mlflow.log_artifact('optimization_history.png')
    plt.close()

    # Plot parallel coordinate plot
    plt.figure(figsize=(12, 6))
    optuna.visualization.matplotlib.plot_parallel_coordinate(study)
    plt.tight_layout()

    # Save the figure and log it to MLflow
    plt.savefig('parallel_coordinate.png')
    mlflow.log_artifact('parallel_coordinate.png')
    plt.close()

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
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by=["Product Name", "Date"]).reset_index(drop=True)

    # Date-based features
    df["day_of_week"] = df["Date"].dt.dayofweek      # Monday=0, Sunday=6
    df["is_weekend"] = df["day_of_week"].isin([5,6]).astype(int)
    df["day_of_month"] = df["Date"].dt.day
    df["day_of_year"] = df["Date"].dt.dayofyear
    df["month"] = df["Date"].dt.month
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)

    # Lag features
    df["lag_1"] = df.groupby("Product Name")["Total Quantity"].shift(1)
    df["lag_7"] = df.groupby("Product Name")["Total Quantity"].shift(7)
    df["lag_14"] = df.groupby("Product Name")["Total Quantity"].shift(14)
    df["lag_30"] = df.groupby("Product Name")["Total Quantity"].shift(30)

    # Rolling window features (short-term)
    df["rolling_mean_7"] = df.groupby("Product Name")["Total Quantity"].transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=1).mean())
    df["rolling_mean_14"] = df.groupby("Product Name")["Total Quantity"].transform(
        lambda x: x.shift(1).rolling(window=14, min_periods=1).mean())
    df["rolling_std_7"] = df.groupby("Product Name")["Total Quantity"].transform(
        lambda x: x.shift(1).rolling(window=14, min_periods=1).std())  # You can change window size if desired.

    # (Optional) Add longer-window features:
    # df["rolling_mean_60"] = df.groupby("Product Name")["Total Quantity"].transform(
    #     lambda x: x.shift(1).rolling(window=60, min_periods=1).mean())
    # df["rolling_std_60"] = df.groupby("Product Name")["Total Quantity"].transform(
    #     lambda x: x.shift(1).rolling(window=60, min_periods=1).std())

    # Additional features can be added here (difference, pct change, Fourier terms, etc.)

    return df

def create_target(df: pd.DataFrame, horizon: int = 7) -> pd.DataFrame:
    """
    Create an aggregated target as the average of the next `horizon` days' 'Total Quantity'.
    This means each record's target is the average demand over the next 7 days.
    """
    df = df.sort_values(by=["Product Name", "Date"]).reset_index(drop=True)
    df["target"] = df.groupby("Product Name")["Total Quantity"].transform(
        lambda x: x.shift(-1).rolling(window=horizon, min_periods=1).mean()
    )
    return df

# ==========================================
# 2. Automatic Train/Validation/Test Split
# ==========================================

def get_train_valid_test_split(df: pd.DataFrame, train_frac: float = 0.7, valid_frac: float = 0.1):
    """
    Automatically splits the DataFrame into train, validation, and test sets based on the date range.
    """
    unique_dates = df["Date"].drop_duplicates().sort_values()
    n = len(unique_dates)

    train_cutoff = unique_dates.iloc[int(n * train_frac)]
    valid_cutoff = unique_dates.iloc[int(n * (train_frac + valid_frac))]

    print("Train cutoff date:", train_cutoff)
    print("Validation cutoff date:", valid_cutoff)

    train_df = df[df["Date"] < train_cutoff].copy()
    valid_df = df[(df["Date"] >= train_cutoff) & (df["Date"] < valid_cutoff)].copy()
    test_df  = df[df["Date"] >= valid_cutoff].copy()

    return train_df, valid_df, test_df

# ==========================================
# 3. Hyperparameter Tuning using Optuna
# ==========================================

def hyperparameter_tuning(train_df, valid_df, feature_columns, target_column, n_trials: int = 100):
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())

    # Define the objective function with mlflow tracking
    def objective_with_mlflow(trial, train_df, valid_df, feature_columns, target_column):
        with mlflow.start_run(nested=True):
            param = {
                "objective": "reg:squarederror",
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "n_estimators": trial.suggest_int("n_estimators", 100, 600),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "gamma": trial.suggest_float("gamma", 0, 10),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
                "reg_lambda": trial.suggest_float("reg_lambda", 0, 1),
                "random_state": 42
            }

            # Log parameters
            mlflow.log_params(param)

            model = XGBRegressor(**param)
            model.fit(train_df[feature_columns], train_df[target_column])

            # Calculate and log validation metrics
            valid_pred = model.predict(valid_df[feature_columns])
            rmse = compute_rmse(valid_df[target_column], valid_pred)
            mape = compute_mape(valid_df[target_column], valid_pred)

            mlflow.log_metric("val_rmse", rmse)
            mlflow.log_metric("val_mape", mape)

            print(f"Trial {trial.number}: RMSE = {rmse}, MAPE = {mape}")
            return rmse

    study.optimize(lambda trial: objective_with_mlflow(trial, train_df, valid_df, feature_columns, target_column), n_trials=n_trials)

    print("Best trial:")
    trial = study.best_trial
    for key, value in trial.params.items():
        print(f"  {key}: {value}")

    # Visualize and log hyperparameter optimization results
    log_hyperparameter_importances(study)
    log_hyperparameter_optimization_history(study)

    return trial.params, study

# ==========================================
# 4. Model Training Function
# ==========================================

def model_training(train_df: pd.DataFrame, valid_df: pd.DataFrame, feature_columns: list, target_column: str, params: dict = None):
    """
    Trains an XGBoost one-step forecasting model using the specified features and target.
    Uses an eval_set to display loss during training.
    """
    if params is None:
        params = {
            "objective": "reg:squarederror",
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 200,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        }
    model = XGBRegressor(**params)
    eval_set = [(train_df[feature_columns], train_df[target_column]),
                (valid_df[feature_columns], valid_df[target_column])]
    model.fit(train_df[feature_columns],
              train_df[target_column],
              eval_set=eval_set,
              verbose=True)
    return model

# ==========================================
# 5. Iterative Forecasting Function
# ==========================================

def iterative_forecast(model, df_product: pd.DataFrame, forecast_days: int = 7, product_columns: list = None) -> pd.DataFrame:
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
    history = df_product.copy().sort_values(by="Date").reset_index(drop=True)
    last_date = history["Date"].iloc[-1]
    forecasts = []

    base_feature_columns = [
        "lag_1", "lag_7", "lag_14", "lag_30",
        "rolling_mean_7", "rolling_mean_14", "rolling_std_7",
        "day_of_week", "is_weekend", "day_of_month", "day_of_year", "month", "week_of_year"
    ]

    for _ in range(forecast_days):
        next_date = last_date + pd.Timedelta(days=1)

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
        feature_row["lag_7"] = history.iloc[-7]["Total Quantity"] if len(history) >= 7 else last_qty
        feature_row["lag_14"] = history.iloc[-14]["Total Quantity"] if len(history) >= 14 else last_qty
        feature_row["lag_30"] = history.iloc[-30]["Total Quantity"] if len(history) >= 30 else last_qty

        # Rolling statistics
        qty_list = history["Total Quantity"].tolist()
        feature_row["rolling_mean_7"] = np.mean(qty_list[-7:]) if len(qty_list) >= 7 else np.mean(qty_list)
        feature_row["rolling_mean_14"] = np.mean(qty_list[-14:]) if len(qty_list) >= 14 else np.mean(qty_list)
        feature_row["rolling_std_7"] = np.std(qty_list[-7:]) if len(qty_list) >= 7 else np.std(qty_list)

        # Convert dictionary to DataFrame
        X_pred = pd.DataFrame([feature_row])

        # Handle product encoding (Ensure all product columns exist)
        if product_columns is not None:
            current_product = df_product["Product Name"].iloc[0]
            dummy_features = {col: 0 for col in product_columns}  # Initialize all product columns to 0
            product_dummy = f"prod_{current_product}"

            if product_dummy in dummy_features:
                dummy_features[product_dummy] = 1  # Set the correct product column

            # Merge product dummies with X_pred
            X_pred = pd.concat([X_pred, pd.DataFrame([dummy_features])], axis=1)

        # Ensure correct column order (same as training)
        all_columns = base_feature_columns + (product_columns if product_columns is not None else [])
        for col in all_columns:
            if col not in X_pred.columns:
                X_pred[col] = 0
        X_pred = X_pred[all_columns]

        # Make prediction
        next_qty = model.predict(X_pred)[0]
        next_qty = np.round(next_qty)  # Round predicted quantity

        # Store forecasted value
        forecasts.append({"Date": next_date, "Predicted Quantity": next_qty})

        # Update history with new prediction for next iteration
        new_row = feature_row.copy()
        new_row["Date"] = next_date
        new_row["Product Name"] = df_product["Product Name"].iloc[0]
        new_row["Total Quantity"] = next_qty
        history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)

        last_date = next_date  # Move to the next day

    return pd.DataFrame(forecasts)

# ==========================================
# 6. Model Saving Function
# ==========================================

def save_model(model):
    with open("final_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model saved as final_model.pkl")

# ==========================================
# 7. Main Pipeline
# ==========================================

def main():
    # Set up MLflow tracking
    experiment_name = "LSTM_vs_XGBoost_Comparison"
    mlflow.set_experiment(experiment_name)
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    with mlflow.start_run(run_name="xgboost_demand_forecast"):
        # 1. Load data
        try:
            df = pd.read_csv("/content/Data.csv")
        except FileNotFoundError:
            # Fall back to the path in your original code
            try:
                df = pd.read_csv("data\\transactions_20230103_20241231.csv")
            except FileNotFoundError:
                print("Error: Input data file not found. Please check the file path.")
                return

        # Log dataset info
        mlflow.log_param("dataset_path", "Data.csv")

        # 2. Create features and target on original data (keep for forecasting)
        original_df = extract_features(df.copy())
        original_df = create_target(original_df, horizon=7)
        original_df = original_df.dropna().reset_index(drop=True)

        # For training, work on a copy and then one-hot encode "Product Name"
        df_train = extract_features(df.copy())
        df_train = create_target(df_train, horizon=7)
        df_train = df_train.dropna().reset_index(drop=True)

        # Preserve original product name for evaluation
        df_train["Product"] = df_train["Product Name"]
        df_train = pd.get_dummies(df_train, columns=["Product Name"], prefix="prod")

        # Update feature_columns to include one-hot encoded product columns.
        product_columns = [col for col in df_train.columns if col.startswith("prod_")]
        other_features = [
            "lag_1", "lag_7", "lag_14", "lag_30",
            "rolling_mean_7", "rolling_mean_14", "rolling_std_7",
            "day_of_week", "is_weekend", "day_of_month", "day_of_year", "month", "week_of_year"
        ]
        feature_columns = other_features + product_columns
        target_column = "target"

        # Log dataset info
        mlflow.log_param("num_products", len(product_columns))
        mlflow.log_param("num_features", len(feature_columns))
        mlflow.log_param("dataset_size", len(df_train))

        # Automatic Time-Based Train/Validation/Test Split
        train_df, valid_df, test_df = get_train_valid_test_split(df_train, train_frac=0.7, valid_frac=0.1)
        mlflow.log_param("train_size", len(train_df))
        mlflow.log_param("valid_size", len(valid_df))
        mlflow.log_param("test_size", len(test_df))

        # Hyperparameter Tuning using Optuna
        print("Starting hyperparameter tuning with Optuna...")
        best_params, study = hyperparameter_tuning(train_df, valid_df, feature_columns, target_column, n_trials=5)

        # Log best parameters
        mlflow.log_params(best_params)

        # Train Final Model on Train+Validation
        train_valid_df = pd.concat([train_df, valid_df], ignore_index=True)
        final_model = model_training(train_valid_df, valid_df, feature_columns, target_column, params=best_params)

        # Log feature importances
        log_feature_importance(final_model, feature_columns)

        # SHAP Analysis
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(valid_df[feature_columns])

        # Create and save SHAP summary plot
        plt.figure()
        shap.summary_plot(shap_values, valid_df[feature_columns], show=False)
        plt.tight_layout()
        plt.savefig("shap_summary.png")
        mlflow.log_artifact("shap_summary.png")
        plt.close()

        # Compute metrics for all datasets
        # Training metrics
        train_pred = final_model.predict(train_df[feature_columns])
        train_rmse = compute_rmse(train_df[target_column], train_pred)
        train_mape = compute_mape(train_df[target_column], train_pred)

        # Validation metrics
        valid_pred = final_model.predict(valid_df[feature_columns])
        valid_rmse = compute_rmse(valid_df[target_column], valid_pred)
        valid_mape = compute_mape(valid_df[target_column], valid_pred)

        # Test metrics
        test_pred = final_model.predict(test_df[feature_columns])
        test_rmse = compute_rmse(test_df[target_column], test_pred)
        test_mape = compute_mape(test_df[target_column], test_pred)

        # Log all metrics
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("train_mape", train_mape)
        mlflow.log_metric("valid_rmse", valid_rmse)
        mlflow.log_metric("valid_mape", valid_mape)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("test_mape", test_mape)

        print("Training RMSE:", train_rmse)
        print("Validation RMSE:", valid_rmse)
        print("Test RMSE:", test_rmse)

        # Log actual vs predicted plots
        log_prediction_actual_scatter(train_df[target_column], train_pred, "Train")
        log_prediction_actual_scatter(valid_df[target_column], valid_pred, "Validation")
        log_prediction_actual_scatter(test_df[target_column], test_pred, "Test")

        # Log metrics comparison chart
        train_metrics = {"RMSE": train_rmse, "MAPE": train_mape}
        valid_metrics = {"RMSE": valid_rmse, "MAPE": valid_mape}
        test_metrics = {"RMSE": test_rmse, "MAPE": test_mape}
        log_metrics_comparison(train_metrics, valid_metrics, test_metrics)

        # Calculate RMSE per product
        test_df = test_df.copy()
        test_df["predicted"] = test_pred
        rmse_per_product = test_df.groupby("Product").apply(lambda d: compute_rmse(d[target_column], d["predicted"]))
        print("Individual RMSE per product:")
        print(rmse_per_product)

        # Log RMSE per product
        log_rmse_per_product(rmse_per_product)

        # Save and log the model
        mlflow.xgboost.log_model(final_model, "xgboost_model")

        # Iterative Forecasting
        products = original_df["Product Name"].unique()
        all_forecasts = []

        for product in products:
            df_product = original_df[original_df["Product Name"] == product].copy()
            if len(df_product) >= 60:
                fc = iterative_forecast(final_model, df_product, forecast_days=7, product_columns=product_columns)
                fc["Product Name"] = product
                all_forecasts.append(fc)

        all_forecasts_df = pd.concat(all_forecasts, ignore_index=True)
        print("7-day forecasts for each product:")
        print(all_forecasts_df)

        all_forecasts_df.to_csv("7_day_forecasts.csv", index=False)
        save_model(final_model)

        # Log the forecast CSV as an artifact
        mlflow.log_artifact("7_day_forecasts.csv")

if __name__ == "__main__":
    main()

#prophet
import pandas as pd
import numpy as np
from math import sqrt
import optuna
import pickle
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import mlflow
import mlflow.pyfunc
import matplotlib.pyplot as plt
import seaborn as sns

# =======================
# Utility Functions
# =======================

def compute_rmse(y_true, y_pred):
    """Computes Root Mean Squared Error."""
    return sqrt(mean_squared_error(y_true, y_pred))

def compute_mape(y_true, y_pred):
    """Computes Mean Absolute Percentage Error."""
    # Avoid division by zero
    mask = y_true != 0
    return mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100

# =======================
# MLflow Logging Functions
# =======================

def log_feature_importance(model, feature_columns):
    """Log feature importance plot to MLflow for Prophet model"""
    # Seasonality components are the "features" in Prophet
    # We'll check if the model has seasonality components
    components = ['trend']
    if model.seasonalities:
        components.extend(list(model.seasonalities.keys()))

    # Create dummy importance values (since Prophet doesn't have built-in feature importance)
    importances = [1.0] * len(components)  # Equal importance as a fallback

    # Create feature importance dataframe
    feature_importance = pd.DataFrame({
        'Feature': components,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Prophet Model Components')
    plt.tight_layout()

    # Save the figure and log it to MLflow
    plt.savefig('feature_importance.png')
    mlflow.log_artifact('feature_importance.png')
    plt.close()

def log_prediction_actual_scatter(y_true, y_pred, set_name):
    """Log scatter plot of actual vs predicted values"""
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5)

    # Add perfect prediction line
    max_val = max(y_true.max(), y_pred.max())
    min_val = min(y_true.min(), y_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')

    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Predicted vs Actual Values ({set_name})')
    plt.grid(True)
    plt.tight_layout()

    # Save the figure and log it to MLflow
    filename = f'actual_vs_predicted_{set_name}.png'
    plt.savefig(filename)
    mlflow.log_artifact(filename)
    plt.close()

def log_metrics_comparison(train_metrics, valid_metrics, test_metrics):
    """Log comparison of RMSE and MAPE across dataset splits"""
    metrics = ['RMSE', 'MAPE']
    datasets = ['Train', 'Validation', 'Test']

    # Create a dataframe for plotting
    data = {
        'Dataset': ['Train', 'Validation', 'Test', 'Train', 'Validation', 'Test'],
        'Metric': ['RMSE', 'RMSE', 'RMSE', 'MAPE', 'MAPE', 'MAPE'],
        'Value': [
            train_metrics['RMSE'], valid_metrics['RMSE'], test_metrics['RMSE'],
            train_metrics['MAPE'], valid_metrics['MAPE'], test_metrics['MAPE']
        ]
    }
    df = pd.DataFrame(data)

    # Create the grouped bar chart
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Metric', y='Value', hue='Dataset', data=df)
    plt.title('Performance Metrics Comparison')
    plt.tight_layout()

    # Save the figure and log it to MLflow
    plt.savefig('metrics_comparison.png')
    mlflow.log_artifact('metrics_comparison.png')
    plt.close()

def log_rmse_per_product(rmse_per_product):
    """Log RMSE per product as a bar chart"""
    plt.figure(figsize=(12, 6))
    rmse_per_product.sort_values().plot(kind='bar')
    plt.title('RMSE per Product')
    plt.ylabel('RMSE')
    plt.xlabel('Product')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the figure and log it to MLflow
    plt.savefig('rmse_per_product.png')
    mlflow.log_artifact('rmse_per_product.png')
    plt.close()

def log_hyperparameter_importances(study):
    """Log hyperparameter importance from Optuna study"""
    param_importances = optuna.importance.get_param_importances(study)

    # Convert to dataframe for plotting
    importance_df = pd.DataFrame({
        'Parameter': list(param_importances.keys()),
        'Importance': list(param_importances.values())
    }).sort_values('Importance', ascending=False)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Parameter', data=importance_df)
    plt.title('Hyperparameter Importance')
    plt.tight_layout()

    # Save the figure and log it to MLflow
    plt.savefig('hyperparameter_importance.png')
    mlflow.log_artifact('hyperparameter_importance.png')
    plt.close()

def log_hyperparameter_optimization_history(study):
    """Log the optimization history from Optuna study"""
    # Plot optimization history
    plt.figure(figsize=(10, 6))
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.tight_layout()

    # Save the figure and log it to MLflow
    plt.savefig('optimization_history.png')
    mlflow.log_artifact('optimization_history.png')
    plt.close()

    # Plot parallel coordinate plot
    plt.figure(figsize=(12, 6))
    optuna.visualization.matplotlib.plot_parallel_coordinate(study)
    plt.tight_layout()

    # Save the figure and log it to MLflow
    plt.savefig('parallel_coordinate.png')
    mlflow.log_artifact('parallel_coordinate.png')
    plt.close()

def log_prophet_components(model, forecast, product_name=None):
    """Log Prophet model components plot to MLflow"""
    fig = model.plot_components(forecast)
    plt.tight_layout()

    # Save the figure and log it to MLflow
    filename = f'prophet_components{"_" + product_name if product_name else ""}.png'
    plt.savefig(filename)
    mlflow.log_artifact(filename)
    plt.close()

# =======================
# 1. Feature Engineering for Prophet
# =======================

def prepare_prophet_data(df: pd.DataFrame, product_name=None) -> pd.DataFrame:
    """
    Prepare data for Prophet model by renaming columns to 'ds' and 'y'.
    Optionally filters data for a specific product.
    """
    df = df.copy()
    df["ds"] = pd.to_datetime(df["Date"])
    df["y"] = df["Total Quantity"]

    # Filter for specific product if provided
    if product_name:
        df = df[df["Product Name"] == product_name].copy()

    # Add additional regressors that might be useful
    df["day_of_week"] = df["ds"].dt.dayofweek  # Monday=0, Sunday=6
    df["is_weekend"] = df["day_of_week"].isin([5,6]).astype(int)

    return df[["ds", "y", "Product Name", "day_of_week", "is_weekend"]]

def prepare_prophet_future(dates, product_name=None):
    """Prepare future dataframe for Prophet forecast"""
    future = pd.DataFrame({"ds": dates})

    if product_name:
        future["Product Name"] = product_name

    # Add additional regressors
    future["day_of_week"] = future["ds"].dt.dayofweek
    future["is_weekend"] = future["day_of_week"].isin([5,6]).astype(int)

    return future

# ==========================================
# 2. Automatic Train/Validation/Test Split
# ==========================================

def get_train_valid_test_split(df: pd.DataFrame, train_frac: float = 0.7, valid_frac: float = 0.1):
    """
    Automatically splits the DataFrame into train, validation, and test sets based on the date range.
    """
    unique_dates = df["ds"].drop_duplicates().sort_values()
    n = len(unique_dates)

    train_cutoff = unique_dates.iloc[int(n * train_frac)]
    valid_cutoff = unique_dates.iloc[int(n * (train_frac + valid_frac))]

    print("Train cutoff date:", train_cutoff)
    print("Validation cutoff date:", valid_cutoff)

    train_df = df[df["ds"] < train_cutoff].copy()
    valid_df = df[(df["ds"] >= train_cutoff) & (df["ds"] < valid_cutoff)].copy()
    test_df  = df[df["ds"] >= valid_cutoff].copy()

    return train_df, valid_df, test_df

# ==========================================
# 3. Hyperparameter Tuning using Optuna
# ==========================================

def hyperparameter_tuning(train_df, valid_df, n_trials: int = 100):
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())

    # Define the objective function with mlflow tracking
    def objective_with_mlflow(trial, train_df, valid_df):
        with mlflow.start_run(nested=True):
            param = {
                "changepoint_prior_scale": trial.suggest_float("changepoint_prior_scale", 0.001, 0.5, log=True),
                "seasonality_prior_scale": trial.suggest_float("seasonality_prior_scale", 0.01, 10, log=True),
                "holidays_prior_scale": trial.suggest_float("holidays_prior_scale", 0.01, 10, log=True),
                "seasonality_mode": trial.suggest_categorical("seasonality_mode", ["additive", "multiplicative"]),
                "changepoint_range": trial.suggest_float("changepoint_range", 0.8, 0.95),
                "yearly_seasonality": trial.suggest_categorical("yearly_seasonality", [True, False, 'auto']),
                "weekly_seasonality": trial.suggest_categorical("weekly_seasonality", [True, False, 'auto']),
                "daily_seasonality": trial.suggest_categorical("daily_seasonality", [True, False, 'auto']),
            }

            # Add extra regressors flag
            use_weekend_regressor = trial.suggest_categorical("use_weekend_regressor", [True, False])

            # Log parameters
            mlflow.log_params(param)
            mlflow.log_param("use_weekend_regressor", use_weekend_regressor)

            # Initialize the Prophet model with the parameters
            model = Prophet(**param)

            # Add regressors if specified
            if use_weekend_regressor:
                model.add_regressor('is_weekend')

            # Fit the model on training data
            model.fit(train_df[["ds", "y"] + (["is_weekend"] if use_weekend_regressor else [])])

            # Create validation future dataframe
            future = valid_df[["ds"] + (["is_weekend"] if use_weekend_regressor else [])]

            # Make predictions
            forecast = model.predict(future)

            # Extract predictions and actual values
            valid_pred = forecast["yhat"].values
            valid_true = valid_df["y"].values

            # Calculate metrics
            rmse = compute_rmse(valid_true, valid_pred)
            mape = compute_mape(valid_true, valid_pred)

            mlflow.log_metric("val_rmse", rmse)
            mlflow.log_metric("val_mape", mape)

            print(f"Trial {trial.number}: RMSE = {rmse}, MAPE = {mape}")
            return rmse

    study.optimize(lambda trial: objective_with_mlflow(trial, train_df, valid_df), n_trials=n_trials)

    print("Best trial:")
    trial = study.best_trial
    for key, value in trial.params.items():
        print(f"  {key}: {value}")

    # Visualize and log hyperparameter optimization results
    log_hyperparameter_importances(study)
    log_hyperparameter_optimization_history(study)

    return trial.params, study

# ==========================================
# 4. Model Training Function
# ==========================================

def model_training(train_df: pd.DataFrame, params: dict = None, regressor_cols=None):
    """
    Trains a Prophet model using the specified parameters.

    Parameters:
    - train_df: DataFrame with columns 'ds' (datetime) and 'y' (target variable)
    - params: Dictionary of Prophet parameters
    - regressor_cols: List of columns to use as regressors

    Returns:
    - Trained Prophet model
    """
    if params is None:
        params = {
            "changepoint_prior_scale": 0.05,
            "seasonality_prior_scale": 10.0,
            "holidays_prior_scale": 10.0,
            "seasonality_mode": "additive",
            "yearly_seasonality": True,
            "weekly_seasonality": True,
            "daily_seasonality": False,
        }

    if regressor_cols is None:
        regressor_cols = []

    model = Prophet(**params)

    # Add all regressors
    for col in regressor_cols:
        if col in train_df.columns and col not in ['ds', 'y']:
            model.add_regressor(col)

    # Fit the model
    model.fit(train_df[['ds', 'y'] + regressor_cols])

    return model

# ==========================================
# 5. Forecasting Function
# ==========================================

def forecast_future(model, df: pd.DataFrame, days_ahead: int = 7, include_history: bool = True, regressor_cols=None):
    """
    Generate forecast for future dates using a trained Prophet model.

    Parameters:
    - model: Trained Prophet model
    - df: DataFrame with historical data including 'ds' and regressor columns
    - days_ahead: Number of days to forecast ahead
    - include_history: Whether to include historical dates in the forecast
    - regressor_cols: List of regressor columns to include in future DataFrame

    Returns:
    - DataFrame with forecast
    """
    if regressor_cols is None:
        regressor_cols = []

    # Get the latest date in the data
    last_date = df['ds'].max()

    # Create future dataframe
    if include_history:
        # Include historical dates plus future dates
        future_dates = pd.date_range(
            start=df['ds'].min(),
            end=last_date + pd.Timedelta(days=days_ahead),
            freq='D'
        )
    else:
        # Only include future dates
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=days_ahead,
            freq='D'
        )

    future = pd.DataFrame({'ds': future_dates})

    # Add regressor values for future dates
    for col in regressor_cols:
        if col == 'is_weekend':
            future[col] = future['ds'].dt.dayofweek.isin([5, 6]).astype(int)
        elif col == 'day_of_week':
            future[col] = future['ds'].dt.dayofweek
        else:
            # For other regressors, try to extrapolate from historical data
            # This is a simplified approach - in practice, you might need more sophisticated methods
            if col in df.columns:
                # For demonstration, use the mean value from history
                future[col] = df[col].mean()

    # Generate forecast
    forecast = model.predict(future)

    # If we have actual values, add them to the forecast
    if include_history:
        actual_data = df[['ds', 'y']].rename(columns={'y': 'actual'})
        forecast = pd.merge(forecast, actual_data, on='ds', how='left')

    return forecast

# ==========================================
# 6. Model Saving Function
# ==========================================

class ProphetPyfuncModel(mlflow.pyfunc.PythonModel):
    def __init__(self, prophet_model, regressor_cols=None):
        self.prophet_model = prophet_model
        self.regressor_cols = regressor_cols or []

    def predict(self, context, model_input):
        # Ensure model_input has the required columns
        future = model_input.copy()

        if 'ds' not in future.columns and 'Date' in future.columns:
            future['ds'] = pd.to_datetime(future['Date'])

        # Make sure all regressors are present
        for col in self.regressor_cols:
            if col not in future.columns:
                if col == 'is_weekend':
                    future[col] = future['ds'].dt.dayofweek.isin([5, 6]).astype(int)
                elif col == 'day_of_week':
                    future[col] = future['ds'].dt.dayofweek

        # Generate predictions
        predictions = self.prophet_model.predict(future)
        return predictions[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

def save_model(model, regressor_cols=None):
    """Save Prophet model for later use"""
    with open("final_model.pkl", "wb") as f:
        pickle.dump({'model': model, 'regressor_cols': regressor_cols}, f)

    print("Model saved as final_model.pkl")

# ==========================================
# 7. Main Pipeline
# ==========================================

def main():
    # Set up MLflow tracking
    experiment_name = "LSTM_vs_XGBoost_Comparison"
    mlflow.set_experiment(experiment_name)
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    with mlflow.start_run(run_name="xgboost_demand_forecast"):
        # 1. Load data
        try:
            df = pd.read_csv("/content/Data.csv")
        except FileNotFoundError:
            # Fall back to the path in your original code
            try:
                df = pd.read_csv("data\\transactions_20230103_20241231.csv")
            except FileNotFoundError:
                print("Error: Input data file not found. Please check the file path.")
                return

        # Log dataset info
        mlflow.log_param("dataset_path", "Data.csv")

        # Prepare data for Prophet
        prophet_data = prepare_prophet_data(df)

        # Store original data for per-product forecasting
        all_products = prophet_data["Product Name"].unique()

        # Log dataset info
        mlflow.log_param("num_products", len(all_products))
        mlflow.log_param("num_features", 3)  # ds, y, and potential regressors
        mlflow.log_param("dataset_size", len(prophet_data))

        # Automatic Time-Based Train/Validation/Test Split
        train_df, valid_df, test_df = get_train_valid_test_split(prophet_data, train_frac=0.7, valid_frac=0.1)
        mlflow.log_param("train_size", len(train_df))
        mlflow.log_param("valid_size", len(valid_df))
        mlflow.log_param("test_size", len(test_df))

        # Hyperparameter Tuning using Optuna
        print("Starting hyperparameter tuning with Optuna...")
        best_params, study = hyperparameter_tuning(train_df, valid_df, n_trials=5)

        # Extract regressor columns based on Optuna results
        use_weekend_regressor = best_params.pop('use_weekend_regressor', False)
        regressor_cols = ['is_weekend'] if use_weekend_regressor else []

        # Log best parameters
        mlflow.log_params(best_params)
        mlflow.log_param("use_weekend_regressor", use_weekend_regressor)

        # Train Final Model on Train+Validation
        train_valid_df = pd.concat([train_df, valid_df], ignore_index=True)
        final_model = model_training(train_valid_df, params=best_params, regressor_cols=regressor_cols)

        # Generate forecasts for evaluation
        train_forecast = forecast_future(final_model, train_df, days_ahead=0, include_history=True, regressor_cols=regressor_cols)
        valid_forecast = forecast_future(final_model, valid_df, days_ahead=0, include_history=True, regressor_cols=regressor_cols)
        test_forecast = forecast_future(final_model, test_df, days_ahead=0, include_history=True, regressor_cols=regressor_cols)

        # Log Prophet component plots
        log_prophet_components(final_model, train_forecast)

        # Log feature importances (adapted for Prophet)
        log_feature_importance(final_model, [])

        # Compute metrics for all datasets
        # Training metrics
        train_true = train_forecast['actual'].dropna().values
        train_pred = train_forecast.loc[train_forecast['actual'].notna(), 'yhat'].values
        train_rmse = compute_rmse(train_true, train_pred)
        train_mape = compute_mape(train_true, train_pred)

        # Validation metrics
        valid_true = valid_forecast['actual'].dropna().values
        valid_pred = valid_forecast.loc[valid_forecast['actual'].notna(), 'yhat'].values
        valid_rmse = compute_rmse(valid_true, valid_pred)
        valid_mape = compute_mape(valid_true, valid_pred)

        # Test metrics
        test_true = test_forecast['actual'].dropna().values
        test_pred = test_forecast.loc[test_forecast['actual'].notna(), 'yhat'].values
        test_rmse = compute_rmse(test_true, test_pred)
        test_mape = compute_mape(test_true, test_pred)

        # Log all metrics
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("train_mape", train_mape)
        mlflow.log_metric("valid_rmse", valid_rmse)
        mlflow.log_metric("valid_mape", valid_mape)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("test_mape", test_mape)

        print("Training RMSE:", train_rmse)
        print("Validation RMSE:", valid_rmse)
        print("Test RMSE:", test_rmse)

        # Log actual vs predicted plots
        log_prediction_actual_scatter(train_true, train_pred, "Train")
        log_prediction_actual_scatter(valid_true, valid_pred, "Validation")
        log_prediction_actual_scatter(test_true, test_pred, "Test")

        # Log metrics comparison chart
        train_metrics = {"RMSE": train_rmse, "MAPE": train_mape}
        valid_metrics = {"RMSE": valid_rmse, "MAPE": valid_mape}
        test_metrics = {"RMSE": test_rmse, "MAPE": test_mape}
        log_metrics_comparison(train_metrics, valid_metrics, test_metrics)

        # Per-product forecasting and evaluation
        all_forecasts = []
        product_rmse = {}

        for product in all_products:
            print(f"Processing product: {product}")

            # Filter data for this product
            product_data = prophet_data[prophet_data["Product Name"] == product].copy()

            # Skip products with insufficient data
            if len(product_data) < 60:
                print(f"Skipping {product} - insufficient data: {len(product_data)} records")
                continue

            # Split data
            product_train, product_valid, product_test = get_train_valid_test_split(
                product_data, train_frac=0.7, valid_frac=0.1
            )

            # Train product-specific model
            product_model = model_training(
                pd.concat([product_train, product_valid]),
                params=best_params,
                regressor_cols=regressor_cols
            )

            # Evaluate on test data
            product_test_forecast = forecast_future(
                product_model,
                product_test,
                days_ahead=0,
                include_history=True,
                regressor_cols=regressor_cols
            )

            # Calculate metrics
            product_test_true = product_test_forecast['actual'].dropna().values
            product_test_pred = product_test_forecast.loc[product_test_forecast['actual'].notna(), 'yhat'].values

            if len(product_test_true) > 0:
                product_rmse[product] = compute_rmse(product_test_true, product_test_pred)

            # Generate 7-day forecast
            future_forecast = forecast_future(
                product_model,
                product_data,
                days_ahead=7,
                include_history=False,
                regressor_cols=regressor_cols
            )

            # Format forecast for output
            future_forecast["Product Name"] = product
            future_forecast = future_forecast.rename(columns={
                "ds": "Date",
                "yhat": "Predicted Quantity"
            })[["Date", "Product Name", "Predicted Quantity"]]

            # Round predictions to whole numbers
            future_forecast["Predicted Quantity"] = future_forecast["Predicted Quantity"].round()

            all_forecasts.append(future_forecast)

        # Combine all forecasts
        if all_forecasts:
            all_forecasts_df = pd.concat(all_forecasts, ignore_index=True)
            print("7-day forecasts for each product:")
            print(all_forecasts_df)

            all_forecasts_df.to_csv("7_day_forecasts.csv", index=False)

            # Log the forecast CSV as an artifact
            mlflow.log_artifact("7_day_forecasts.csv")

        # Log RMSE per product
        if product_rmse:
            rmse_per_product = pd.Series(product_rmse)
            log_rmse_per_product(rmse_per_product)

        # Save and log the model
        prophet_pyfunc = ProphetPyfuncModel(final_model, regressor_cols)
        mlflow.pyfunc.log_model(
            "prophet_model",
            python_model=prophet_pyfunc,
            registered_model_name="prophet_demand_forecast"
        )

        # Save the model to disk
        save_model(final_model, regressor_cols)

if __name__ == "__main__":
    main()
