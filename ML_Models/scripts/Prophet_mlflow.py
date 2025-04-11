from prophet import Prophet
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mlflow
import os

# ==== Load and prepare data ====
df = pd.read_excel("/Users/sheethunaik/Documents/Forecast_MLOPS/SupplyChainOptimization/Supermarket Transactions New.xlsx")
df.columns = df.columns.str.strip()
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])  # remove rows where date is invalid
df['Quantity'] = df['Quantity'].astype(int)
df = df.sort_values(['Product Name', 'Date'])

products = df['Product Name'].unique()

# ==== Prophet + MLflow ====
mlflow.set_tracking_uri("http://localhost:5001")

experiment_id = "245297520303277004"

with mlflow.start_run(experiment_id=experiment_id, run_name="prophet_demand_model"):
    results = []

    for product in products:
        product_df = df[df['Product Name'] == product].copy()
        product_df = product_df.groupby('Date')['Quantity'].sum().reset_index()
        product_df = product_df.rename(columns={'Date': 'ds', 'Quantity': 'y'})

        train_size = int(len(product_df) * 0.8)
        train_df = product_df.iloc[:train_size]
        test_df = product_df.iloc[train_size:]

        model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        model.fit(train_df)

        future = model.make_future_dataframe(periods=len(test_df))
        forecast = model.predict(future)
        forecast_test = forecast.iloc[-len(test_df):]

        rmse = np.sqrt(mean_squared_error(test_df['y'], forecast_test['yhat']))
        results.append({'Product': product, 'RMSE': rmse})

        fig = model.plot(forecast)
        plt.title(f"{product} Forecast vs Actual")
        fig_path = f"forecast_{product}.png"
        fig.savefig(fig_path)
        plt.close()

        mlflow.log_artifact(fig_path)

    # Save RMSE results and log average RMSE
    results_df = pd.DataFrame(results)
    avg_rmse = results_df['RMSE'].mean()
    print(f"✅ Prophet Avg RMSE across products: {avg_rmse:.2f}")
    mlflow.log_metric("avg_rmse", avg_rmse)

    results_df.to_csv("prophet_results.csv", index=False)
    mlflow.log_artifact("prophet_results.csv")
