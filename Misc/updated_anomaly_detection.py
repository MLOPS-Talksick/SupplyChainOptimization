# from logger import logger
import polars as pl
from typing import Dict, Tuple
import numpy as np
import pandas as pd


def calculate_zscore(series: pl.Series) -> pl.Series:
    """Calculate Z-score for a Polars Series."""
    try:
        mean = series.mean()
        std = series.std()
        if std == 0 or std is None:
            return pl.Series([0] * len(series))
        return (series - mean) / std
    except Exception as e:
        # print(f"Error calculating Z-score: {e}")
        raise


def iqr_bounds(series: pl.Series) -> Tuple[float, float]:
    """Calculate IQR-based lower and upper bounds for anomaly detection."""
    try:
        if series.is_empty():
            raise ValueError("Cannot compute IQR for an empty series.")

        q1 = np.percentile(series, 25)
        q3 = np.percentile(series, 75)
        iqr = q3 - q1

        if iqr == 0:
            return q1, q3

        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (2.0 * iqr)
        if series.min() >= 0:
            lower_bound = max(0, lower_bound)
        print(f"Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")
        return lower_bound, upper_bound
    except Exception as e:
        print(f"Error calculating IQR bounds: {e}")
        raise


def detect_anomalies(
    df: pl.DataFrame,
) -> Tuple[Dict[str, pl.DataFrame], pl.DataFrame]:
    """
    Detect anomalies in transaction data per product per day using IQR checks,
    odd time-of-day checks, and invalid-format checks.
    Returns:
      - Dictionary of anomaly types → DataFrames
      - Clean DataFrame with anomalies removed
    """
    anomalies = {}
    clean_df = df.clone()
    anomaly_transaction_ids = set()

    try:
        df = df.with_columns(
            pl.col("Date").cast(pl.Datetime).alias("datetime")
        )

        # Extract date and hour from datetime
        df = df.with_columns([
            pl.col("datetime").dt.date().alias("date_only"),
            pl.col("datetime").dt.hour().alias("hour"),
        ])

        # 1. Price Anomalies
        price_anomalies = []
        product_date_combinations = df.select(
            ["Product Name", "date_only"]
        ).unique()

        for row in product_date_combinations.iter_rows(named=True):
            product = row["Product Name"]
            date = row["date_only"]
            subset = df.filter(
                (pl.col("Product Name") == product)
                & (pl.col("date_only") == date)
            )

            if "Unit Price" in df.columns:
                if len(subset) >= 4:
                    lower_bound, upper_bound = iqr_bounds(subset["Unit Price"])
                    iqr_anoms = subset.filter(
                        (pl.col("Unit Price") < lower_bound)
                        | (pl.col("Unit Price") > upper_bound)
                    )
                    if len(iqr_anoms) > 0:
                        price_anomalies.append(iqr_anoms)
                        anomaly_transaction_ids.update(
                            iqr_anoms["Transaction ID"].to_list()
                        )

        anomalies["price_anomalies"] = (
            pl.concat(price_anomalies) if price_anomalies else pl.DataFrame()
        )
        print(f"Price anomalies detected: {len(price_anomalies)} sets.")

        # 2. Quantity Anomalies
        quantity_anomalies = []
        for row in product_date_combinations.iter_rows(named=True):
            product = row["Product Name"]
            date = row["date_only"]
            subset = df.filter(
                (pl.col("Product Name") == product)
                & (pl.col("date_only") == date)
            )

            if len(subset) >= 4:
                print(f"Processing product: {product}, date: {date}")
                lower_bound, upper_bound = iqr_bounds(subset["Quantity"])
                iqr_anoms = subset.filter(
                    (pl.col("Quantity") < lower_bound)
                    | (pl.col("Quantity") > upper_bound)
                )
                if len(iqr_anoms) > 0:
                    quantity_anomalies.append(iqr_anoms)
                    anomaly_transaction_ids.update(
                        iqr_anoms["Transaction ID"].to_list()
                    )

        anomalies["quantity_anomalies"] = (
            pl.concat(quantity_anomalies)
            if quantity_anomalies
            else pl.DataFrame()
        )
        print(
            f"Quantity anomalies detected: {len(quantity_anomalies)} sets."
        )

        df = df.with_columns(
            (pl.col("hour") != 0).alias("has_time")
        )

        # Proceed if any datetime has a time component
        if df.filter(pl.col("has_time") == True).shape[0] > 0:
            # Detect time anomalies
            time_anomalies = df.filter(
                (pl.col("hour") < 6) | (pl.col("hour") > 22)
            )

            # Store anomalies in a dictionary
            anomalies["time_anomalies"] = time_anomalies
            anomaly_transaction_ids.update(time_anomalies["Transaction ID"].to_list())

            print(f"Time anomalies detected: {len(time_anomalies)} transactions.")
        else:
            print("No time component found in the Date column, skipping anomaly detection.")

        # # 3. Time-of-day Anomalies
        # time_anomalies = df.filter(
        #     (pl.col("hour") < 6) | (pl.col("hour") > 22)
        # )
        # anomalies["time_anomalies"] = time_anomalies
        # anomaly_transaction_ids.update(
        #     time_anomalies["Transaction ID"].to_list()
        # )
        # print(time_anomalies.head())
        # print(
        #     f"Time anomalies detected: {len(time_anomalies)} transactions."
        # )

        # 4. Invalid Format Checks
        format_anomalies = df.filter((pl.col("Quantity") <= 0))
        anomalies["format_anomalies"] = format_anomalies
        anomaly_transaction_ids.update(
            format_anomalies["Transaction ID"].to_list()
        )
        print(
            f"Format anomalies detected: {len(format_anomalies)} transactions."
        )

        # Filter out anomaly transactions from the clean DataFrame
        clean_df = clean_df.filter(
            ~pl.col("Transaction ID").is_in(list(anomaly_transaction_ids))
        )
        print(
            f"Clean data size after anomaly removal: {clean_df.shape[0]} rows."
        )

    except Exception as e:
        print(f"Error detecting anomalies: {e}")
        raise

    return anomalies, clean_df




def standardize_date_format(
    df: pl.DataFrame, date_column: str = "Date"
) -> pl.DataFrame:
    """
    Standardizes date formats in the given column, handling multiple
    date string formats and casting them to a consistent datetime type.
    """
    try:
        print("Standardizing date formats...")

        if date_column not in df.columns:
            print(f"Column '{date_column}' not found in DataFrame.")
            return df

        # Make sure we're working with string representations first
        df = df.with_columns(pl.col(date_column).cast(pl.Utf8))

        # Handle empty DataFrame
        if df.is_empty():
            return df

        # Create a more comprehensive date conversion
        df = df.with_columns(
            pl.when(
                pl.col(date_column).str.contains(
                    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(\.\d+)?$"
                )
            )  # 2019-01-03 08:46:08
            .then(
                pl.col(date_column).str.strptime(
                    pl.Datetime, "%Y-%m-%d %H:%M:%S%.3f", strict=False
                )
            )
            .when(
                pl.col(date_column).str.contains(r"^\d{4}-\d{2}-\d{2}$")
            )  # 2019-01-03
            .then(
                pl.col(date_column).str.strptime(
                    pl.Datetime, "%Y-%m-%d", strict=False
                )
            )
            .when(
                pl.col(date_column).str.contains(r"^\d{2}-\d{2}-\d{4}$")
            )  # 01-03-2019
            .then(
                pl.col(date_column).str.strptime(
                    pl.Datetime, "%m-%d-%Y", strict=False
                )
            )
            .when(
                pl.col(date_column).str.contains(r"^\d{2}/\d{2}/\d{4}$")
            )  # 03/01/2019
            .then(
                pl.col(date_column).str.strptime(
                    pl.Datetime, "%d/%m/%Y", strict=False
                )
            )
            .when(
                pl.col(date_column).str.contains(r"^\d{1,2}/\d{1,2}/\d{4}$")
            )  # 3/1/2019
            .then(
                pl.col(date_column).str.strptime(
                    pl.Datetime, "%m/%d/%Y", strict=False
                )
            )
            .when(
                pl.col(date_column).str.contains(r"^\d{1,2}-\d{1,2}-\d{4}$")
            )  # 3-1-2019
            .then(
                pl.col(date_column).str.strptime(
                    pl.Datetime, "%m-%d-%Y", strict=False
                )
            )
            .otherwise(None)
            .alias(date_column)
        )

        # Try to convert any null values with additional formats
        null_mask = df[date_column].is_null()
        if null_mask.sum() > 0:
            print(
                f"Some date values could not be parsed: {null_mask.sum()} nulls"
            )

            # Try one more time with the pandas parser which is more flexible
            try:
                # Convert to pandas, apply conversion, then back to polars
                temp_df = df.filter(null_mask).to_pandas()
                if not temp_df.empty:
                    temp_df[date_column] = pd.to_datetime(
                        temp_df[date_column], errors="coerce"
                    )
                    temp_pl = pl.from_pandas(temp_df)

                    # Update only the previously null values
                    df = df.with_columns(
                        pl.when(null_mask)
                        .then(pl.lit(temp_pl[date_column]))
                        .otherwise(pl.col(date_column))
                        .alias(date_column)
                    )
            except Exception as e:
                print(f"Additional date parsing attempt failed: {e}")

        null_count = df[date_column].null_count()
        if null_count > 0:
            print(
                f"Date column '{date_column}' has {null_count} null values after conversion. Check data format."
            )

        print("Date standardization completed successfully.")
        return df
    except Exception as e:
        print(f"Unexpected error during date processing: {e}")
        # Return original DataFrame rather than failing
        return df


df = pl.read_excel("C:/Users/svaru/Downloads/Supermarket Transactions New.xlsx")

# df = standardize_date_format(df, date_column="Date")

print("Detecting Anomalies...")
anomalies, df = detect_anomalies(df)

# Check if any anomalies were found
if any(not anom.is_empty() for anom in anomalies.values()):
    print("Sending an Email for Alert...")
    try:
        # send_anomaly_alert(anomalies=anomalies)
        print("Anomaly alert sent successfully.")
    except Exception as e:
        print(f"Failed to send anomaly alert: {e}")
else:
    print("No anomalies detected. No email sent.")

# Check if DataFrame became empty after anomaly detection
if df.is_empty():
    print(
        "DataFrame became empty after anomaly detection. Skipping further processing."
    )