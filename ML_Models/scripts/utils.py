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
import io
from datetime import datetime, timedelta
from google.cloud.sql.connector import Connector
import sqlalchemy
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv
from google.cloud import storage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_training_utils')

load_dotenv()


host = os.getenv("MYSQL_HOST")
user = os.getenv("MYSQL_USER")
password = os.getenv("MYSQL_PASSWORD")
database = os.getenv("MYSQL_DATABASE")
instance = os.getenv("INSTANCE_CONN_NAME")
email = "talksick530@gmail.com"

def extracting_time_series_and_lagged_features_pd(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each row, compute additional time-series features:
      - day_of_week, is_weekend, day_of_month, day_of_year, month, week_of_year
      - lag_1, lag_7, and rolling_mean_7 of 'Total Quantity'
    """
    # If the DataFrame is empty, return an empty DataFrame with the expected columns.
    if df.empty:
        return pd.DataFrame(
            columns=[
                "Date",
                "Product Name",
                "Total Quantity",
                "day_of_week",
                "is_weekend",
                "day_of_month",
                "day_of_year",
                "month",
                "week_of_year",
                "lag_1",
                "lag_7",
                "rolling_mean_7",
            ]
        )

    # Ensure the 'Date' column is in datetime format
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])

        # Create date-based features
        df["day_of_week"] = df["Date"].dt.dayofweek  # Monday=0, Sunday=6
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df["day_of_month"] = df["Date"].dt.day
        df["day_of_year"] = df["Date"].dt.dayofyear
        df["month"] = df["Date"].dt.month
        # Using isocalendar to get week of year (as integer)
        df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
    else:
        return df

    # Check if 'Total Quantity' exists and then compute lag and rolling features
    if "Total Quantity" in df.columns:
        # Sort by product and date
        df = df.sort_values(by=["Product Name", "Date"]).reset_index(drop=True)

        # Compute lag features by grouping by 'Product Name'
        df["lag_1"] = df.groupby("Product Name")["Total Quantity"].shift(1)
        df["lag_7"] = df.groupby("Product Name")["Total Quantity"].shift(7)

        # Compute a rolling mean of the previous 7 days.
        # Note: We shift by 1 to ensure that the rolling window uses data strictly before the current row.
        df["rolling_mean_7"] = df.groupby("Product Name")[
            "Total Quantity"
        ].transform(
            lambda x: x.shift(1).rolling(window=7, min_periods=1).mean()
        )
    else:
        raise KeyError("Column 'Total Quantity' not found in DataFrame.")

    return df



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
            ip_type = "PRIVATE",
        )
        return conn

    pool = sqlalchemy.create_engine(
        "mysql+pymysql://",  # or "postgresql+pg8000://" for PostgreSQL, "mssql+pytds://" for SQL Server
        creator=getconn,
    )
    with pool.connect() as db_conn:
        result = db_conn.execute(sqlalchemy.text(query))
        print(result.scalar())
    df = pd.read_sql(query, pool)
    print(df.head())
    connector.close()
    return df


def get_cloud_sql_connection():
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
            ip_type = "PRIVATE",

        )
        return conn

    pool = sqlalchemy.create_engine(
        "mysql+pymysql://",  # or "postgresql+pg8000://" for PostgreSQL, "mssql+pytds://" for SQL Server
        creator=getconn,
    )

    return pool



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

    # Check for large date gaps
    time_gaps = []
    for product in df['Product Name'].unique():
        product_df = df[df['Product Name'] == product].sort_values('Date')
        if len(product_df) > 1:
            date_diffs = product_df['Date'].diff().dt.days.iloc[1:]
            max_gap = date_diffs.max()
            if max_gap > 30:  # Flag gaps larger than 30 days
                time_gaps.append((product, max_gap))

    return {
        'product_counts': product_counts,
        'product_time_coverage': product_time_coverage,
        'monthly_distribution': monthly_distribution,
        'weekly_distribution': weekly_distribution,
        'time_gaps': time_gaps
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

    time_gaps = analysis_results['time_gaps']
    products, gaps = zip(*time_gaps)
    # Plotting the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(products, gaps, color='skyblue')
    plt.xlabel('Product Name')
    plt.ylabel('Maximum Gap (days)')
    plt.title('Maximum Time Gaps Between Products (Gaps > 30 Days)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_gaps.png'))

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
            ip_type = "PRIVATE",
        )
        return conn

    pool = sqlalchemy.create_engine(
        "mysql+pymysql://",  # or "postgresql+pg8000://" for PostgreSQL, "mssql+pytds://" for SQL Server
        creator=getconn,
    )
    df = pd.read_sql(query, pool)
    connector.close()
    return df




def get_connection():
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
            ip_type = "PRIVATE",
        )
        return conn

    pool = sqlalchemy.create_engine(
        "mysql+pymysql://",  # or "postgresql+pg8000://" for PostgreSQL, "mssql+pytds://" for SQL Server
        creator=getconn,
    )
    return pool

def send_email(
    emailid,
    body,
    subject="Automated Email",
    smtp_server="smtp.gmail.com",
    smtp_port=587,
    sender="talksick530@gmail.com",
    username="talksick530@gmail.com",
    password="celm dfaq qllh ymjv",
    attachment=None,
):
    """
    Sends an email to the given email address with a message body.
    If an attachment (pandas DataFrame) is provided, it will be converted to CSV and attached.

    Parameters:
      emailid (str): Recipient email address.
      body (str): Email text content.
      subject (str): Subject of the email.
      smtp_server (str): SMTP server address.
      smtp_port (int): SMTP server port.
      sender (str): Sender's email address.
      username (str): Username for SMTP login.
      password (str): Password for SMTP login.
      attachment (str or pd.DataFrame, optional): File path or DataFrame to attach.
    """
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = emailid
    msg.set_content(body)

    if attachment is not None:
        if isinstance(attachment, pd.DataFrame):
            # Attach DataFrame as CSV
            csv_buffer = io.StringIO()
            attachment.to_csv(csv_buffer, index=False)
            csv_bytes = csv_buffer.getvalue().encode("utf-8")
            msg.add_attachment(csv_bytes, maintype="text", subtype="csv", filename="anomalies.csv")
        elif isinstance(attachment, str):  # Assume it's a file path
            with open(attachment, "rb") as f:
                file_data = f.read()
                file_name = attachment.split("/")[-1]
                msg.add_attachment(file_data, maintype="application", subtype="octet-stream", filename=file_name)

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            if username and password:
                server.login(username, password)
            server.send_message(msg)
        logger.info(f"Email sent successfully to: {emailid}")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        raise



def upload_to_gcs(file_name, gcs_bucket_name):
    # Initialize Google Cloud Storage client
    storage_client = storage.Client()

    # Upload Keras model
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
