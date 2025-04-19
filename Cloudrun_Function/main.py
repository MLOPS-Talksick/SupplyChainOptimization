import functions_framework
import os
import pandas as pd
from google.cloud import storage
import sqlalchemy
import io
from google.cloud.sql.connector import Connector, IPTypes
from sqlalchemy.dialects.mysql import insert
import pymysql
from sqlalchemy import text
import numpy as np
import requests
import json
from dotenv import load_dotenv

load_dotenv()

def call_model_retrain():
    url = os.environ.get("TRIGGER_TRAINING_URL")

    data = {
        "PROJECT_ID": os.environ.get("PROJECT_ID"),
        "REGION": os.environ.get("REGION"),
        "BUCKET_URI": os.environ.get("BUCKET_URI"),
        "IMAGE_URI": os.environ.get("MODEL_TRAINING_IMAGE_URI")
    }
    
    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(url, data=json.dumps(data), headers=headers)
    return response.text, response.status_code

def interpolate_missing_dates_for_product(df, product_name, last_date_sql):
    # Filter the dataframe for the specific product
    product_df = df[df['Product Name'] == product_name].copy()
    
    # Ensure 'date' column is in datetime format
    product_df['date'] = pd.to_datetime(product_df['date'])
    
    # Get the first date in the product's df and the last date in the SQL
    first_date_df = product_df['date'].min()
    last_date_sql = pd.to_datetime(last_date_sql)
    
    # Create a complete date range between the two dates
    full_date_range = pd.date_range(start=last_date_sql + pd.Timedelta(days=1), end=first_date_df - pd.Timedelta(days=1))
    
    # Create a DataFrame for missing dates
    missing_dates = pd.DataFrame({'date': full_date_range})
    
    # Forward fill, backward fill, and average both methods for total_quantity
    missing_dates['forward_fill'] = missing_dates['date'].apply(
        lambda x: product_df.loc[product_df['date'] <= x, 'total_quantity'].ffill().iloc[-1] if not product_df[product_df['date'] <= x].empty else np.nan)
    
    missing_dates['backward_fill'] = missing_dates['date'].apply(
        lambda x: product_df.loc[product_df['date'] >= x, 'total_quantity'].bfill().iloc[0] if not product_df[product_df['date'] >= x].empty else np.nan)
    
    # Average of both forward and backward fills
    missing_dates['interpolated_quantity'] = missing_dates[['forward_fill', 'backward_fill']].mean(axis=1)
    
    # Add product name to the missing data
    missing_dates['Product Name'] = product_name
    
    return missing_dates[['date', 'Product Name', 'interpolated_quantity']]

def get_db_connection() -> sqlalchemy.engine.base.Engine:
    """
    Initializes a connection pool for a Cloud SQL instance of MySQL.

    Uses the Cloud SQL Python Connector package.
    """
    db_user = os.environ.get("MYSQL_USER")
    db_pass = os.environ.get("MYSQL_PASSWORD")
    db_name = os.environ.get("MYSQL_DATABASE")
    instance_connection_name = os.environ.get("INSTANCE_CONN_NAME")
    
    # ip_type = IPTypes.PRIVATE if os.environ.get("PRIVATE_IP") else IPTypes.PUBLIC
    ip_type = IPTypes.PRIVATE

    # initialize Cloud SQL Python Connector object
    connector = Connector(ip_type=ip_type, refresh_strategy="LAZY")

    def getconn() -> pymysql.connections.Connection:
        conn: pymysql.connections.Connection = connector.connect(
            instance_connection_name,
            "pymysql",
            user=db_user,
            password=db_pass,
            db=db_name,
        )
        return conn

    pool = sqlalchemy.create_engine(
        "mysql+pymysql://",
        creator=getconn,
        # ...
    )
    return pool


def upsert_df(df: pd.DataFrame, table_name: str, engine):
    """
    Inserts or updates rows in a MySQL table based on duplicate keys.
    If a record with the same primary key exists, it will be replaced with the new record.
    
    Parameters:
        df (pd.DataFrame): The DataFrame to insert/update.
        table_name (str): The target table name.
        engine: SQLAlchemy engine.
    """
    # Convert DataFrame to a list of dictionaries (each dict represents a row)
    data = df.to_dict(orient='records')
    
    # Build dynamic column list and named placeholders
    columns = df.columns.tolist()
    col_names = ", ".join(columns)
    placeholders = ", ".join(":" + col for col in columns)
    
    # Build the update clause to update every column with its new value
    update_clause = ", ".join(f"{col} = VALUES({col})" for col in columns)
    
    # Construct the SQL query using ON DUPLICATE KEY UPDATE
    sql = text(
        f"INSERT INTO {table_name} ({col_names}) VALUES ({placeholders}) "
        f"ON DUPLICATE KEY UPDATE {update_clause}"
    )
    
    # Execute the query in a transactional scope
    with engine.begin() as conn:
        conn.execute(sql, data)

def create_database(engine):

    product_sql = text(f"CREATE TABLE IF NOT EXISTS PRODUCT ("
        f"    product_name VARCHAR(255) PRIMARY KEY"
        f");"
    )


    sales_sql = text(f"CREATE TABLE IF NOT EXISTS SALES ("
        f"    sale_date DATE NOT NULL,"
        f"    product_name VARCHAR(255) NOT NULL,"
        f"    total_quantity INTEGER NOT NULL,"
        f"    PRIMARY KEY (sale_date, product_name),"
        f"    FOREIGN KEY (product_name) REFERENCES PRODUCT(product_name)"
        f");"
    )

    predict_sql = text(f"CREATE TABLE IF NOT EXISTS PREDICT ("
        f"    sale_date DATE NOT NULL,"
        f"    product_name VARCHAR(255) NOT NULL,"
        f"    total_quantity INTEGER NOT NULL,"
        f"    PRIMARY KEY (sale_date, product_name),"
        f"    FOREIGN KEY (product_name) REFERENCES PRODUCT(product_name)"
        f");"
    )


    with engine.begin() as conn:
        conn.execute(product_sql)
        conn.execute(sales_sql)
        conn.execute(predict_sql)


@functions_framework.cloud_event
def hello_gcs(cloud_event):
    data = cloud_event.data

    bucket_name = data["bucket"]
    file_name = data["name"]
    
    # if not file_name.lower().endswith('.csv'):
    #     print(f"Skipping non-CSV file: {file_name}")
    #     return

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)

    data = blob.download_as_bytes()
    df = pd.read_csv(io.BytesIO(data))
    
    # Ensure correct data types
    # df['date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df['Date'] = df['Date'].str.strip()
    df['sale_date'] = pd.to_datetime(df['Date'])

    # Database connection
    engine = get_db_connection()

    try:

        create_database(engine)

        query = "SELECT product_name FROM PRODUCT"
        product_names = pd.read_sql(query, engine)

        # Insert into PRODUCT table (if not exists)
        product_df = df[['Product Name']].drop_duplicates()
        product_df.rename(columns={'Product Name': 'product_name'}, inplace=True)

        print(product_df.head(10))
        upsert_df(product_df, 'PRODUCT', engine)
        
        # # # Insert into SALES table
        sales_df = df[['sale_date', 'Product Name', 'Total Quantity']].drop_duplicates()
        sales_df.rename(columns={'Product Name': 'product_name', 'Total Quantity': 'total_quantity',}, inplace=True)
        #     sales_df.to_sql(
        #         'Sales', 
        #         connection, 
        #         if_exists='append', 
        #         index=False, 
        #         method=insert_on_conflict_update_sales
        #     )
        upsert_df(sales_df, 'SALES', engine)
            
        print(f"Successfully processed {file_name}")

        # Find missing values
        new_products = product_df[~product_df['product_name'].isin(product_names['product_name'])]

        print("New products:", new_products['product_name'].tolist())
        try:
            response_text, response_status_code = call_model_retrain()

            if response_status_code != 200:
                raise Exception(f"Model Training API call failed with status code {response_status_code}: {response_text}")

            print(response_text)

        except Exception as e:
            print(f"Error: {e}")
            raise

        blob.delete()


    
    except Exception as e:
        print(f"Error processing {file_name}: {str(e)}")
        return False