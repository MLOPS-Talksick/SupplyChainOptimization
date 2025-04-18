import pandas as pd
import pandas as pd
from google.cloud.sql.connector import Connector
import sqlalchemy
import os
from dotenv import load_dotenv
from sqlalchemy import text

load_dotenv()

host = os.getenv("MYSQL_HOST")
user = os.getenv("MYSQL_USER")
password = os.getenv("MYSQL_PASSWORD")
database = os.getenv("MYSQL_DATABASE")
instance = os.getenv("INSTANCE_CONN_NAME", "primordial-veld-450618-n4:us-central1:mlops-sql")

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
            ip_type="PRIVATE",
        )
        return conn

    pool = sqlalchemy.create_engine(
        "mysql+pymysql://",  # or "postgresql+pg8000://" for PostgreSQL, "mssql+pytds://" for SQL Server
        creator=getconn,
    )

    df = pd.read_sql(query, pool)
    connector.close()
    return df


def upsert_df(df: pd.DataFrame, table_name: str):
    """
    Inserts or updates rows in a MySQL table based on duplicate keys.
    If a record with the same primary key exists, it will be replaced with the new record.
    
    Parameters:
        df (pd.DataFrame): The DataFrame to insert/update.
        table_name (str): The target table name.
        engine: SQLAlchemy engine.
    """

    connector = Connector()

    def getconn():
        conn = connector.connect(
            instance,
            "pymysql",  # Database driver
            user=user,  # Database user
            password=password,  # Database password
            db=database,
            ip_type="PRIVATE",
        )
        return conn

    pool = sqlalchemy.create_engine(
        "mysql+pymysql://",  # or "postgresql+pg8000://" for PostgreSQL, "mssql+pytds://" for SQL Server
        creator=getconn,
    )

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
    with pool.begin() as conn:
        conn.execute(sql, data)