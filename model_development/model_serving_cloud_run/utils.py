import pandas as pd
import pandas as pd
from google.cloud.sql.connector import Connector
import sqlalchemy
import os
# from dotenv import load_dotenv

# load_dotenv()

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

    host = os.getenv("MYSQL_HOST")
    user = os.getenv("MYSQL_USER")
    password = os.getenv("MYSQL_PASSWORD")
    database = os.getenv("MYSQL_DATABASE")
    connector = Connector()

    def getconn():
        conn = connector.connect(
            "primordial-veld-450618-n4:us-central1:mlops-sql",  # Cloud SQL instance connection name
            "pymysql",  # Database driver
            user=user,  # Database user
            password=password,  # Database password
            db=database,
        )
        return conn

    pool = sqlalchemy.create_engine(
        "mysql+pymysql://",  # or "postgresql+pg8000://" for PostgreSQL, "mssql+pytds://" for SQL Server
        creator=getconn,
    )

    df = pd.read_sql(query, pool)
    connector.close()
    return df