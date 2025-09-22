import os
import psycopg2
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector


load_dotenv()

# define env variables
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_DB = os.getenv("POSTGRES_DB")
POSTGRES_HOST = os.getenv("POSTGRES_HOST")

connection = pyscopg2.connect(
    database=POSTGRES_DB, password=POSTGRES_PASSWORD, host=POSTGRES_HOST
)
