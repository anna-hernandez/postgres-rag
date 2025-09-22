import os
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
from psycopg.rows import dict_row


load_dotenv()

# define env variables
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_DB = os.getenv("POSTGRES_DB")
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_USER = os.getenv("POSTGRES_USER")


def create_cursor():

    # establish connection with database
    connection = psycopg.connect(
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        host=POSTGRES_HOST,
    )
    connection.autocommit = True
    cursor = connection.cursor()

    # install pgvector extension
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # create the table, including embedding column
    cursor.execute("DROP TABLE IF EXISTS data")

    cursor.execute(
        """CREATE TABLE data (
                id text PRIMARY KEY,
                content text,
                metadata text,
                embedding vector(1536));"""
    )

    return cursor


def insert_embeddings(cursor, data_tuples):
    

    # establish index. there are two at the moment in pgvector
    # hnsw and ivf flat
    # specify the index to use and how to calculate the distance
    cursor.execute("CREATE INDEX ON data USING hnsw (embedding vector_l2_ops);")

    # insert embeddings into table
    # they need to be numpy arrays
    for dt in data_tuples:
        cursor.execute(
            "INSERT INTO data (id, content, metadata, embedding) VALUES (%s, %s, %s, %s)",
            dt,
        )

def query_db():
    cursor = 