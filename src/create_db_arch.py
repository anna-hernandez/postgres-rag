import os
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import Vector, register_vector


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

    register_vector(connection)

    cursor = connection.cursor()
    return cursor


def create_table(cursor):
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


def insert_embeddings(cursor, sources_list):
    """
    Args:
    sources_list: list(dict)
        list of source data to store
    """

    # establish index. there are two at the moment in pgvector
    # hnsw and ivf flat
    # specify the index to use and how to calculate the distance
    cursor.execute("CREATE INDEX ON data USING hnsw (embedding vector_cosine_ops);")

    # insert embeddings into table
    # they need to be numpy arrays

    for item in sources_list:
        cursor.execute(
            "INSERT INTO data (id, content, metadata, embedding) VALUES (%s,%s,%s,%s)",
            (
                item["id"],
                item["content"],
                item["metadata"]["created_at"],
                item["embedding"],
            ),
        )


def query_db(cursor, query):
    response = cursor.execute(query)
    return response.fetchall()


def semantic_search(cursor, query_embedding, limit):
    # you have to convert the query embedding into a string for the query to work
    query_embedding = Vector(query_embedding)

    # cursor.execute(
    #     f"SELECT * FROM data ORDER BY embedding <=> %s LIMIT %s;",
    #     (query_embedding, limit),
    # )
    # results = cursor.fetchall()
    cursor.execute(
        "SELECT id, content, embedding <=> %s AS distance FROM data ORDER BY distance limit %s;",
        (query_embedding, limit),
    )
    distances = cursor.fetchall()
    print("Number of rows:", len(distances))
    print("All rows:", distances)
    return distances
