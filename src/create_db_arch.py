import os
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import Vector, register_vector
import numpy as np

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


def create_keyword_index(cursor):
    # create a tsvector column for full-text search
    # if it doesn't already exist
    # this column is created from the content column
    # it is a generated column, so it is always up to date
    # the keyword STORED means that the values of this column
    # are stored on disk, not computed on the fly
    cursor.execute(
        """
                   ALTER TABLE data
                   ADD COLUMN IF NOT EXISTS search_vector tsvector 
                   GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;
                   """
    )

    # add an index to the tsvector column
    # if it doesn't already exist
    cursor.execute(
        """CREATE INDEX IF NOT EXISTS search_idx
                     ON data using GIN (search_vector);"""
    )
    return cursor


def query_db(cursor, query):
    response = cursor.execute(query)
    return response.fetchall()


def get_embedding(client, text_to_embed):
    # Embed a line of text

    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text_to_embed,
        encoding_format="float",
    )
    # Extract the AI output embedding as a list of floats
    embedding = response.data[0].embedding

    return embedding


def semantic_search(client, cursor, query, limit):
    # you have to convert the query embedding into a string for the query to work
    query_embedding = get_embedding(client, query)
    query_embedding = Vector(query_embedding)

    cursor.execute(
        "SELECT id, content, embedding <=> %s AS distance FROM data ORDER BY distance limit %s;",
        (query_embedding, limit),
    )

    # get all the results
    # `distances` is a list of tuples
    # each tuple is (id, embedding, distance)
    distances = cursor.fetchall()
    return distances


def keyword_search(cursor, query, limit=5):
    # create the keyword index if it doesn't exist
    cursor = create_keyword_index(cursor)
    query = "go"
    query = f"""
    SELECT id, content, ts_rank(to_tsvector('english', content), websearch_to_tsquery('{query}')) as rank
    from data 
    where search_vector @@ websearch_to_tsquery('{query}')
    order by rank desc
    limit {limit};
    """

    response = cursor.execute(query)

    # get all the results
    # `response.fetchall()` is a list of tuples
    # each tuple is (id, content, rank)
    return response.fetchall()


def hybrid_search(client, cursor, query, limit=5, enforce_limit=True):
    """
    Combine keyword search and semantic search

    Args:
    client: OpenAI client
        client to use to get embeddings
    cursor: psycopg cursor
        cursor to use to query the database
    query: str
        query to search for in the database
    limit: int
        number of results to return from each search method

    Returns:
    list of unique content from both search methods
    """

    # combine keyword search and semantic search
    keyword_results = keyword_search(cursor, query, limit=limit)
    semantic_results = semantic_search(client, cursor, query, limit=limit)

    # convert to numpy arrays to merge easily
    # add a column to indicate the search method
    semantic_results = np.array(semantic_results)
    semantic_results = np.column_stack(
        (
            semantic_results,
            np.array(["semantic_search"] * semantic_results.shape[0]),
        )
    )
    keyword_results = np.array(keyword_results)
    keyword_results = np.column_stack(
        (
            keyword_results,
            np.array(["keyword_search"] * keyword_results.shape[0]),
        )
    )

    # as long as you have enough records in your database,
    # semantic search will always return `limit` results
    # but keyword search might return less than `limit` results
    # if there are not enough matches. In that case,
    # return only the semantic search results
    if len(keyword_results) == 0:
        results = semantic_results
    else:
        results = np.hstack(
            (semantic_results[:, 1], keyword_results[:, 1]),
        ).tolist()

    # remove duplicates and return as a list
    results = np.unique(results).tolist()

    # if you want to enforce the limit, return only the first `limit` results
    if enforce_limit:
        results = results[:limit]

    return results
