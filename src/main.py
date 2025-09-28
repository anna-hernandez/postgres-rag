from dotenv import load_dotenv
from openai import OpenAI
import os
from uuid import uuid4
from datetime import datetime
from create_db_arch import DatabaseConnection, get_embeddings
import numpy as np
import sys
from pgvector.psycopg import Vector

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


data = [
    "today we went to the cinema",
    "yesterday we went to the park",
    "tomorrow we'll go to the pool",
    "my brother called a week ago",
]


def get_llm_response(query):
    response = client.responses.create(
        model="gpt-4o-mini",
        input=query,
    )

    return response.output_text


def format_content(texts, embeddings):
    """formats the content to be inserted into the database.
    Args:
        texts (list): list of strings
        embeddings (list): list of lists of floats
    Returns:
        list: list of dictionaries with keys: id, content,
        created_at, embedding
    """

    ids = [str(uuid4()) for _ in texts]
    embedding_vectors = [Vector(e) for e in embeddings]
    dates = [datetime.now().isoformat()] * len(texts)

    return [
        {"id": i, "content": t, "created_at": d, "embedding": e}
        for (i, t, d, e) in zip(ids, texts, dates, embedding_vectors)
    ]


def main(create_db, query_type="hybrid"):

    # create connection (cursor) to database
    # the connection parameters are read from environment variables
    # the cursor is created regardless of whether the database is created or not
    # the cursor is used to execute SQL queries
    cursor = DatabaseConnection(
        db=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host=os.getenv("POSTGRES_HOST"),
        port=5432,
    )

    # if --create_db flag is passed, create the database and insert the data
    # (otherwise, just connect to the database)
    if create_db:

        # create database table
        print("Createing database table...")

        cursor.create_table()

        print("Getting source embeddings...")

        # for each text in the input data, create its embeddings
        # store embeddings and metadata into database
        embeddings = get_embeddings(client, data)

        # formatted_content is a dictionary
        formatted_content = format_content(data, embeddings)

        # insert embeddings in database
        # formatted_content is a list of tuples
        # each tuple is (id, content, metadata, embedding)
        cursor.insert_embeddings(formatted_content)
    else:
        print("Connecting to existing database...")

    # 1. retrieve response
    if query_type == "sql":
        print("Running SQL query...")
        # simple SQL query
        query = "select * from data;"
        response = cursor.query_db(query)

    elif query_type in ("keyword", "semantic", "hybrid"):
        query = "go"
        if query_type == "keyword":
            print("Running keyword search...")
            # TODO: set rank threshold
            response = cursor.keyword_search(query, limit=2)

        elif query_type == "semantic":
            print("Running semantic search...")
            # TODO: set distance threshold
            response = cursor.semantic_search(client, query, 2)

        elif query_type == "hybrid":
            print("Running hybrid search...")
            response = (
                cursor.hybrid_search(
                    client,
                    query,
                    limit=2,
                    enforce_limit=True,
                ),
            )
        # TODO: set distance threshold
        # TODO: log this
        for row in response:
            print("id:", row[0], "content:", row[1], "distance:", row[2])

        # 2. augment original query
        print("Building augmented query...")
        query += f"\nRelevant data:"
        for idx, item in enumerate(response):
            print("id:", item[1])
            query += f"\n{item[1]}"
        # TODO: log this
        print(query)

        response = get_llm_response(
            query=query,
        )
    print("\nResponse:")
    print(response)


if __name__ == "__main__":
    # 0. prepare data
    # data is a list of strings
    create_db = False
    query_type = "hybrid"
    if len(sys.argv) > 1:
        if sys.argv[1] == "--create_db":
            create_db = True
        else:
            query_type = sys.argv[1][2:]

    main(create_db, query_type)
