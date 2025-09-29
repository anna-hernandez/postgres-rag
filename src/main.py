from dotenv import load_dotenv
from openai import OpenAI
import os
from datetime import datetime
from database import DatabaseConnection
from utils import get_embeddings, format_content, get_llm_response
import sys

load_dotenv()
# TODO: check where is best to instantiate the client
# it could be here, or inside the functions that need it
# or it could be passed as an argument to the functions that need it
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


data = [
    "today we went to the cinema",
    "yesterday we went to the park",
    "tomorrow we'll go to the pool",
    "my brother called a week ago",
]


def rag(create_db, query, query_type="hybrid"):

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
            print("response: ", response)

        # 2. augment original query
        print("Building augmented query...")
        query += f"\nRelevant data:"
        for idx, item in enumerate(response):
            query += f"\n{item}"
        # TODO: log this
        print(query)

        response = get_llm_response(
            client=client,
            query=query,
        )
    print("\nResponse:")
    return response


if __name__ == "__main__":
    # 0. prepare data
    # data is a list of strings
    # TODO: load data from file(s)
    # TODO: load data from url(s)
    # TODO: check if data is already in database
    # instead of relying on the user to pass the --create_db flag
    create_db = False
    query_type = "hybrid"
    if len(sys.argv) > 1:
        if sys.argv[1] == "--create_db":
            create_db = True
        else:
            query_type = sys.argv[1][2:]

    query = "What did we do yesterday?"
    rag(create_db, query, query_type)
