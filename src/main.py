from dotenv import load_dotenv
from openai import OpenAI
import os
from uuid import uuid4
from datetime import datetime
from create_db_arch import (
    create_table,
    insert_embeddings,
    create_cursor,
    query_db,
    semantic_search,
    keyword_search,
    get_embedding,
    hybrid_search,
)
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


def format_content(text, embedding):
    return {
        "id": str(uuid4()),
        "content": text,
        "metadata": {
            "created_at": datetime.now().isoformat(),
        },
        "embedding": Vector(embedding),
    }


if __name__ == "__main__":

    # 0. prepare data
    # data is a list of strings
    create_db = False
    if len(sys.argv) > 1:
        if sys.argv[1] == "--create_db":
            create_db = True

    # crete connection (cursor) to database
    # if --create_db flag is passed, create the database and insert the data
    # otherwise, just connect to the database
    cursor = create_cursor()
    if create_db:
        print("Creating database...")

        # for each text in the input data, create its embeddings
        # store embeddings and metadata into database
        sources = []
        for text in data:
            embedding = get_embedding(client, text)

            # formatted_content is a dictionary
            formatted_content = format_content(text, embedding)
            sources.append(formatted_content)

        # create database table
        create_table(cursor)
        # insert embeddings in database
        # formatted_content is a list of tuples
        # each tuple is (id, content, metadata, embedding)
        insert_embeddings(cursor, sources)

    # 1. retrieve response

    query = "select * from data;"
    response = query_db(cursor, query)
    print("\n\nSQL query")
    print("----------------------")
    print(response)

    query = "i go to the cinema"

    print("\n\nKeyword search")
    print("----------------------")
    # TODO: set rank threshold
    response = keyword_search(cursor, query, limit=2)
    for row in response:
        print("id:", row[0], "distance:", row[1])

    print("\n\nSemantic search distances")
    print("----------------------")
    response = semantic_search(client, cursor, query, 2)
    for row in response:
        print("id:", row[0], "distance:", row[1])

    print("\n\nHybrid search")
    print("----------------------")
    response = hybrid_search(client, cursor, query, limit=2, enforce_limit=True)
    print(response)

    # 2. augment original query
    query += f"\nRelevant data:"
    for idx, item in enumerate(response):
        print("id:", item)
        query += f"\n{item}"

    print("\n\nAugmented query")
    print("----------------------")
    print(query)

    response = get_llm_response(
        query=query,
    )
    print("\n\nResponse")
    print("----------------------")

    print(response)
