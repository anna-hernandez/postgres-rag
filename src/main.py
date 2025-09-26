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


def get_embedding(text_to_embed):
    # Embed a line of text

    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text_to_embed,
        encoding_format="float",
    )
    # Extract the AI output embedding as a list of floats
    embedding = response.data[0].embedding

    return embedding


def get_llm_response(query):
    response = client.responses.create(
        model="gpt-4o-mini",
        input=query,
    )

    return response.output_text


def format_content(text, embedding):
    # return {
    #     "id": str(uuid4()),
    #     "content": text,
    #     "metadata": {
    #         "created_at": datetime.now().isoformat(),
    #     },
    #     "embedding": embedding,
    # }

    return (
        str(uuid4()),
        text,
        datetime.now().isoformat(),
        Vector(embedding),
    )


if __name__ == "__main__":

    # # for each text in the input data, create its embeddings
    # # store embeddings and metadata into database
    # sources = []
    # for text in data:
    #     embedding = get_embedding(text)
    #     formatted_content = format_content(text, embedding)
    #     sources.append(formatted_content)

    # create database
    cursor = create_cursor()
    # create_table(cursor)
    # # insert embeddings in database
    # # formatted_content is a list of tuples
    # # each tuple is (id, content, metadata, embedding)
    # insert_embeddings(cursor, sources)

    # 1. retrieve response

    query = "select * from data;"
    response = query_db(cursor, query)
    print("\n\nRegular SQL query")
    print("----------------------")
    print(response)

    query = "What do i like to do?"
    query_embedding = get_embedding(query)
    distances = semantic_search(cursor, query_embedding, 2)
    print("\n\nSemantic search distances")
    print("----------------------")
    for row in distances:
        print("id:", row[0], "content:", row[1], "distance:", row[2])

    # 2. augment original query
    query += f"\nRelevant data:"
    for idx, item in enumerate(distances):
        print("id:", item[0], "content:", item[1], "distance:", item[2])
        query += f"\n{item[1]}"

    print("\n\nAugmented query")
    print("----------------------")
    print(query)

    response = get_llm_response(
        query=query,
    )
    print("\n\nResponse")
    print("----------------------")

    print(response)
