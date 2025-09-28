from pgvector.psycopg import Vector
from uuid import uuid4
import numpy as np
from datetime import datetime


def get_embeddings(client, texts):
    # Embed a line of text

    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=texts,
        encoding_format="float",
    )
    # Extract the AI output embedding as a list of floats
    embeddings = [i.embedding for i in response.data]

    return embeddings


def get_llm_response(client, query):
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
