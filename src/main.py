from dotenv import load_dotenv
from openai import OpenAI
import os
from uuid import uuid4
from datetime import datetime

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


data = [
    "this is a sample sentence",
    "this is another sample sentence",
    "and yet another sample sentence",
]


def get_embedding(text_to_embed):
    # Embed a line of text

    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input="The food was delicious and the waiter...",
        encoding_format="float",
    )
    # Extract the AI output embedding as a list of floats
    embedding = response.data[0].embedding

    return embedding


def format_content(text, embedding):
    return {
        "id": str(uuid4()),
        "content": text,
        "metadata": {
            "created_at": datetime.now().isoformat(),
        },
        "embedding": embedding,
    }


if __name__ == "__main__":

    for text in data:
        embedding = get_embedding(text)
        formatted_content = format_content(text, embedding)
        print(formatted_content)
