the simplest rag system but it shows that it can be build without complexity with open source libraries.
many more features and optimisations can be build on top but the concept is simple enough.
many things are left to do, like reorganising the repo, etc but the core functions are there

# RAG Postgres with Docker

This repository sets up a PostgreSQL database with the [pgvector](https://github.com/pgvector/pgvector) extension using Docker Compose. The setup is intended for projects that require vector search capabilities, such as those using OpenAI embeddings.

## Purpose

Demonstrate the simplest implementation of RAG with an open source vector store.

## Features

- PostgreSQL database with the `pgvector` extension for vector similarity search.
- Easy local development using Docker Compose.
- Persistent storage using Docker volumes.

## Getting Started

### Prerequisites

- [Docker](https://www.docker.com/)
- [Docker Compose](https://docs.docker.com/compose/)

### Setup

1. **Clone the repository:**
   ```sh
   git clone <your-repo-url>
   
   ```

2. **Create .env using the .env-example files**
Use these files as examples to create `.env` files for both source code and docker container.

3. **Launch the database**
```
cd rag-posgres/docker
docker compose --env-file .env -d up
```

4. **Run the code**
```
cd rag-postgres/
python3 src/main.py
```

### Text Search Implementation

## Semantic Search
## Keyword Search
I implemented different approaches to text search, as explained below together with their performance characteristics.

## Regular Keyword Matching (LIKE)

Regular keyword matching using the `LIKE` operator has several limitations, namely it is case sensitive, scanninng through database records is sequential, and it cannot be indexed.
Hence, it is not suitable for large datasets due to poor scalability.
Postgres however, has build-in methods for full-text search.

### Tokenization Process

**Document Processing:**
- `ts_vector()` tokenizes database documents by:
  - Removing stopwords (common words like "the", "and", "is")
  - Stripping punctuation
  - Reducing words to their base forms (lexemes)
  - Adding positional information for each lexeme in the sentence

**Query Processing:**
- Use `to_tsquery()` to tokenize search queries in the same format

### Performance Optimization

To improve search performance and scalability:

1. **Precompute tokenization** - Process documents ahead of time rather than during search
2. **Store as separate column** - Add a dedicated `tsvector` column to your table
3. **Add index** - Create a GIN or GiST index on the tsvector column

This approach transforms the search from a sequential scan to an indexed lookup, making it much more scalable for large datasets.

### Result Ranking

PostgreSQL provides two ranking functions to prioritize search results:

- **`ts_rank`** - Ranks based on the number of matching words
- **`ts_rank_cd`** - Enhanced ranking that also considers the proximity of matching words (closer matches rank higher)

Both functions help determine which results are most relevant to display first.

## Hybrid Search
Combining the results of the above two methods and re-ranking, as each of the methods uses different criteria.
Optionally, one can enforce the limit of results after joining those from both methods.
Given sufficient records in the database, the semantic search will always return as many documents as the indicated `limit` number. That's not the case for keyword search, which can return less than the limit threshold of documents.