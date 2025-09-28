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
Hence, it is not suitable for large datasets due to poor scalability. Postgres however, has build-in methods for full-text search.

The full-text search is based on finding matches between tokenized texts.
There are two tokenization processes:
1. `ts_vector()` tokenizes database documents by removing stopwords, punctuation, getting the lexeme of words and adding a reference to their position in the sentence.
2. `to_tsquery()` to tokenize search queries in the same format. There are actually a few variations of this function (e.g. `painto_tsquery()`, `websearch_to_tsquery()`, etc.) with distinct nuances

To improve search performance and scalability, we avoid computing the vectors on the fly. We precompute them instead. 
1. Process all documents with `to_tsvector`
2. Create a new column in the database with the created vectors
3. Add a GIN or GiST index on the vectors column
4. When running the search, run it against the indexed column, not the original documents

This approach transforms the search from a sequential scan to an indexed lookup, making it much more scalable for large datasets.

Additionally, you can rank the results based on similarity to the query. There are two main functions in PostgreSQL:

- **`ts_rank`**, which ranks based on the number of matching words
- **`ts_rank_cd`**, which is an enhanced ranking that also considers the proximity of matching words (closer matches rank higher)

Here I used `ts_rank` for simplicity.

## Hybrid Search
Combining the results of the above two methods and re-ranking, as each of the methods uses different criteria.
Optionally, one can enforce the limit of results after joining those from both methods.
Given sufficient records in the database, the semantic search will always return as many documents as the indicated `limit` number. That's not the case for keyword search, which can return less than the limit threshold of documents.

There are two input parameters:
- **`--create_db`**: boolen that, if True, indicates that the database needs to be built from scratch given some input sources; if False, otherwise, and hence it is only used for querying
- **`--query_tybe`**: string indicating the type of search, options are `sql` (for a regular, non-text based, SQL query), `semantic`, `keyword`, `hybrid`

Future improvements
- one could decide to use an agent to decide which type of query is more appropriate at each time (sql, semantic, keyword, hybrid)
- add logging
- store answers in database for future review. If app, add button with thumbs up and down to record user reply. use already computed answers as cache
- use structure outputs