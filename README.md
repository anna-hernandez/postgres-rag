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
