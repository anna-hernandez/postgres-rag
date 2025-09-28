import psycopg
from pgvector.psycopg import Vector, register_vector
import numpy as np
from utils import get_embeddings


class DatabaseConnection:
    def __init__(
        self,
        db,
        user,
        password,
        host="localhost",
        port=5432,
    ):
        """Initialize the DatabaseConnection with connection parameters.
        Args:
            db (str): Database name
            user (str): Database user
            password (str): Database password
            host (str, optional): Database host. Defaults to "localhost".
            port (int, optional): Database port. Defaults to 5432.
        """
        self.dbname = db
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.cursor = self.connect()

    def connect(self):
        self.connection = psycopg.connect(
            dbname=self.dbname,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
        )
        self.connection.autocommit = True
        register_vector(self.connection)
        self.cursor = self.connection.cursor()

        return self.cursor

    def create_table(self):
        # install pgvector extension
        self.cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")

        # create the table, including embedding column
        # TODO: move table name to settings
        self.cursor.execute("DROP TABLE IF EXISTS data")

        self.cursor.execute(
            """CREATE TABLE data (
                    id text PRIMARY KEY,
                    content text,
                    metadata text,
                    embedding vector(1536));"""
        )

        return self.cursor

    def insert_embeddings(self, sources_list):
        """
        Args:
        sources_list: list(dict)
            list of source data to store
        """

        # establish index. there are two at the moment in pgvector
        # hnsw and ivf flat
        # specify the index to use and how to calculate the distance
        self.cursor.execute(
            "CREATE INDEX ON data USING hnsw (embedding vector_cosine_ops);"
        )

        # insert embeddings into table
        # they need to be numpy arrays

        for item in sources_list:
            self.cursor.execute(
                "INSERT INTO data (id, content, metadata, embedding) VALUES (%s,%s,%s,%s)",
                (
                    item["id"],
                    item["content"],
                    item["created_at"],
                    item["embedding"],
                ),
            )

    def create_keyword_index(self):
        # create a tsvector column for full-text search
        # if it doesn't already exist
        # this column is created from the content column
        # it is a generated column, so it is always up to date
        # the keyword STORED means that the values of this column
        # are stored on disk, not computed on the fly
        self.cursor.execute(
            """
                    ALTER TABLE data
                    ADD COLUMN IF NOT EXISTS search_vector tsvector 
                    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;
                    """
        )

        # add an index to the tsvector column
        # if it doesn't already exist
        self.cursor.execute(
            """CREATE INDEX IF NOT EXISTS search_idx
                        ON data using GIN (search_vector);"""
        )
        return self.cursor

    def query_db(self, query):
        """
        Args:
        query: str
            SQL query to execute
        Returns:
        list of tuples
            query results
        """
        response = self.cursor.execute(query)
        return response.fetchall()

    def semantic_search(self, client, query, limit):

        # convert query to embedding
        query_embedding = get_embeddings(client, query)
        # `get_embeddings` returns a list of embeddings
        # but for `query_embedding` we only have one query
        # so we take the first element of the list
        # and convert it to a Vector
        query_embedding = Vector(query_embedding[0])

        # query the database for the closest embeddings
        # using the <=> operator to calculate the distance
        # and ordering the results by distance
        # limit the results to the specified number
        # TODO: move table and column names to settings
        # TODO: move query to settings
        self.cursor.execute(
            "SELECT id, content, embedding <=> %s AS distance FROM data ORDER BY distance limit %s;",
            (query_embedding, limit),
        )

        # get all the results as a list of tuples
        # each tuple is (id, embedding, distance)
        return self.cursor.fetchall()

    def keyword_search(self, query, limit=5):
        # create the keyword index if it doesn't exist
        cursor = self.create_keyword_index()
        # search the database for the query
        # using full-text search
        # and ordering the results by rank
        # limit the results to the specified number
        # TODO: move table and column names to settings
        # TODO: move query to settings
        query = f"""
        SELECT id, content, ts_rank(to_tsvector('english', content), websearch_to_tsquery('{query}')) as rank
        from data 
        where search_vector @@ websearch_to_tsquery('{query}')
        order by rank desc
        limit {limit};
        """

        self.cursor.execute(query)

        # get all the results
        # `response.fetchall()` is a list of tuples
        # each tuple is (id, content, rank)
        return self.cursor.fetchall()

    def hybrid_search(self, client, query, limit=5, enforce_limit=True):
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
        keyword_results = self.keyword_search(query, limit=limit)
        semantic_results = self.semantic_search(client, query, limit=limit)

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
