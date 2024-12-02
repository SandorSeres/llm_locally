import os
import json
import logging
from datetime import datetime
from io import BytesIO
from typing import Optional, List

from fastapi import UploadFile
from neo4j import GraphDatabase

from langchain_core.documents import Document

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader as DocxLoader,
)

import numpy as np

class Neo4jRAG:
    def __init__(self, url: str = None, username: str = None, password: str = None, max_pool_size: int = 10):
        self.url = url or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.username = username or os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD")
        self.embedding = OpenAIEmbeddings()
        self.driver = GraphDatabase.driver(
            self.url,
            auth=(self.username, self.password),
            max_connection_pool_size=max_pool_size
        )
        self.db = Neo4jVector(
            url=self.url,
            username=self.username,
            password=self.password,
            embedding=self.embedding,
            index_name="docstore",
        )
        logging.info("Neo4jRAG initialized successfully")

    def close(self):
        self.driver.close()

    def execute_query(self, query: str, parameters: Optional[dict] = None) -> list:
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return [record for record in result]

    # Session handling and persist
    def save_session(self, session_id: str, session_data: dict):
        query = """
        MERGE (s:Session {id: $session_id})
        SET s.history = $history, s.updated_at = $updated_at
        """
        parameters = {
            "session_id": session_id,
            "history": json.dumps(session_data["history"]),  # Convert to JSON string
            "updated_at": datetime.utcnow().isoformat()
        }
        self.execute_query(query, parameters)
        logging.info("Successfully saved session")

    def get_session(self, session_id: str) -> Optional[dict]:
        query = """
        MATCH (s:Session {id: $session_id})
        RETURN s.history AS history
        """
        parameters = {"session_id": session_id}
        results = self.execute_query(query, parameters)
        if results:
            history_json = results[0]["history"]
            return {"history": json.loads(history_json)}  # JSON string -> Python dictionary
        return None

    # RAG search
    def search(self, query: str, k: int = 3) -> str:
        try:
            logging.info("Starting search method")
            query_embedding = self.embedding.embed_query(query)
            # Ensure embedding is a numpy array of type float32
            query_embedding = np.array(query_embedding).astype('float32')
            docs_with_score = self.db.similarity_search_with_score_by_vector(query_embedding, k=k)
            results = []
            for doc, score in docs_with_score:
                # Use .get to avoid KeyError
                file_name = doc.metadata.get('file_name', 'Unknown')
                chunk_id = doc.metadata.get('chunk_id', 'Unknown')
                results.append(
                    f"Score: {score}\nFile: {file_name}\nChunk ID: {chunk_id}\n{doc.page_content}"
                )
        except AttributeError as ae:
            logging.error(f"AttributeError during search: {str(ae)}")
            return "Attribute error during search."
        except Exception as e:
            logging.error(f"Error during search: {str(e)}")
            return "Error during search."
        return "\n\n".join(results)

    # RAG upload from REST API
    def upload_document(self, file: UploadFile):
        """
        Uploads and processes the document, then adds it to the Neo4j Vector index.

        :param file: The uploaded file from the REST API.
        """
        try:
            # Read the file content
            content = file.file.read()
            file_extension = os.path.splitext(file.filename)[-1].lower()

            # Create a temporary file for text documents
            temp_file_path = f"/tmp/{file.filename}"

            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(content)

            # Choose the document loader based on the file format
            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_file_path)
            elif file_extension == ".docx":
                loader = DocxLoader(temp_file_path)
            elif file_extension == ".txt":
                loader = TextLoader(temp_file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")

            # Load the documents
            documents = loader.load()

            # Split the text into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            chunks = text_splitter.split_documents(documents)

            # Generate embeddings for the chunks
            chunk_texts = [chunk.page_content for chunk in chunks]
            embeddings = self.embedding.embed_documents(chunk_texts)

            # Prepare data for Neo4j
            data = [
                {
                    "id": chunk.metadata.get("id", f"chunk_{index}"),  # Generate unique ID
                    "embedding": embedding,
                    "text": chunk.page_content,
                    "metadata": chunk.metadata,
                }
                for index, (chunk, embedding) in enumerate(zip(chunks, embeddings))
            ]

            # Add documents to the Neo4j database
            self.add_documents(data)

            logging.info(f"Successfully uploaded and processed file: {file.filename}")

        except ValueError as ve:
            logging.error(f"Validation error: {str(ve)}")
            raise Exception(f"File upload failed: {str(ve)}")

        except Exception as e:
            logging.error(f"Error uploading document: {str(e)}")
            raise Exception(f"Failed to upload document: {str(e)}")

        finally:
            # Remove the temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def add_documents(self, data: List[dict]):
        """
        Adds documents to the Neo4j database.

        :param data: List of document data containing id, embedding, text, and metadata fields.
        """
        query = """
        UNWIND $data AS row
        MERGE (c:Chunk {id: row.id})
        WITH c, row
        CALL db.create.setNodeVectorProperty(c, 'embedding', row.embedding)
        WITH c, row
        SET c.text = row.text
        SET c += row.metadata
        """
        self.execute_query(query, {"data": data})

