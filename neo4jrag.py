import os
import json
import logging
from datetime import datetime
from typing import Optional, List
from neo4j import GraphDatabase 
from fastapi import UploadFile, HTTPException
from llama_index.core import  Document
from langchain_openai import OpenAIEmbeddings
import shutil
from typing import List, Dict, Any

# Logolás konfigurálása
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO)


import os
import logging
import json
from typing import List, Optional
from neo4j import GraphDatabase

class Neo4jManager:
    def __init__(self, url: str = None, username: str = None, password: str = None, max_pool_size: int = 10):
        self.url = url or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.username = username or os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD")
        self.driver = GraphDatabase.driver(
            self.url,
            auth=(self.username, self.password),
            max_connection_pool_size=max_pool_size
        )
        logging.info("Neo4jManager initialized successfully")

    def close(self):
        """Close the Neo4j connection."""
        self.driver.close()

    def count_chunks(self) -> int:
        """Count the number of chunks in the database."""
        query = """
        MATCH (n:Chunk)
        RETURN count(n) AS total_chunks
        """
        with self.driver.session() as session:
            result = session.run(query)
            count = result.single()["total_chunks"]
            return count

    def save_chunk(self, text: str, embedding: List[float], metadata: dict):
        """Save a chunk to the database."""
        query = """
        MERGE (n:Chunk {
            text: $text,
            embedding: $embedding
        })
        SET n.metadata = $metadata
        """
        parameters = {
            "text": text,
            "embedding": embedding,
            "metadata": json.dumps(metadata)  # Store metadata as JSON
        }
        with self.driver.session() as session:
            session.run(query, parameters)
            logging.info("Chunk successfully saved to Neo4j")


    def create_vector_index(self, index_name: str, dimensions: int = 1536, similarity_function: str = 'cosine'):
        """Create a vector index in the database."""
        with self.driver.session() as session:
            # Drop existing index if it exists
            session.run(f"DROP INDEX {index_name} IF EXISTS")
            # Create the vector index
            session.run(
                f"""
                CREATE VECTOR INDEX {index_name}
                FOR (n:Chunk)
                ON (n.embedding)
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: {dimensions},
                        `vector.similarity_function`: '{similarity_function}'
                    }}
                }}
                """
            )
        logging.info(f"Vector index '{index_name}' created successfully")

    def search_chunks(self, query_embedding: List[float], k: int = 10) -> List[Dict[str, Any]]:
        """Search for the most similar chunks using a vector index."""
        index_name = "chunk_embedding_index"
        query = f"""
        CALL db.index.vector.queryNodes(
            '{index_name}',
            $k,
            $query_embedding
        ) YIELD node, score
        RETURN node.text AS text, node.metadata AS metadata, score
        ORDER BY score DESC
        LIMIT $k
        """
        parameters = {
            "query_embedding": query_embedding,
            "k": k
        }
        with self.driver.session() as session:
            results = session.run(query, parameters)
            
            processed_results = []
            for record in results:
                text = record.get("text", "")
                metadata_raw = record.get("metadata")
                score = record.get("score", 0.0)
                
                # Ellenőrizd, hogy a metadata nem None és valóban string-e
                if metadata_raw is not None:
                    if isinstance(metadata_raw, str):
                        try:
                            metadata = json.loads(metadata_raw)
                        except json.JSONDecodeError as e:
                            logging.error(f"JSON decoding failed for metadata: {metadata_raw} with error: {e}")
                            metadata = {}
                    elif isinstance(metadata_raw, dict):
                        # Ha már dict, akkor nincs szükség json.loads-ra
                        metadata = metadata_raw
                    else:
                        logging.warning(f"Unexpected metadata type: {type(metadata_raw)}. Metadata will be set to empty dict.")
                        metadata = {}
                else:
                    logging.warning("Metadata is None. Setting metadata to empty dict.")
                    metadata = {}
                
                processed_results.append({
                    "text": text,
                    "metadata": metadata,
                    "score": score
                })
            
            return processed_results

    def execute_query(self, query: str, parameters: Optional[dict] = None) -> List[dict]:
        """Execute a generic Cypher query."""
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return [record for record in result]


class SessionManager:
    def __init__(self, neo4j_manager: Neo4jManager):
        self.neo4j_manager = neo4j_manager

    def save_session(self, session_id: str, session_data: dict):
        query = """
        MERGE (s:Session {id: $session_id})
        SET s.history = $history, s.updated_at = $updated_at
        """
        parameters = {
            "session_id": session_id,
            "history": json.dumps(session_data["history"]),
            "updated_at": datetime.utcnow().isoformat()
        }
        self.neo4j_manager.execute_query(query, parameters)
        logging.info("Successfully saved session")

    def get_session(self, session_id: str) -> Optional[dict]:
        query = """
        MATCH (s:Session {id: $session_id})
        RETURN s.history AS history
        """
        parameters = {"session_id": session_id}
        results = self.neo4j_manager.execute_query(query, parameters)
        if results:
            history_json = results[0]["history"]
            return {"history": json.loads(history_json)}
        return None


class VectorStoreManager:
    def __init__(self, neo4j_manager: Neo4jManager):
        self.neo4j_manager = neo4j_manager
        self.embedding = OpenAIEmbeddings()
        # Vektorindex létrehozása
        self.neo4j_manager.create_vector_index("chunk_embedding_index", dimensions=1536)

    def search(self, query: str, k: int = 3) -> List[dict]:
        try:
            logging.info(f"Starting search method for query: {query}")
            
            # Leképezed a kérdést embedding-re
            query_embedding = self.embedding.embed_query(query)
            
            # Keresés Neo4j-ban
            results = self.neo4j_manager.search_chunks(query_embedding)
            
            # Csak a top-k eredményt adja vissza
            return results[:k]
        except Exception as e:
            logging.error(f"Error during search: {str(e)}", exc_info=True)
            return []
        

    async def upload_document(self, file: UploadFile):
        temp_dir = None  # Initialize here for the finally block
        try:
            # Read the file content
            content = await file.read()
            file_extension = os.path.splitext(file.filename)[-1].lower()

            # Supported file formats
            supported_formats = ['.pdf', '.docx', '.txt', '.md']

            if file_extension not in supported_formats:
                raise ValueError(f"Unsupported file format: {file_extension}")

            # Create a temporary directory for processing
            temp_dir = f"/tmp/{file.filename}_temp"
            os.makedirs(temp_dir, exist_ok=True)

            temp_file_path = os.path.join(temp_dir, file.filename)

            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(content)

            documents = []

            # Process the document based on its format
            if file_extension == ".pdf":
                documents = self.load_pdf(temp_file_path)
            elif file_extension == ".docx":
                documents = self.load_docx(temp_file_path)
            elif file_extension in [".txt", ".md"]:
                with open(temp_file_path, 'r', encoding='utf-8') as txt_file:
                    text = txt_file.read()
                    documents = [Document(text=text, metadata={"file_name": file.filename})]
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")

            # Chunk text with metadata
            def chunk_text_with_metadata(text, chunk_size=512, file_name="unknown_file"):
                logging.info("Splitting text into chunks")
                words = text.split()
                chunks = []
                for i in range(0, len(words), chunk_size):
                    chunk_content = ' '.join(words[i:i + chunk_size])
                    metadata = {
                        "chunk_index": i // chunk_size,
                        "chunk_start": i,
                        "chunk_end": i + chunk_size,
                        "file_name": file_name
                    }
                    chunks.append(Document(text=chunk_content, metadata=metadata))
                return chunks

            chunks = []
            for doc in documents:
                logging.info(f'Processing document: {type(doc)}')
                chunks.extend(chunk_text_with_metadata(doc.text, file_name=doc.metadata["file_name"]))

            # Create embeddings for chunks and add to vector store
            for chunk in chunks:
                logging.info('Creating embedding for chunk')
                embedding = self.embedding.embed_query(chunk.text)
                
                # Mentés a Neo4j-ba
                self.neo4j_manager.save_chunk(chunk.text, embedding, chunk.metadata)

            logging.info(f"Successfully uploaded and processed file: {file.filename}")
            logging.info(f"Total chunks in vectorstore: {self.neo4j_manager.count_chunks()}")

        except ValueError as ve:
            logging.error(f"Validation error: {str(ve)}", exc_info=True)
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logging.error(f"Error uploading document: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal Server Error")
        finally:
            # Remove the temporary directory and its contents
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    # Function to load data from .pdf files
    def load_pdf(self, filepath: str)-> List[Document]:
        import PyPDF2
        pdf_text = ""
        with open(filepath, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                pdf_text += page.extract_text()
        return [Document(text=pdf_text, metadata={"file_name": os.path.basename(filepath)})]

    def load_docx(self, filepath: str) -> List[Document]:
        try:
            from docx import Document as DocxDocument  # Import python-docx
            doc = DocxDocument(filepath)
            full_text = []
            for para in doc.paragraphs:
                full_text.append(para.text)
            combined_text = "\n".join(full_text)
            return [Document(text=combined_text, metadata={"file_name": os.path.basename(filepath)})]
        except Exception as e:
            logging.error(f"Error loading docx file {filepath}: {str(e)}",exc_info=True)
            return []

