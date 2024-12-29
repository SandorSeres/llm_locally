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
import uuid
from typing import List, Optional
from neo4j import GraphDatabase
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import httpx
import asyncio

#
# http://localhost:7474/browser/
#


# Logolás konfigurálása
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO)



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

    def create_topics(self, topics: List[str]):
        """
        Create Topic nodes in Neo4j from a given list of topics.

        Args:
            topics (List[str]): A list of topic names to create.
        """
        query = """
        UNWIND $topics AS topic_name
        MERGE (t:Topic {name: topic_name})
        RETURN count(*) AS topics_created
        """
        with self.driver.session() as session:
            result = session.run(query, {"topics": topics})
            count = result.single()["topics_created"]
            logging.info(f"{count} topics created successfully.")

    def link_chunk_to_topic(self, chunk_id: str, topic: str):
        """
        Link a chunk to a topic in the graph database.

        Args:
            chunk_id (str): The unique ID of the chunk.
            topic (str): The name of the topic to link.
        """
        query = """
        MATCH (chunk:Chunk {chunk_id: $chunk_id})
        MATCH (topic:Topic {name: $topic})
        MERGE (chunk)-[:RELATED_TO]->(topic)
        """
        with self.driver.session() as session:
            session.run(query, {"chunk_id": chunk_id, "topic": topic})
            logging.info(f"Chunk '{chunk_id}' linked to topic '{topic}'.")
            
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
        """Save a chunk to the database with expanded metadata."""
        query = """
        MERGE (n:Chunk {chunk_id: $chunk_id})  // A chunk_id-t használjuk azonosítóként
        SET n.text = $text,
            n.embedding = $embedding,
            n.file_name = $file_name,
            n.chunk_index = $chunk_index,
            n.chunk_start = $chunk_start,
            n.chunk_end = $chunk_end
        """
        parameters = {
            "chunk_id": metadata.get("chunk_id"),  # Új kulcs: chunk_id
            "text": text,
            "embedding": embedding,
            "file_name": metadata.get("file_name"),
            "chunk_index": metadata.get("chunk_index"),
            "chunk_start": metadata.get("chunk_start"),
            "chunk_end": metadata.get("chunk_end")
        }
        with self.driver.session() as session:
            session.run(query, parameters)
            logging.info(f"Chunk '{metadata.get('chunk_id')}' successfully saved to Neo4j")

    def save_questions_to_neo4j(self, chunk_id: str, questions: List[str]):
        """
        Save the generated questions to Neo4j and link them to the chunk.
        """
        query = """
        MATCH (chunk:Chunk {chunk_id: $chunk_id})
        UNWIND $questions AS question
        MERGE (q:Question {text: question})
        MERGE (chunk)-[:GENERATES]->(q)
        """
        with self.driver.session() as session:
            session.run(query, {"chunk_id": chunk_id, "questions": questions})
        logging.info(f"Questions for chunk '{chunk_id}' saved successfully.")

    def save_summary_to_neo4j(self, document_name: str, summary_text: str):
        query = """
        MATCH (doc:Document {name: $document_name})
        MERGE (summary:Summary {text: $summary_text})
        MERGE (doc)-[:HAS_SUMMARY]->(summary)
        """
        with self.driver.session() as session:
            session.run(query, {"document_name": document_name, "summary_text": summary_text})

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
        index_name = "chunk_embedding_index"
        query = f"""
        CALL db.index.vector.queryNodes('{index_name}', $k, $query_embedding)
        YIELD node, score
        RETURN node.text AS text, node.file_name AS file_name,
               node.chunk_index AS chunk_index, node.chunk_start AS chunk_start,
               node.chunk_end AS chunk_end, score
        ORDER BY score DESC LIMIT $k
        """
        parameters = {"query_embedding": query_embedding, "k": k}
        
        with self.driver.session() as session:
            results = session.run(query, parameters)
            processed_results = []
            for record in results:
                processed_results.append({
                    "text": record["text"],
                    "metadata": {
                        "file_name": record["file_name"],
                        "chunk_index": record["chunk_index"],
                        "chunk_start": record["chunk_start"],
                        "chunk_end": record["chunk_end"]
                    },
                    "score": record["score"]
                })
            return processed_results

    
    def advanced_search(self, query_embedding: List[float], k: int = 10) -> List[Dict[str, Any]]:
        """
        Combines embedding-based search with various graph relationships and includes question nodes for enhanced accuracy.

        Args:
            query_embedding (List[float]): Query embedding vector.
            k (int): Number of top results to return.

        Returns:
            List[Dict[str, Any]]: Combined and ranked results from embedding and graph searches.
        """
        # Step 1: Perform embedding-based search
        embedding_results = self.search_chunks(query_embedding, k)
        chunk_ids = [result['metadata']['chunk_index'] for result in embedding_results]

        # Step 2: Graph relationship-based search
        graph_query = """
        MATCH (chunk:Chunk)-[:SIMILAR_TO|BELONGS_TO|NEXT]->(related_chunk:Chunk)
        WHERE chunk.chunk_index IN $chunk_ids
        RETURN DISTINCT related_chunk.text AS text, 
               related_chunk.file_name AS file_name,
               related_chunk.chunk_index AS chunk_index,
               related_chunk.chunk_start AS chunk_start,
               related_chunk.chunk_end AS chunk_end,
               'graph' AS source
        """
        graph_results = self.execute_query(graph_query, {"chunk_ids": chunk_ids})

        # Step 3: Document summaries
        summary_query = """
        MATCH (chunk:Chunk)-[:BELONGS_TO]->(doc:Document)-[:HAS_SUMMARY]->(summary:Summary)
        WHERE chunk.chunk_index IN $chunk_ids
        RETURN DISTINCT summary.text AS text,
               doc.name AS document_name,
               'summary' AS source
        """
        summary_results = self.execute_query(summary_query, {"chunk_ids": chunk_ids})

        # Step 4: Generated questions
        question_query = """
        MATCH (chunk:Chunk)-[:GENERATES]->(q:Question)
        WHERE chunk.chunk_index IN $chunk_ids
        RETURN DISTINCT q.text AS text,
               chunk.file_name AS file_name,
               'question' AS source
        """
        question_results = self.execute_query(question_query, {"chunk_ids": chunk_ids})

        # Step 5: Combine and rank results
        combined_results = []

        # Add embedding results
        for result in embedding_results:
            combined_results.append({
                "text": result["text"],
                "metadata": {
                    "file_name": result["metadata"].get("file_name", "unknown"),
                    "chunk_index": result["metadata"].get("chunk_index"),
                    "chunk_start": result["metadata"].get("chunk_start"),
                    "chunk_end": result["metadata"].get("chunk_end"),
                },
                "score": result["score"],
                "source": "embedding"
            })

        # Add graph results
        for record in graph_results:
            combined_results.append({
                "text": record["text"],
                "metadata": {
                    "file_name": record.get("file_name", "unknown"),
                    "chunk_index": record.get("chunk_index"),
                    "chunk_start": record.get("chunk_start"),
                    "chunk_end": record.get("chunk_end"),
                },
                "score": None,
                "source": record.get("source", "graph")
            })

        # Add summaries
        for record in summary_results:
            combined_results.append({
                "text": record["text"],
                "metadata": {
                    "document_name": record["document_name"]
                },
                "score": None,
                "source": record.get("source", "summary")
            })

        # Add questions
        for record in question_results:
            combined_results.append({
                "text": record["text"],
                "metadata": {
                    "file_name": record.get("file_name", "unknown"),
                },
                "score": None,
                "source": record.get("source", "question")
            })

        # Step 6: (Optional) Rank results by relevance or additional criteria
        ranked_results = sorted(
            combined_results,
            key=lambda x: x["score"] if x["score"] is not None else 0,
            reverse=True
        )

        return ranked_results[:k]

    def advanced_search_with_topics(self, query_embedding: List[float], query_text: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Combines embedding-based search with various graph relationships, question nodes,
        and topic nodes for enhanced accuracy. Identifies relevant topics for the query using LLM.

        Args:
            query_embedding (List[float]): Query embedding vector.
            query_text (str): The raw query text.
            k (int): Number of top results to return.

        Returns:
            List[Dict[str, Any]]: Combined and ranked results from embedding, graph,
                                  topic, and question-based searches.
        """
        # Step 1: Identify topics using LLM
        try:
            topics = self.identify_chunk_topics(query_text, self.neo4j_manager.get_all_topics())
        except Exception as e:
            logging.error(f"Error identifying topics for query: {str(e)}")
            topics = []

        # Step 2: Perform embedding-based search
        embedding_results = self.search_chunks(query_embedding, k)
        chunk_ids = [result['metadata']['chunk_index'] for result in embedding_results]

        # Step 3: Graph relationship-based search
        graph_query = """
        MATCH (chunk:Chunk)-[:SIMILAR_TO|BELONGS_TO|NEXT]->(related_chunk:Chunk)
        WHERE chunk.chunk_index IN $chunk_ids
        RETURN DISTINCT related_chunk.text AS text, 
               related_chunk.file_name AS file_name,
               related_chunk.chunk_index AS chunk_index,
               related_chunk.chunk_start AS chunk_start,
               related_chunk.chunk_end AS chunk_end,
               'graph' AS source
        """
        graph_results = self.execute_query(graph_query, {"chunk_ids": chunk_ids})

        # Step 4: Document summaries
        summary_query = """
        MATCH (chunk:Chunk)-[:BELONGS_TO]->(doc:Document)-[:HAS_SUMMARY]->(summary:Summary)
        WHERE chunk.chunk_index IN $chunk_ids
        RETURN DISTINCT summary.text AS text,
               doc.name AS document_name,
               'summary' AS source
        """
        summary_results = self.execute_query(summary_query, {"chunk_ids": chunk_ids})

        # Step 5: Generated questions
        question_query = """
        MATCH (chunk:Chunk)-[:GENERATES]->(q:Question)
        WHERE chunk.chunk_index IN $chunk_ids
        RETURN DISTINCT q.text AS text,
               chunk.file_name AS file_name,
               'question' AS source
        """
        question_results = self.execute_query(question_query, {"chunk_ids": chunk_ids})

        # Step 6: Topic-based search
        topic_query = """
        MATCH (t:Topic)<-[:RELATED_TO]-(chunk:Chunk)
        WHERE t.name IN $topics
        RETURN DISTINCT chunk.text AS text, 
               chunk.file_name AS file_name,
               chunk.chunk_index AS chunk_index,
               chunk.chunk_start AS chunk_start,
               chunk.chunk_end AS chunk_end,
               t.name AS topic_name,
               'topic' AS source
        """
        topic_results = self.execute_query(topic_query, {"topics": topics})

        # Step 7: Combine and rank results
        combined_results = []

        # Add embedding results
        for result in embedding_results:
            combined_results.append({
                "text": result["text"],
                "metadata": {
                    "file_name": result["metadata"].get("file_name", "unknown"),
                    "chunk_index": result["metadata"].get("chunk_index"),
                    "chunk_start": result["metadata"].get("chunk_start"),
                    "chunk_end": result["metadata"].get("chunk_end"),
                },
                "score": result["score"],
                "source": "embedding"
            })

        # Add graph results
        for record in graph_results:
            combined_results.append({
                "text": record["text"],
                "metadata": {
                    "file_name": record.get("file_name", "unknown"),
                    "chunk_index": record.get("chunk_index"),
                    "chunk_start": record.get("chunk_start"),
                    "chunk_end": record.get("chunk_end"),
                },
                "score": None,
                "source": record.get("source", "graph")
            })

        # Add summaries
        for record in summary_results:
            combined_results.append({
                "text": record["text"],
                "metadata": {
                    "document_name": record["document_name"]
                },
                "score": None,
                "source": record.get("source", "summary")
            })

        # Add questions
        for record in question_results:
            combined_results.append({
                "text": record["text"],
                "metadata": {
                    "file_name": record.get("file_name", "unknown"),
                },
                "score": None,
                "source": record.get("source", "question")
            })

        # Add topic-based results
        for record in topic_results:
            combined_results.append({
                "text": record["text"],
                "metadata": {
                    "file_name": record.get("file_name", "unknown"),
                    "topic_name": record.get("topic_name", "unknown"),
                    "chunk_index": record.get("chunk_index"),
                    "chunk_start": record.get("chunk_start"),
                    "chunk_end": record.get("chunk_end"),
                },
                "score": None,
                "source": record.get("source", "topic")
            })

        # Step 8: Rank results by relevance or additional criteria
        ranked_results = sorted(
            combined_results,
            key=lambda x: x["score"] if x["score"] is not None else 0,
            reverse=True
        )

        return ranked_results[:k]


    def create_document_relationships(self):
        """
        Create BELONGS_TO relationships between chunks and their document nodes.
        Assumes 'file_name' is stored as a separate property in each chunk.
        """
        query = """
        MATCH (chunk:Chunk)
        WHERE chunk.file_name IS NOT NULL
        MERGE (doc:Document {name: chunk.file_name})
        MERGE (chunk)-[:BELONGS_TO]->(doc)
        RETURN count(*) AS relationships_created
        """
        with self.driver.session() as session:
            result = session.run(query)
            count = result.single()["relationships_created"]
            logging.info(f"{count} BELONGS_TO relationships created successfully.")


    def create_similarity_relationships(self, index_name: str = "chunk_embedding_index", top_k: int = 10, threshold: float = 0.8):
        query = f"""
        MATCH (c1:Chunk)
        CALL db.index.vector.queryNodes(
            '{index_name}',
            $top_k,
            c1.embedding
        ) YIELD node, score
        WHERE score > $threshold AND node <> c1
        MERGE (c1)-[r:SIMILAR_TO]->(node)
        ON CREATE SET r.similarity = score
        """
        with self.driver.session() as session:
            session.run(query, {"top_k": top_k, "threshold": threshold})
            logging.info(f"SIMILAR_TO relationships created for top {top_k} chunks with similarity threshold > {threshold}")


    def create_next_relationship(self, prev_chunk_id: str, current_chunk_id: str):
        """
        Create a NEXT relationship between two chunks identified by their unique chunk_id.
        """
        query = """
        MATCH (c1:Chunk {chunk_id: $prev_chunk_id})
        MATCH (c2:Chunk {chunk_id: $current_chunk_id})
        WHERE c1 <> c2
        MERGE (c1)-[:NEXT]->(c2)
        """
        parameters = {
            "prev_chunk_id": prev_chunk_id,
            "current_chunk_id": current_chunk_id
        }
        with self.driver.session() as session:
            logging.info(f"Creating NEXT relationship: {prev_chunk_id} -> {current_chunk_id}")
            result = session.run(query, parameters)
            logging.info(f"NEXT relationship creation result: {result.consume().counters}")

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

class TopicManager:
    def __init__(self, neo4j_manager: Neo4jManager):
        self.neo4j_manager = neo4j_manager
        self.topics = []

    def load_topics_from_file(self, filepath: str) -> List[str]:
        """
        Load topics from a text or JSON file.
        Args:
            filepath (str): Path to the file containing topics.

        Returns:
            List[str]: A list of topics.
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                if filepath.endswith(".json"):
                    self.topics = json.load(file).get("topics", [])
                else:
                    self.topics = [line.strip() for line in file.readlines() if line.strip()]
            logging.info(f"{len(self.topics)} topics loaded successfully from {filepath}.")
        except Exception as e:
            logging.error(f"Error loading topics from file: {str(e)}", exc_info=True)
        return self.topics

    def save_topics_to_file(self, filepath: str):
        """
        Save the current list of topics to a file.
        Args:
            filepath (str): Path to the file to save topics.
        """
        try:
            if filepath.endswith(".json"):
                with open(filepath, 'w', encoding='utf-8') as file:
                    json.dump({"topics": self.topics}, file, ensure_ascii=False, indent=4)
            else:
                with open(filepath, 'w', encoding='utf-8') as file:
                    file.writelines([f"{topic}\n" for topic in self.topics])
            logging.info(f"{len(self.topics)} topics saved to {filepath}.")
        except Exception as e:
            logging.error(f"Error saving topics to file: {str(e)}", exc_info=True)

    def create_topics_in_neo4j(self):
        """
        Create topics in the Neo4j database.
        """
        if not self.topics:
            logging.warning("No topics to create in Neo4j.")
            return
        self.neo4j_manager.create_topics(self.topics)
        logging.info("Topics successfully created in Neo4j.")

class EmbeddingGenerator:
    """
    Egy általános osztály az Ollama API használatához embedding generálására.
    """
    def __init__(self, api_url: str = None, embedding_size: int = 768):
        self.api_url = api_url
        self.embedding_size = embedding_size

    def embed_query(self, text):
        url = "http://ollama:11434/api/embeddings"  # Állítsd be a megfelelő címet
        
        payload = {
            "model": "nomic-embed-text",  # Vagy a használni kívánt embedding modell neve ,768
            "prompt": text
        }
        
        with httpx.Client() as client:  # Szinkron kliens használata
            response = client.post(url, json=payload)
            
        if response.status_code == 200:
            result = response.json()
            return result["embedding"]
        else:
            raise Exception(f"Hiba történt: {response.status_code}, {response.text}")

class VectorStoreManager:
    def __init__(self, neo4j_manager: Neo4jManager, topic_manager: TopicManager):
        self.neo4j_manager = neo4j_manager
        self.topic_manager = topic_manager  # A lifespan-ből kapja a példányt
        api_url= os.getenv("OLLAMA_EMBEDDING_API_URL","")
        self.embedding = EmbeddingGenerator(api_url=api_url)  # EmbeddingGenerator használata

        # Vektorindex létrehozása
        self.neo4j_manager.create_vector_index("chunk_embedding_index", dimensions=self.embedding.embedding_size)

    def upload_new_topics(self, filepath: str):
        """
        Új témák feltöltése és Neo4j adatbázisban történő létrehozása.
        """
        new_topics = self.topic_manager.load_topics_from_file(filepath)
        self.topic_manager.create_topics_in_neo4j()
        return new_topics

    def search(self, query: str, k: int = 3) -> List[dict]:
        """
        Keresés egy lekérdezés alapján.
        
        Args:
            query (str): A felhasználói kérdés.
            k (int): Az eredmények maximális száma.

        Returns:
            List[dict]: A top-k keresési eredmények.
        """
        try:
            logging.info(f"Starting search method for query: {query}")
            
            # Leképezed a kérdést embedding-re
            query_embedding = self.embedding.embed_query(query)
            
            # Keresés Neo4j-ban
            results = self.neo4j_manager.advanced_search_with_topics(query_embedding)
            
            # Csak a top-k eredményt adja vissza
            return results[:k]
        except Exception as e:
            logging.error(f"Error during search: {str(e)}", exc_info=True)
            return []

    def search_chunks_by_document(self, query_embedding: List[float], k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve chunks and related chunks from the same document."""
        index_name = "chunk_embedding_index"
        query = f"""
        CALL db.index.vector.queryNodes(
            '{index_name}',
            $k,
            $query_embedding
        ) YIELD node, score
        MATCH (node)-[:BELONGS_TO]->(doc:Document)<-[:BELONGS_TO]-(related_chunk:Chunk)
        RETURN node.text AS text, node.metadata AS metadata, score,
               related_chunk.text AS related_text, related_chunk.metadata AS related_metadata
        ORDER BY score DESC
        LIMIT $k
        """
        parameters = {
            "query_embedding": query_embedding,
            "k": k
        }
        with self.driver.session() as session:
            results = session.run(query, parameters)
            return [
                {
                    "text": record["text"],
                    "metadata": json.loads(record["metadata"]),
                    "related_text": record["related_text"],
                    "related_metadata": json.loads(record["related_metadata"]),
                    "score": record["score"]
                }
                for record in results
            ]

    def search_chunks_with_relationships(self, query_embedding: List[float], k: int = 10) -> List[Dict[str, Any]]:
        """Search for similar chunks and their related chunks using graph relationships."""
        index_name = "chunk_embedding_index"
        query = f"""
        CALL db.index.vector.queryNodes(
            '{index_name}',
            $k,
            $query_embedding
        ) YIELD node, score
        MATCH (node)-[:SIMILAR_TO]->(related_chunk:Chunk)
        RETURN node.text AS text, node.metadata AS metadata, score,
               related_chunk.text AS related_text, related_chunk.metadata AS related_metadata
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
                related_text = record.get("related_text", "")
                related_metadata_raw = record.get("related_metadata", "")
                score = record.get("score", 0.0)
                
                processed_results.append({
                    "text": text,
                    "metadata": json.loads(metadata_raw) if metadata_raw else {},
                    "related_text": related_text,
                    "related_metadata": json.loads(related_metadata_raw) if related_metadata_raw else {},
                    "score": score
                })
            
            return processed_results
        

    def generate_hypothetical_questions(self, chunk_text: str, max_questions: int = 3) -> List[str]:
        """
        Generate hypothetical questions from a chunk of text using an LLM.

        Args:
            chunk_text (str): The text chunk to generate questions for.
            max_questions (int): The maximum number of questions to return.

        Returns:
            List[str]: A list of generated questions (up to max_questions).
        """
        if not chunk_text.strip():
            logging.warning("Empty or invalid chunk text provided for question generation.")
            return []

        llm = ChatOpenAI(
            model="gpt-4o",  # Chat modell
            temperature=0,  # Alacsony hőmérséklet a következetes válaszok érdekében
            max_tokens=512  # Token limit
        )

        # Prompt szöveg közvetlen megadása
        prompt = (
            f"Based on the following text, generate up to {max_questions} relevant questions:\n\n"
            f"{chunk_text}\n\nQuestions:"
        )

        try:
            # LLM hívás az invoke metódussal
            response = llm.invoke([HumanMessage(content=prompt)])

            # Válasz feldolgozása
            raw_output = response.content  # A válasz szövege
            questions = [q.strip() for q in raw_output.split("\n") if q.strip()]

            # Maximum `max_questions` visszaadása
            return questions[:max_questions]

        except Exception as e:
            logging.error(f"Error generating hypothetical questions: {str(e)}")
            return []  # Hiba esetén üres listát adunk vissza


    def summarize_document(self, document_text: str, chunk_size: int = 2000) -> str:
        """
        Summarize a document while handling token limits by breaking it into smaller chunks.

        Args:
            document_text (str): The text of the document to summarize.
            chunk_size (int): The maximum number of tokens for each chunk.

        Returns:
            str: The final summarized text of the entire document.
        """
        llm = ChatOpenAI(
            model="gpt-4o",  # Chat modell
            temperature=0,  # Alacsony hőmérséklet a következetes válaszok érdekében
            max_tokens=512  # Token limit
        )
        
        # Step 1: Split document into manageable chunks
        words = document_text.split()
        chunks = [
            " ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)
        ]

        # Step 2: Generate summaries for each chunk
        chunk_summaries = []
        for chunk in chunks:
            try:
                response = llm([HumanMessage(content=f"Summarize the following text in 3 sentences:\n\n{chunk}")])
                summary = response.content.strip()  # Az összefoglaló szöveg
                chunk_summaries.append(summary)
            except Exception as e:
                logging.error(f"Error summarizing chunk: {str(e)}")
                chunk_summaries.append("Error summarizing this chunk.")

        # Step 3: Combine chunk summaries into a single summary
        combined_summaries = "\n".join(chunk_summaries)  # Külön változó a summarizált szövegekhez
        try:
            response = llm([HumanMessage(content=f"Combine the following summaries into a cohesive summary of the entire document:\n\n{combined_summaries}")])
            final_summary = response.content.strip()
        except Exception as e:
            logging.error(f"Error creating final summary: {str(e)}")
            final_summary = "Error generating summary."
        
        return final_summary


    async def simple_upload_document(self, file_or_content, filename=None):
        """ Nem túl nagy chunk szám és dokumentum számhoz """
        temp_dir = None
        try:
            # Input fájl vagy tartalom feldolgozása
            if isinstance(file_or_content, UploadFile):
                content = await file_or_content.read()
                filename = file_or_content.filename
            elif isinstance(file_or_content, bytes):
                content = file_or_content
                if filename is None:
                    raise ValueError("Filename must be provided when uploading bytes content")
            else:
                raise ValueError("Invalid input type. Expected UploadFile or bytes.")

            file_extension = os.path.splitext(filename)[-1].lower()
            supported_formats = ['.pdf', '.docx', '.txt', '.md']
            if file_extension not in supported_formats:
                raise ValueError(f"Unsupported file format: {file_extension}")

            # Ideiglenes könyvtár létrehozása
            temp_dir = f"/tmp/{filename}_temp"
            os.makedirs(temp_dir, exist_ok=True)
            temp_file_path = os.path.join(temp_dir, filename)
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(content)

            documents = []
            if file_extension == ".pdf":
                documents = self.load_pdf(temp_file_path)
            elif file_extension == ".docx":
                documents = self.load_docx(temp_file_path)
            elif file_extension in [".txt", ".md"]:
                with open(temp_file_path, 'r', encoding='utf-8') as txt_file:
                    text = txt_file.read()
                    documents = [Document(text=text, metadata={"file_name": filename})]

            # Szöveg chunk-okra bontása
            def chunk_text_with_metadata(text, chunk_size=512, file_name="unknown_file"):
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

            # Chunk-ok létrehozása és mentése
            chunks = []
            for doc in documents:
                chunks.extend(chunk_text_with_metadata(doc.text, file_name=doc.metadata["file_name"]))

            previous_chunk_id = None

            for idx, chunk in enumerate(chunks):
                embedding = await self.embedding.embed_query(chunk.text)
                unique_id = str(uuid.uuid4())  # Egyedi azonosító generálása
                chunk_id = f"{chunk.metadata['file_name']}_chunk_{idx}_{unique_id}"  # Globálisan egyedi azonosító

                # Chunk mentése Neo4j-ba
                self.neo4j_manager.save_chunk(
                    chunk.text,
                    embedding,
                    {"chunk_id": chunk_id, **chunk.metadata}
                )

                # Hipotetikus kérdések generálása
                questions = self.generate_hypothetical_questions(chunk.text)  
                self.neo4j_manager.save_questions_to_neo4j(chunk_id, questions)

                # NEXT kapcsolat építése a létező metódussal
                if previous_chunk_id:
                    self.neo4j_manager.create_next_relationship(previous_chunk_id, chunk_id)

                previous_chunk_id = chunk_id
            # Dokumentum kapcsolatok építése
            self.neo4j_manager.create_document_relationships()

            # Hasonlósági kapcsolatok építése
            self.neo4j_manager.create_similarity_relationships(top_k=10, threshold=0.8)

            # Dokumentum összefoglalójának létrehozása
            document_text = " ".join([chunk.text for chunk in chunks])
            summary = self.summarize_document(document_text)
            self.neo4j_manager.save_summary_to_neo4j(filename, summary)

            logging.info(f"File '{filename}' feldolgozása befejeződött.")
            logging.info(f"Összes chunk a rendszerben: {self.neo4j_manager.count_chunks()}")

        except Exception as e:
            logging.error(f"Error uploading document: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal Server Error")
        finally:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)


    async def upload_document(self, file_or_content, filename=None):
        """ Több ezres dokumentum számhoz és százezres chunkhoz """
        temp_dir = None
        try:
            # Input fájl vagy tartalom feldolgozása
            if isinstance(file_or_content, UploadFile):
                content = await file_or_content.read()
                filename = file_or_content.filename
            elif isinstance(file_or_content, bytes):
                content = file_or_content
                if filename is None:
                    raise ValueError("Filename must be provided when uploading bytes content")
            else:
                raise ValueError("Invalid input type. Expected UploadFile or bytes.")

            file_extension = os.path.splitext(filename)[-1].lower()
            supported_formats = ['.pdf', '.docx', '.txt', '.md']
            if file_extension not in supported_formats:
                raise ValueError(f"Unsupported file format: {file_extension}")

            # Ideiglenes könyvtár létrehozása
            temp_dir = f"/tmp/{filename}_temp"
            os.makedirs(temp_dir, exist_ok=True)
            temp_file_path = os.path.join(temp_dir, filename)
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(content)

            documents = []
            if file_extension == ".pdf":
                documents = self.load_pdf(temp_file_path)
            elif file_extension == ".docx":
                documents = self.load_docx(temp_file_path)
            elif file_extension in [".txt", ".md"]:
                with open(temp_file_path, 'r', encoding='utf-8') as txt_file:
                    text = txt_file.read()
                    documents = [Document(text=text, metadata={"file_name": filename})]

            # Szöveg chunk-okra bontása
            def chunk_text_with_metadata(text, chunk_size=512, file_name="unknown_file"):
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

            # Topikok lekérése a TopicManager-ből
            topics = self.topic_manager.topics  # Automatikusan lekérjük az aktuális topikokat

            # Chunk-ok létrehozása és mentése
            chunks = []
            for doc in documents:
                chunks.extend(chunk_text_with_metadata(doc.text, file_name=doc.metadata["file_name"]))

            previous_chunk_id = None

            for idx, chunk in enumerate(chunks):
                embedding = self.embedding.embed_query(chunk.text)
                unique_id = str(uuid.uuid4())  # Egyedi azonosító generálása
                chunk_id = f"{chunk.metadata['file_name']}_chunk_{idx}_{unique_id}"  # Globálisan egyedi azonosító

                # Chunk mentése Neo4j-ba
                self.neo4j_manager.save_chunk(
                    chunk.text,
                    embedding,
                    {"chunk_id": chunk_id, **chunk.metadata}
                )

                # Hipotetikus kérdések generálása
                questions = self.generate_hypothetical_questions(chunk.text)
                self.neo4j_manager.save_questions_to_neo4j(chunk_id, questions)

                # Kapcsolódás topikokhoz LLM segítségével
                chunk_topics = self.identify_chunk_topics(chunk.text, topics)
                for topic in chunk_topics:
                    self.neo4j_manager.link_chunk_to_topic(chunk_id, topic)

                # NEXT kapcsolat építése a létező metódussal
                if previous_chunk_id:
                    self.neo4j_manager.create_next_relationship(previous_chunk_id, chunk_id)

                previous_chunk_id = chunk_id

            # Dokumentum kapcsolatok építése
            self.neo4j_manager.create_document_relationships()

            # Hasonlósági kapcsolatok inkrementális építése
            self.neo4j_manager.create_similarity_relationships(top_k=20, threshold=0.8)

            # Dokumentum összefoglalójának létrehozása
            document_text = " ".join([chunk.text for chunk in chunks])
            summary = self.summarize_document(document_text)
            self.neo4j_manager.save_summary_to_neo4j(filename, summary)

            logging.info(f"File '{filename}' feldolgozása befejeződött.")
            logging.info(f"Összes chunk a rendszerben: {self.neo4j_manager.count_chunks()}")

        except Exception as e:
            logging.error(f"Error uploading document: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal Server Error")
        finally:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def identify_chunk_topics(self, chunk_text: str, topics: List[str]) -> List[str]:
        """
        Uses an LLM to identify which topics from the list are relevant to the chunk.

        Args:
            chunk_text (str): The text of the chunk.
            topics (List[str]): List of topics to check.

        Returns:
            List[str]: Topics that are relevant to the chunk.
        """
        llm = ChatOpenAI(
            model="gpt-4o",  # Chat modell
            temperature=0,  # Alacsony hőmérséklet a következetes válaszok érdekében
            max_tokens=100  # Token limit
        )
        prompt = (
            f"Given the text:\n\n{chunk_text}\n\n"
            f"Identify the topics from this list that are relevant:\n{', '.join(topics)}"
        )

        try:
            # LLM hívás az invoke metódussal
            response = llm.invoke([HumanMessage(content=prompt)])

            # Válasz feldolgozása
            identified_topics = response.content.strip()  # Válasz szövegének elérése
            relevant_topics = [topic.strip() for topic in identified_topics.split(",") if topic.strip() in topics]

            return relevant_topics

        except Exception as e:
            logging.error(f"Error identifying topics for chunk: {str(e)}")
            return []  # Hiba esetén üres listát adunk vissza
           
    def periodic_full_rebuild(self):
        """
        Teljes similarity újragenerálása időszakosan.
        """
        self.rebuild_similarity_index()
        self.create_similarity_relationships(top_k=50, threshold=0.8)
        logging.info("Teljes hasonlósági gráf újragenerálva.")
    
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

