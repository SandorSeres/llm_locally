import os
import json
import logging
from datetime import datetime
from typing import Optional, List
from neo4j import GraphDatabase 
from fastapi import UploadFile, HTTPException
from llama_index.core import  Document
import shutil
from typing import List, Dict, Any
import uuid
from typing import List, Optional
from neo4j import GraphDatabase
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
import httpx
import asyncio
import sys
import importlib.util
# Python verzió meghatározása
python_version = f"cpython-{sys.version_info.major}{sys.version_info.minor}"

# Betöltjük a model_manager modult
spec_model_manager = importlib.util.spec_from_file_location("model_manager", f"/app/__pycache__/model_manager.{python_version}.pyc")
model_manager = importlib.util.module_from_spec(spec_model_manager)
spec_model_manager.loader.exec_module(model_manager)

spec_content_manager = importlib.util.spec_from_file_location("content_manager", f"/app/__pycache__/content_manager.{python_version}.pyc")
content_manager = importlib.util.module_from_spec(spec_content_manager)
spec_content_manager.loader.exec_module(content_manager)

#
# http://localhost:7474/browser/
#


# Logolás konfigurálása
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO)



class Neo4jManager:
    def __init__(self, model_manager,url: str = None, username: str = None, password: str = None, max_pool_size: int = 10):
        self.url = url or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.username = username or os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD")
        self.driver = GraphDatabase.driver(
            self.url,
            auth=(self.username, self.password),
            max_connection_pool_size=max_pool_size
        )
        self.model_manager = model_manager
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

    def get_all_topics(self) -> List[str]:
        """
        Lekérdezi az összes Topic node-ot, és visszaadja a nevüket egy listaként.
        """
        query = """
        MATCH (t:Topic)
        RETURN t.name AS topic_name
        """
        with self.driver.session() as session:
            result = session.run(query)
            return [record["topic_name"] for record in result]

    async def identify_chunk_topics(self, chunk_text: str, topics: List[str]) -> List[str]:
        """
        Uses an LLM to identify which topics from the list are relevant to the chunk.

        Args:
            chunk_text (str): The text of the chunk.
            topics (List[str]): List of topics to check.

        Returns:
            List[str]: Topics that are relevant to the chunk.
        """
        prompt = (
            f"Given the text:\n\n{chunk_text}\n\n"
            f"Identify the topics from this list that are relevant:\n{', '.join(topics)}"
        )

        try:
            # LLM hívás az invoke metódussal
            response = await self.model_manager.generate_complete([{"role": "user", "content": prompt}])

            # Válasz feldolgozása
            identified_topics = response  # Válasz szövegének elérése
            relevant_topics = [topic.strip() for topic in identified_topics.split(",") if topic.strip() in topics]

            return relevant_topics

        except Exception as e:
            logging.error(f"Error identifying topics for chunk: {str(e)}")
            return []  # Hiba esetén üres listát adunk vissza

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

    def save_questions_with_embedding_to_neo4j(
        self,
        chunk_id: str,
        question_embeddings: List[tuple]
    ):
        """
        Save the generated questions (with embedding) to Neo4j and link them to the chunk.

        question_embeddings: List[tuple(text, embedding)]
        """
        query = """
        MATCH (chunk:Chunk {chunk_id: $chunk_id})
        UNWIND $question_embeddings AS qe
        MERGE (q:Question {text: qe.question_text})
        SET q.embedding = qe.embedding
        MERGE (chunk)-[:GENERATES]->(q)
        """
        # A paraméterek "list of dictionaries" formában mehetnek, pl.:
        q_list = []
        for (q_text, q_emb) in question_embeddings:
            q_list.append({"question_text": q_text, "embedding": q_emb})

        with self.driver.session() as session:
            session.run(query, {"chunk_id": chunk_id, "question_embeddings": q_list})
        logging.info(f"{len(question_embeddings)} Question nodes (with embedding) saved for chunk '{chunk_id}'.")

    def save_summary_to_neo4j(self, document_name: str, summary_text: str):
        query = """
        MATCH (doc:Document {name: $document_name})
        MERGE (summary:Summary {text: $summary_text})
        MERGE (doc)-[:HAS_SUMMARY]->(summary)
        """
        with self.driver.session() as session:
            session.run(query, {"document_name": document_name, "summary_text": summary_text})

    def create_vector_index(self, index_name: str, label: str, property_name: str = "embedding", dimensions: int = 1536, similarity_function: str = 'cosine'):
        """
        Create a vector index in the database for a specific label and property, ensuring it does not already exist.

        Args:
            index_name (str): Name of the index.
            label (str): Node label (e.g., Chunk, Question) for which the index is created.
            property_name (str): Property to index (default: "embedding").
            dimensions (int): Number of dimensions for the vector index.
            similarity_function (str): Similarity function to use (default: 'cosine').
        """
        with self.driver.session() as session:
            # Check if the index already exists
            existing_indexes = session.run("SHOW INDEXES").data()
            existing_index_names = [index["name"] for index in existing_indexes]

            if index_name in existing_index_names:
                logging.info(f"Vector index '{index_name}' already exists. Skipping creation.")
                return

            # Create the index for the specified label and property
            try:
                session.run(
                    f"""
                    CREATE VECTOR INDEX {index_name}
                    FOR (n:{label})
                    ON (n.{property_name})
                    OPTIONS {{
                        indexConfig: {{
                            `vector.dimensions`: {dimensions},
                            `vector.similarity_function`: '{similarity_function}'
                        }}
                    }}
                    """
                )
                logging.info(f"Vector index '{index_name}' created successfully for label '{label}' on property '{property_name}'.")
            except Exception as e:
                logging.error(f"Error creating vector index '{index_name}': {str(e)}", exc_info=True)
    

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

    async def advanced_search_with_topics(self, query_embedding: List[float], query_text: str, k: int = 10) -> List[Dict[str, Any]]:
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
            topics = await self.identify_chunk_topics(query_text, self.get_all_topics())
        except Exception as e:
            logging.error(f"Error identifying topics for query: {str(e)}")
            topics = []

        # Step 2A: Perform embedding-based search
        embedding_results = self.search_chunks(query_embedding, k)
        chunk_ids = [result['metadata']['chunk_index'] for result in embedding_results]

        # A chunk_indexek kinyerése
        chunk_indexes_from_embedding = [res["metadata"]["chunk_index"] for res in embedding_results if res["metadata"].get("chunk_index") is not None]

        # ---(2B) Question-based embedding search---
        #  Itt a question_embedding_index-ből keressük a top K releváns question node-ot.
        question_index_name = "question_embedding_index"
        question_query = f"""
        CALL db.index.vector.queryNodes('{question_index_name}', $k, $query_embedding)
        YIELD node, score
        WHERE node:Question
        RETURN id(node) AS question_id,
               node.text AS question_text,
               score
        ORDER BY score DESC
        LIMIT $k
        """
        params = {"query_embedding": query_embedding, "k": k}
        question_results = []
        with self.driver.session() as session:
            q_res = session.run(question_query, params).data()
            for rec in q_res:
                question_results.append({
                    "question_id": rec["question_id"],
                    "question_text": rec["question_text"],
                    "score": rec["score"]
                })

        # Ha találtunk releváns question node-okat, megnézzük, mely chunk(ok)hoz tartoznak
        question_based_results = []
        if question_results:
            q_ids = [r["question_id"] for r in question_results]
            # Kikeressük, hogy ezek a question node-ok mely Chunk node-okra mutatnak ([:GENERATES]->(c:Chunk))
            get_chunk_for_question_query = """
            MATCH (q:Question)-[:GENERATES]->(c:Chunk)
            WHERE id(q) IN $q_ids
            RETURN id(q) AS question_id,
                   q.text AS question_text,
                   c.chunk_id AS chunk_id,
                   c.text AS chunk_text,
                   c.file_name AS file_name,
                   c.chunk_index AS chunk_index,
                   c.chunk_start AS chunk_start,
                   c.chunk_end AS chunk_end
            """
            with self.driver.session() as session:
                link_res = session.run(get_chunk_for_question_query, {"q_ids": q_ids}).data()

            for row in link_res:
                qid = row["question_id"]
                q_item = next((x for x in question_results if x["question_id"] == qid), None)
                if q_item:
                    question_score = q_item["score"]
                    question_text = q_item["question_text"]
                else:
                    question_score = 0
                    question_text = ""

                question_based_results.append({
                    "text": row["chunk_text"],
                    "metadata": {
                        "chunk_id": row["chunk_id"],
                        "file_name": row["file_name"],
                        "chunk_index": row["chunk_index"],  # itt is szerepel
                        "chunk_start": row["chunk_start"],
                        "chunk_end": row["chunk_end"],
                    },
                    "score": question_score,
                    "source": "question_embedding",
                    "matching_question": question_text
                })

        # A question-based chunk_indexek:
        chunk_indexes_from_questions = [
            x["metadata"]["chunk_index"] 
            for x in question_based_results 
            if x["metadata"].get("chunk_index") is not None
        ]

        # Összevonjuk a chunk_indexeket:
        all_chunk_indexes = set(chunk_indexes_from_embedding + chunk_indexes_from_questions)

        # ---(3) Graph relationship-based search (továbbra is chunk_index alapon)---
        graph_query = """
        MATCH (chunk:Chunk)-[:SIMILAR_TO|BELONGS_TO|NEXT]->(related_chunk:Chunk)
        WHERE chunk.chunk_index IN $chunk_indexes
        RETURN DISTINCT related_chunk.text AS text, 
               related_chunk.file_name AS file_name,
               related_chunk.chunk_index AS chunk_index,
               related_chunk.chunk_start AS chunk_start,
               related_chunk.chunk_end AS chunk_end,
               'graph' AS source
        """
        graph_results = self.execute_query(graph_query, {"chunk_indexes": list(all_chunk_indexes)})

        # ---(4) Document summaries---
        summary_query = """
        MATCH (chunk:Chunk)-[:BELONGS_TO]->(doc:Document)-[:HAS_SUMMARY]->(summary:Summary)
        WHERE chunk.chunk_index IN $chunk_indexes
        RETURN DISTINCT summary.text AS text,
               doc.name AS document_name,
               'summary' AS source
        """
        summary_results = self.execute_query(summary_query, {"chunk_indexes": list(all_chunk_indexes)})

        # ---(5) Generated questions (a régi, chunk_index alapú)---
        question_query_legacy = """
        MATCH (chunk:Chunk)-[:GENERATES]->(q:Question)
        WHERE chunk.chunk_index IN $chunk_indexes
        RETURN DISTINCT q.text AS text,
               chunk.file_name AS file_name,
               'question' AS source
        """
        question_results_legacy = self.execute_query(question_query_legacy, {"chunk_indexes": list(all_chunk_indexes)})

        # ---(6) Topic-based search---
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

        # ---(7) Combine all results---
        combined_results = []

        # (7A) Add chunk embedding results (a régi eredmények)
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

        # (7B) Add question-based chunk results
        for r in question_based_results:
            combined_results.append(r)
            # Itt r már hasonló formátumot követ: "text", "metadata", "score"

        # (7C) Add graph results
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

        # (7D) Add summaries
        for record in summary_results:
            combined_results.append({
                "text": record["text"],
                "metadata": {
                    "document_name": record["document_name"]
                },
                "score": None,
                "source": record.get("source", "summary")
            })

        # (7E) Add the legacy question results (chunk-index-based)
        for record in question_results_legacy:
            combined_results.append({
                "text": record["text"],
                "metadata": {
                    "file_name": record.get("file_name", "unknown"),
                },
                "score": None,
                "source": record.get("source", "question")
            })

        # (7F) Add topic-based results
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

        # ---(8) Sort by "score" descending, default 0 if None---
        ranked_results = sorted(
            combined_results,
            key=lambda x: x["score"] if x["score"] is not None else 0,
            reverse=True
        )

        return ranked_results[:k]


    def _concat_chunk_text(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Egyszerű segédfüggvény, 
        ami összefűzi a chunkok 'text' mezőjét új sorokkal elválasztva.
        """
        # Ha a chunk-nak nincs 'text' mezője, akkor c.get("text", "")
        return "\n".join(c["text"] for c in chunks if "text" in c)

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

class VectorStoreManager:
    def __init__(self, model_manager, neo4j_manager: Neo4jManager, topic_manager: TopicManager):
        self.model_manager = model_manager
        self.neo4j_manager = neo4j_manager
        self.topic_manager = topic_manager  # A lifespan-ből kapja a példányt
        self.contentManager = content_manager.ContentManager()
        # Vektorindex létrehozása
        self.neo4j_manager.create_vector_index(
            "chunk_embedding_index",
            label="Chunk",
            property_name="embedding",
            dimensions=self.model_manager.embedding_size
        )

        self.neo4j_manager.create_vector_index(
            "question_embedding_index",
            label="Question",
            property_name="embedding",
            dimensions=self.model_manager.embedding_size
        )

    def upload_new_topics(self, filepath: str):
        """
        Új témák feltöltése és Neo4j adatbázisban történő létrehozása.
        """
        new_topics = self.topic_manager.load_topics_from_file(filepath)
        self.topic_manager.create_topics_in_neo4j()
        return new_topics

    async def search(self, query: str, k: int = 3) -> List[dict]:
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
            query_embedding = await self.model_manager.embed_query(query)
            
            # Keresés Neo4j-ban
            #results = await self.neo4j_manager.advanced_search_with_topics(query_embedding,query)
            results = await self.neo4j_manager.advanced_search_with_topics(query_embedding,query)
            
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
        


    async def generate_hypothetical_questions(self, chunk_text: str, max_questions: int = 3) -> List[str]:
        if not chunk_text.strip():
            logging.warning("Empty or invalid chunk text provided for question generation.")
            return []

        prompt = (
            f"Based on the following text, generate up to {max_questions} relevant questions:\n\n"
            f"{chunk_text}\n\nQuestions:"
        )

        try:
            response = await self.model_manager.generate_complete([{"role": "user", "content": prompt}])

            # Feldolgozás: a sorokat tisztítsd meg, szűrd ki az üreseket
            questions = [
                line.strip()
                for line in response.splitlines()
                if line.strip() and line.strip().endswith("?")  # Csak a kérdések megőrzése
            ]

            # Csak a maximális kérdésszámot add vissza
            return questions[:max_questions]

        except Exception as e:
            logging.error(f"Error generating hypothetical questions: {e}", exc_info=True)
            return []


    async def summarize_document(self, document_text: str, chunk_size: int = 2000) -> str:
        """
        Summarize a document while handling token limits by breaking it into smaller chunks.

        Args:
            document_text (str): The text of the document to summarize.
            chunk_size (int): The maximum number of tokens for each chunk.

        Returns:
            str: The final summarized text of the entire document.
        """
        # Step 1: Split document into manageable chunks
        words = document_text.split()
        chunks = [
            " ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)
        ]

        # Step 2: Generate summaries for each chunk
        chunk_summaries = []
        for chunk in chunks:
            try:
                response = await self.model_manager.generate_complete([{"role": "user", "content": f"Summarize the following text in 3 sentences:\n\n{chunk}"}])
                #summary = response.splitlines()  # Az összefoglaló szöveg
                chunk_summaries.append(response)
            except Exception as e:
                logging.error(f"Error summarizing chunk: {str(e)}")
                chunk_summaries.append("Error summarizing this chunk.")

        # Step 3: Combine chunk summaries into a single summary
        combined_summaries = "\n".join(chunk_summaries)  # Külön változó a summarizált szövegekhez
        try:
            response = await self.model_manager.generate_complete([{"role": "user", "content":f"Combine the following summaries into a cohesive summary of the entire document:\n\n{combined_summaries}"}])
            final_summary = response.strip()
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
                embedding = await self.model_manager.embed_query(chunk.text)
                unique_id = str(uuid.uuid4())  # Egyedi azonosító generálása
                chunk_id = f"{chunk.metadata['file_name']}_chunk_{idx}_{unique_id}"  # Globálisan egyedi azonosító

                # Chunk mentése Neo4j-ba
                self.neo4j_manager.save_chunk(
                    chunk.text,
                    embedding,
                    {"chunk_id": chunk_id, **chunk.metadata}
                )

                # Hipotetikus kérdések generálása
                questions = await self.generate_hypothetical_questions(chunk.text)  
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
            summary = await self.summarize_document(document_text)
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
            # 1. Fájl betöltése és ideiglenes könyvtár létrehozása
            content, filename = await self._load_file_content(file_or_content, filename)
            temp_file_path = self._create_temp_file(content, filename)

            # 2. Szöveg kinyerése és feldarabolása
            text = self.contentManager.extract_text(temp_file_path)
            documents = self._prepare_documents(text, filename)
            chunks = self._create_chunks(documents)

            # 3. Topikok lekérése
            topics = self.topic_manager.topics
            logging.info("Topikok lekérve")

            # 4. Chunkok feldolgozása
            await self._process_chunks(chunks, topics)

            # 5. Összefoglaló generálása
            await self._finalize_upload(filename, chunks)

            logging.info(f"File '{filename}' feldolgozása befejeződött.")
        except Exception as e:
            logging.error(f"Error uploading document: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal Server Error")
        finally:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    async def _process_chunks(self, chunks, topics, max_concurrent_tasks=5):
        """
        A chunks feldolgozása, párhuzamosan, de limitált számú szálon.
        
        Args:
            chunks (list): A feldolgozandó chunks lista.
            topics (list): A kapcsolódó témák listája.
            max_concurrent_tasks (int): Egyszerre futtatható párhuzamos feladatok száma.
        """
        semaphore = asyncio.Semaphore(max_concurrent_tasks)  # Limitáljuk a párhuzamos feladatok számát

        async def process_single_chunk(chunk, idx):
            async with semaphore:  # Egyszerre csak max_concurrent_tasks feladat futhat
                # Chunk embedding generálása
                embedding = await self.model_manager.embed_query(chunk.text)
                unique_id = str(uuid.uuid4())
                chunk_id = f"{chunk.metadata['file_name']}_chunk_{idx}_{unique_id}"

                # Chunk mentése Neo4j-ba
                self.neo4j_manager.save_chunk(chunk.text, embedding, {"chunk_id": chunk_id, **chunk.metadata})

                # Kérdések generálása és embeddingek mentése egy lépésben
                questions = await self.generate_hypothetical_questions(chunk.text)
                question_embeddings = [
                    (question_text, await self.model_manager.embed_query(question_text))
                    for question_text in questions
                ]
                self.neo4j_manager.save_questions_with_embedding_to_neo4j(chunk_id, question_embeddings)

                # Topikok azonosítása
                chunk_topics = await self.neo4j_manager.identify_chunk_topics(chunk.text, topics)
                for topic in chunk_topics:
                    self.neo4j_manager.link_chunk_to_topic(chunk_id, topic)

                return chunk_id

        # Párhuzamos feldolgozás limitált szálon
        tasks = [process_single_chunk(chunk, idx) for idx, chunk in enumerate(chunks)]
        chunk_ids = await asyncio.gather(*tasks)

        # NEXT kapcsolatok építése
        for i in range(1, len(chunk_ids)):
            self.neo4j_manager.create_next_relationship(chunk_ids[i - 1], chunk_ids[i])


    async def _finalize_upload(self, filename, chunks):
        # Dokumentum összefoglaló generálása
        document_text = " ".join([chunk.text for chunk in chunks])
        summary = await self.summarize_document(document_text)
        self.neo4j_manager.save_summary_to_neo4j(filename, summary)

        # Kapcsolatok létrehozása
        self.neo4j_manager.create_document_relationships()
        self.neo4j_manager.create_similarity_relationships(
            index_name="chunk_embedding_index", top_k=20, threshold=0.8
        )

    async def _load_file_content(self, file_or_content, filename):
        if isinstance(file_or_content, UploadFile):
            content = await file_or_content.read()
            filename = file_or_content.filename
        elif isinstance(file_or_content, bytes):
            content = file_or_content
            if filename is None:
                raise ValueError("Filename must be provided when uploading bytes content")
        else:
            raise ValueError("Invalid input type. Expected UploadFile or bytes.")
        return content, filename

    def _create_temp_file(self, content, filename):
        temp_dir = f"/tmp/{filename}_temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, filename)
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(content)
        return temp_file_path

    def _prepare_documents(self, text, filename):
        return [Document(text=text, metadata={"file_name": filename})]

    def _create_chunks(self, documents):
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

        chunks = []
        for doc in documents:
            chunks.extend(chunk_text_with_metadata(doc.text, file_name=doc.metadata["file_name"]))
        return chunks
           
           
           
    def periodic_full_rebuild(self):
        """
        Teljes similarity újragenerálása időszakosan.
        """
        self.rebuild_similarity_index()
        self.create_similarity_relationships(top_k=50, threshold=0.8)
        logging.info("Teljes hasonlósági gráf újragenerálva.")
    

