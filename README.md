# Chatbot Project with Neo4j and Small LLM

This project provides a FastAPI-based chatbot application that integrates with a Neo4j database to store and query vector embeddings. The chatbot can work with OpenAI or Ollama containers for small LLMs and supports GPU acceleration.

## Features
- **Neo4j Integration:** Stores document chunks with embeddings and supports vector search.
- **FastAPI Endpoints:**
  - `POST /upload`: Uploads documents (PDF, DOCX, TXT, MD), processes them into chunks, and saves them to Neo4j.
  - `GET /`: Returns an `index.html` template.
- **Vector Indexing:** Automatically creates vector indices in Neo4j for similarity searches.
- **Graph Relationships:** Establishes relationships (`BELONGS_TO`, `SIMILAR_TO`, `NEXT`) between document chunks.

## Project Structure
```
# Chatbot Project with Neo4j and Small LLM

This project provides a FastAPI-based chatbot application that integrates with a Neo4j database to store and query vector embeddings. The chatbot can work with OpenAI or Ollama containers for small LLMs and supports GPU acceleration.

## Features
- **Neo4j Integration:** Stores document chunks with embeddings and supports vector search.
- **FastAPI Endpoints:**
  - `POST /upload`: Uploads documents (PDF, DOCX, TXT, MD), processes them into chunks, and saves them to Neo4j.
  - `GET /`: Returns an `index.html` template.
- **Vector Indexing:** Automatically creates vector indices in Neo4j for similarity searches.
- **Graph Relationships:** Establishes relationships (`BELONGS_TO`, `SIMILAR_TO`, `NEXT`) between document chunks.

## Project Structure
```
.
├── deploy
│   ├── build.sh
│   ├── deploy.sh
│   ├── Dockerfile
│   ├── neo4j.Dockerfile
│   └── remove.sh
├── docker-compose.yml
├── Dockerfile
├── Dockerfile.ollama
├── neo4jrag.py
├── ollama_streaming.py
├── README.md
├── requirements.dock
├── requirements.txt
├── run.sh
├── static
│   ├── hospitaly.png
│   ├── hourglass.gif
│   └── upload.html
└── templates
    └── index.html

```

## Prerequisites
- **Docker and Docker Compose**
- **NVIDIA Docker** (if using GPU)
- **Neo4j** credentials set in the `.env` file:
  ```bash
  NEO4J_USERNAME=neo4j
  NEO4J_PASSWORD=your_password
  ```

## Installation and Setup

### 1. Clone the Repository
```bash
git clone git@github.com:SandorSeres/llm_locally.git
cd llm_locally
```

### 2. Environment Variables
Create a `.env` file with the following:
```bash
NEO4J_URI=bolt://neo4j:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
```

### 3. Build and Run
Start the application using Docker Compose:
```bash
docker-compose up --build
```
- Neo4j Browser: [http://localhost:7474](http://localhost:7474)
- FastAPI Application: [http://localhost:8000](http://localhost:8000)
- FastAPI Upload: [http://localhost:8000](http://localhost:8000/upload)

> **Note:** Before rebuilding the environment, ensure to remove old volumes:
> ```bash
> docker-compose down -v
> ```

### 4. Test Endpoints
- **Upload a document:**
  ```bash
  curl -X POST "http://localhost:8000/upload" \
  -F "file=@example.pdf"
  
  or
  http://localhost:8000/upload

  ```
  
- **Access the main page:** [http://localhost:8000](http://localhost:8000)

## Docker Compose
The `docker-compose.yml` sets up the FastAPI app and Neo4j container:

```yaml
services:
  neo4j:
    image: neo4j:latest
    container_name: neo4j_vector
    ports:
      - "7474:7474"  # HTTP
      - "7687:7687"  # Bolt
    environment:
      - NEO4J_AUTH=${NEO4J_USERNAME}/${NEO4J_PASSWORD}
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
      - neo4j_import:/import
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:7474"]
      interval: 10s
      timeout: 10s
      retries: 5

  app:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: chatbot_app
    ports:
      - "8000:8000"  # FastAPI app HTTP
    depends_on:
      - neo4j
    volumes:
      - app_data:/app/data
    environment:
      - NVIDIA_VISIBLE_DEVICES=all  # GPU visibility
      - NVIDIA_DRIVER_CAPABILITIES=all
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]

volumes:
  neo4j_data:
  neo4j_logs:
  neo4j_import:
  app_data:
```

## Neo4j Setup
The `neo4j.py` handles:
- Saving text chunks with embeddings
- Creating vector indices
- Searching similar chunks
- Establishing graph relationships

### Key Methods:
1. **`save_chunk`**: Save text chunks with metadata and embeddings.
2. **`create_vector_index`**: Create a vector index in Neo4j.
3. **`search_chunks`**: Perform similarity search.
4. **`create_document_relationships`**: Link chunks to their parent documents.
5. **`create_similarity_relationships`**: Establish similarity relationships between chunks.

## GPU Support
The application supports GPU through NVIDIA Docker configurations. Ensure NVIDIA drivers and the CUDA toolkit are installed on your system.

## Dependencies
Install the Python dependencies for development:
```bash
pip install -r requirements.txt
```

## Notes
- Supports **OpenAI** and **Ollama** small LLMs for embedding generation.
- Uploads and processes `.pdf`, `.docx`, `.txt`, and `.md` files into vector chunks.
- Before rebuilding the environment, always run:
  ```bash
  docker-compose down -v
  ```

## License
This project is licensed under the MIT License.

## Author
**Your Name** - [GSandor Seres](https://github.com/SandorSeres)
```

## Prerequisites
- **Docker and Docker Compose**
- **NVIDIA Docker** (if using GPU)
- **Neo4j** credentials set in the `.env` file:
  ```bash
  NEO4J_USERNAME=neo4j
  NEO4J_PASSWORD=your_password
  ```

## Installation and Setup

### 1. Clone the Repository
```bash
git clone git@github.com:SandorSeres/llm_locally.git
cd your_project
```

### 2. Environment Variables
Create a `.env` file with the following:
```bash
NEO4J_URI=bolt://neo4j:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
```

### 3. Build and Run
Start the application using Docker Compose:
```bash
docker-compose up --build
```
- Neo4j Browser: [http://localhost:7474](http://localhost:7474)
- FastAPI Application: [http://localhost:8000](http://localhost:8000)

> **Note:** Before rebuilding the environment, ensure to remove old volumes:
> ```bash
> docker-compose down -v
> ```

### 4. Test Endpoints
- **Upload a document:**
  ```bash
  curl -X POST "http://localhost:8000/upload" \
  -F "file=@example.pdf"
  ```
- **Access the main page:** [http://localhost:8000](http://localhost:8000)

## Docker Compose
The `docker-compose.yml` sets up the FastAPI app and Neo4j container:

```yaml
services:
  neo4j:
    image: neo4j:latest
    container_name: neo4j_vector
    ports:
      - "7474:7474"  # HTTP
      - "7687:7687"  # Bolt
    environment:
      - NEO4J_AUTH=${NEO4J_USERNAME}/${NEO4J_PASSWORD}
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
      - neo4j_import:/import
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:7474"]
      interval: 10s
      timeout: 10s
      retries: 5

  app:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: chatbot_app
    ports:
      - "8000:8000"  # FastAPI app HTTP
    depends_on:
      - neo4j
    volumes:
      - app_data:/app/data
    environment:
      - NVIDIA_VISIBLE_DEVICES=all  # GPU visibility
      - NVIDIA_DRIVER_CAPABILITIES=all
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]

volumes:
  neo4j_data:
  neo4j_logs:
  neo4j_import:
  app_data:
```

## Neo4j Setup
The `neo4j.py` handles:
- Saving text chunks with embeddings
- Creating vector indices
- Searching similar chunks
- Establishing graph relationships

### Key Methods:
1. **`save_chunk`**: Save text chunks with metadata and embeddings.
2. **`create_vector_index`**: Create a vector index in Neo4j.
3. **`search_chunks`**: Perform similarity search.
4. **`create_document_relationships`**: Link chunks to their parent documents.
5. **`create_similarity_relationships`**: Establish similarity relationships between chunks.

## GPU Support
The application supports GPU through NVIDIA Docker configurations. Ensure NVIDIA drivers and the CUDA toolkit are installed on your system.

## Dependencies
Install the Python dependencies for development:
```bash
pip install -r requirements.txt
```

## Notes
- Supports **OpenAI** and **Ollama** small LLMs for embedding generation.
- Uploads and processes `.pdf`, `.docx`, `.txt`, and `.md` files into vector chunks.
- Before rebuilding the environment, always run:
  ```bash
  docker-compose down -v
  ```

## License
This project is licensed under the MIT License.

## Author
**Your Name** - [SandorSeres](https://github.com/SandorSeres)

