# Standard könyvtárak
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, UploadFile
from fastapi import File,  HTTPException
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import aiofiles
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor

from contextlib import asynccontextmanager
from typing import Any, Dict, AsyncGenerator, Optional, List
from pydantic import BaseModel, ConfigDict
from uuid import uuid4
from datetime import timedelta
from neo4j.exceptions import ServiceUnavailable

#############Special own package import because of compiled code #####
import sys 
import importlib.util
# Python verzió meghatározása
python_version = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
print("Python verzió:", python_version)

# Fájl elérési utak generálása
model_manager_path = f"/app/__pycache__/model_manager.{python_version}.pyc"
neo4jrag_path = f"/app/__pycache__/neo4jrag.{python_version}.pyc"

# Ellenőrizzük a fájlok létezését
if not os.path.exists(model_manager_path):
    raise FileNotFoundError(f"{model_manager_path} nem található!")
if not os.path.exists(neo4jrag_path):
    raise FileNotFoundError(f"{neo4jrag_path} nem található!")

# Betöltjük a model_manager modult
spec_model_manager = importlib.util.spec_from_file_location("model_manager", model_manager_path)
model_manager = importlib.util.module_from_spec(spec_model_manager)
spec_model_manager.loader.exec_module(model_manager)

# Betöltjük a neo4jrag modult
spec_neo4jrag = importlib.util.spec_from_file_location("neo4jrag", neo4jrag_path)
neo4jrag = importlib.util.module_from_spec(spec_neo4jrag)
spec_neo4jrag.loader.exec_module(neo4jrag)

print("Modulok sikeresen betöltve.")
#####################################################################
import httpx
import time
import logging
import dotenv
import os
import json
import asyncio
from io import BytesIO


dotenv.load_dotenv("./.env")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Globális változók

model_options = {
    "ollama": ["llama3.2","gemma2","mistral-nemo"],
    "openai": ["gpt-4o", "gpt-4o-mini"]
}

rag = None
session_manager = None
neo4j_manager = None
topic_manager = None


# FastAPI lifecycle események
@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag, session_manager, neo4j_manager, topic_manager
    logger.info("App startup", exc_info=True)
    try:
        time.sleep(10)  # Biztosítja, hogy a Neo4j már elindult
        # Neo4jManager inicializálása
        neo4j_manager = neo4jrag.Neo4jManager(
            url=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password"),
        )
        logger.info("Neo4jManager initialized successfully", exc_info=True)

        # SessionManager inicializálása
        session_manager = neo4jrag.SessionManager(neo4j_manager)
        logger.info("SessionManager initialized successfully", exc_info=True)

        # TopicManager inicializálása
        topic_manager = neo4jrag.TopicManager(neo4j_manager)
        logger.info("TopicManager initialized successfully", exc_info=True)

        # VectorStoreManager inicializálása
        rag = neo4jrag.VectorStoreManager(neo4j_manager, topic_manager)  # Átadjuk a TopicManager példányt
        logger.info("VectorStoreManager initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize components: {e}", exc_info=True)
        rag = None
        session_manager = None
        topic_manager = None
        neo4j_manager = None

    yield

    # Shutdown logika
    try:
        if neo4j_manager:
            neo4j_manager.close()
            logger.info("Neo4jManager closed successfully", exc_info=True)
    except Exception as e:
        logger.error(f"Error during shutdown: {e}", exc_info=True)

    logger.info("App shutdown", exc_info=True)

app = FastAPI(lifespan=lifespan)

# Statikus fájlok és CORS beállítása
app.mount("/static", StaticFiles(directory=os.path.join(os.getcwd(), "static")), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templétek elérési útja
templates = Jinja2Templates(directory="templates")


class QueryModel(BaseModel):
    query: str
    model_type: Optional[str] = None
    model_name: Optional[str] = None

    model_config = ConfigDict(protected_namespaces=())
    
class ResponseModel(BaseModel):
    answer: str
    metadata: str

def create_session() -> str:
    """Új session létrehozása."""
    global session_manager
    session_id = str(uuid4())
    session_data = {"history": []}
    try:
        session_manager.save_session(session_id, session_data)
    except ServiceUnavailable:
        raise HTTPException(status_code=500, detail="Database unavailable.")
    return session_id

def get_session(session_id: str) -> dict:
    """Session lekérése."""
    global session_manager
    try:
        session_data = session_manager.get_session(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found.")
        return session_data
    except ServiceUnavailable:
        raise HTTPException(status_code=500, detail="Database unavailable.")

async def generate_response_stream(query: str, session_id: str, session_data: dict, model_type: str, model_name: str):
    try:
        rag_context = await rag.search(query, k=3)
    except Exception as e:
        rag_context = "No relevant context found in RAG database."
        logger.warning(f"RAG search failed: {e}", exc_info=True)

    history = session_data.get("history", [])

    messages = history.copy()
    messages.append({
        "role": "system",
        "content": (
            "Answer the user question using the information in the context. "
            "If no context is available, use your own info.\n"
            f"Context:\n{rag_context}\n"
            "You shouls all the time tell the source of the information\n"
            "example:\n"
            "<ANSware>\n"
            "Source: <file1>, <file2>"
             
        )
    })
    messages.append({"role": "user", "content": query})

    local_model_manager = model_manager.ModelManager(
        model_type=model_type or os.getenv("MODEL_TYPE", "ollama"),
        model_name=model_name or os.getenv("MODEL_NAME", "llama3.2")
    )

    full_response = ""

    async for chunk in local_model_manager.generate_stream(messages):
        if isinstance(chunk, bytes):
            decoded = chunk.decode("utf-8", errors="ignore")
            lines = decoded.strip().split("\n")
            logger.info(f"lines:{lines}", exc_info=True)
            for line in lines:
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        # A stream vége
                        break
                    try:
                        data_json = json.loads(data_str)
                        # OpenAI formátum: data_json["choices"][0]["delta"]["content"]
                        choices = data_json.get("choices", [])
                        if choices and "delta" in choices[0]:
                            content = choices[0]["delta"].get("content", "")
                            full_response += content
                            # Küldjük a klienseknek is
                            yield f"{line}\n"
                    except json.JSONDecodeError:
                        # Ha nem JSON, átugorjuk
                        pass
        else:
            # Ollama esetén a chunk dict, pl. {"choices": [{"delta": {"content": "..."}}]}
            # Itt is kinyerjük a contentet és hozzáfűzzük a full_response-hoz
            content = ""
            if "choices" in chunk and chunk["choices"]:
                delta = chunk["choices"][0].get("delta", {})
                c = delta.get("content", "")
                content += c
            full_response += content
            yield f"data: {json.dumps(chunk)}\n\n"

    # Miután vége a streamnek, frissítjük a history-t a felhasználó és az asszisztens üzenetével
    if full_response.strip():
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": full_response})
        session_data["history"] = history
        session_manager.save_session(session_id, session_data)


@app.post("/generate")
async def generate(query: QueryModel, request: Request):
    session_id = request.cookies.get("session_id")
    if not session_id:
        session_id = create_session()

    session_data = get_session(session_id)
    if not session_data:
        raise HTTPException(status_code=403, detail="Session not found")
    logger.info(f"Using session data: {session_data}", exc_info=True)

    return StreamingResponse(
        generate_response_stream(query.query, session_id, session_data, query.model_type, query.model_name),
        media_type="text/event-stream"
    )

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    # Új session létrehozása
    session_id = create_session()
    
    # TemplateResponse létrehozása
    response = templates.TemplateResponse("index.html", {
        "request": request,
        "session_id": session_id,
        "model_options": model_options
    })
    
    # Cookie beállítása a válaszban
    response.set_cookie(key="session_id", value=session_id, httponly=True, path="/")
    return response
        
@app.post("/upload_topics/")
async def upload_topics(file: UploadFile):
    try:
        filepath = f"/tmp/{file.filename}"
        with open(filepath, "wb") as buffer:
            buffer.write(await file.read())
        new_topics = topic_manager.load_topics_from_file(filepath)
        topic_manager.create_topics_in_neo4j()
        return {"message": f"{len(new_topics)} topics uploaded successfully."}
    except Exception as e:
        logging.error(f"Error uploading topics: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error uploading topics.")
               
@app.get("/upload/")
async def get_upload_form():
    """
    Visszaadja az upload.html fájlt.
    """
    try:
        return HTMLResponse(content=open('/app/static/upload.html').read(), status_code=200)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="upload.html not found")



async def process_file(file_path: str, original_filename: str) -> dict:
    """Aszinkron fájl feldolgozás."""
    try:
        async with aiofiles.open(file_path, 'rb') as file:
            content = await file.read()
            # Aszinkron művelet meghívása
            await rag.upload_document(content, filename=original_filename)
        return {"filename": original_filename, "status": "success"}
    except Exception as e:
        logger.error(f"Error processing {original_filename}: {str(e)}", exc_info=True)
        return {"filename": original_filename, "status": f"error: {str(e)}"}
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

async def process_files_background(file_paths: List[tuple]):
    """Aszinkron háttérfolyamat a fájlok feldolgozására."""
    for file_path, original_filename in file_paths:
        result = await process_file(file_path, original_filename)
        logger.info(f"{result['filename']} stored with status: {result['status']}", exc_info=True)
    logger.info("File upload process finished.", exc_info=True)

@app.post("/upload/")
async def upload_files(files: List[UploadFile], background_tasks: BackgroundTasks):
    """Fájlok feltöltése és háttérfolyamat indítása."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    temp_file_paths = []

    for file in files:
        # Fájl név és kiterjesztés kezelése
        filename, extension = os.path.splitext(file.filename)
        temp_path = f"temp_{filename}{extension}"  # Temp fájl név a megfelelő kiterjesztéssel

        async with aiofiles.open(temp_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)

        logger.info(f"Temp file created: {temp_path}")
        temp_file_paths.append((temp_path, file.filename))

    # Háttérfolyamat indítása aszinkron módon
    background_tasks.add_task(process_files_background, temp_file_paths)

    return {"message": "File processing started in the background"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

