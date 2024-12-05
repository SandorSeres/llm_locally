#
# A választ streaming módban küldi a kliensnek, így chat módban használható
# https://platform.openai.com/docs/api-reference
#
# Standard könyvtárak
#
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import Request, HTTPException
from fastapi import UploadFile
from contextlib import asynccontextmanager
from typing import Any, Dict
from typing import AsyncGenerator
from typing import Optional
from typing import List
import httpx
from starlette.responses import FileResponse , HTMLResponse
from pydantic import BaseModel

import time
import logging
import dotenv
import os
import openai
import json


#from starlette.middleware.sessions import SessionMiddleware
from datetime import timedelta
from neo4j.exceptions import ServiceUnavailable

# Saját modulok
from neo4jrag import * 


dotenv.load_dotenv("./.env")
logger = logging.getLogger(__name__)

# Globális Neo4jRAG példány
rag = None 
session_manager = None
neo4j_manager = None 

# FastAPI lifecycle események
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logika
    global rag , session_manager , neo4j_manager # Globális változó használata
    logger.info("App startup")
    try:
        time.sleep (10)
        neo4j_manager = Neo4jManager(
            url=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password"),  # Biztonsági okokból jobb környezeti változóból
        )
        # Inicializáljuk a session és vector store kezelőket
        session_manager = SessionManager(neo4j_manager)
        logger.info("Session manager started")
        rag = VectorStoreManager(neo4j_manager)
        logger.info("VectorStoreManager started")
    except Exception as e:
        logger.error(f"Failed to initialize Neo4jRAG: {e}")
        rag = None
        session_manager = None
    yield
    # Shutdown logika
    logger.info("App shutdown")

app = FastAPI(lifespan=lifespan)


# Statikus fájlok könyvtárának csatolása
app.mount("/static", StaticFiles(directory=os.path.join(os.getcwd(), "static")), name="static")
#app.add_middleware(SessionMiddleware, secret_key="sas")

# CORS middleware hozzáadása    
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Engedélyezett források (biztonsági okokból szűkítsd le!)
    allow_credentials=True,
    allow_methods=["*"],  # Engedélyezett metódusok
    allow_headers=["*"],  # Engedélyezett fejlécek
)

MODEL = os.environ["MODEL"]
OPENAI_API_KEY=os.environ["OPENAI_API_KEY"]


from uuid import uuid4
from fastapi import Request


# Új session létrehozása
def create_session() -> str:
    global rag  # Globális változó használata
    session_id = str(uuid4())
    session_data = {"history": []}
    try:
        session_manager.save_session(session_id, session_data)
    except ServiceUnavailable as e:
        raise HTTPException(status_code=500, detail="Database unavailable.")
    return session_id

# Session lekérése
def get_session(session_id: str) -> dict:
    global rag  # Globális változó használata
    try:
        session_data = session_manager.get_session(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found.")
        return session_data
    except ServiceUnavailable as e:
        raise HTTPException(status_code=500, detail="Database unavailable.")


# Egy egyszerű modell a kérésekhez és válaszokhoz
class QueryModel(BaseModel):
    query: str

class ResponseModel(BaseModel):
    answer: str
    metadata: str


#response = openai.chat.completions.create(**body)

async def generate_response_stream(query: str, session_id : str, session_data: dict):
    """
    Generate a response stream for a user query using RAG and OpenAI API.

    Args:
        query: The user query.
        session_id: The user's session ID.
        session_data: Session data containing previous history.
    """

    """
    """
    # RAG keresés
    global rag  # Globális változó használata
    try:
        rag_context = rag.search(query, k=3)
    except Exception as e:
        rag_context = "No relevant context found in RAG database."
        logger.warning(f"RAG search failed: {e}")
    history = session_data.get("history", [])

    history.append({"role": "system", "content": f"""Answer the user question only the information in the context! If no context, then say "Sorry I have no information." on the question language.\n
Query: {query}
Context: {rag_context}"""})
    history.append({"role": "system", "content": query})

    # Teljes válasz összegyűjtésére
    full_response = ""

    body = {
        "model": MODEL,
        "messages": history,
        "stream": True,
        "temperature": 0.1,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }

    # OpenAI streaming hívás
    response = openai.chat.completions.create(**body)

    for chunk in response:
        if chunk.choices and chunk.choices[0].delta and hasattr(chunk.choices[0].delta, "content"):
            content = chunk.choices[0].delta.content
            if content:  
                full_response += content
                yield f"data: {json.dumps({'choices': [{'delta': {'content': content}}]})}\n\n"

    # Iterátor vége: A teljes válasz hozzáadása a history-hoz
    if full_response.strip():
        history.append({"role": "assistant", "content": full_response})
        session_data["history"] = history
        session_manager.save_session(session_id, session_data)


@app.post("/generate")
async def generate(query: QueryModel, request: Request):
    # Cookie-ból session ID lekérése
    session_id = request.cookies.get("session_id")
    if not session_id:
        # Új session létrehozása
        session_id = create_session()

    # Hozzáférés a session adatokhoz
    session_data = get_session(session_id)
    if not session_data:
        raise HTTPException(status_code=403, detail="Session not found")
    logger.info(f"Using session data: {session_data}")

    # Indítsd el a streaming válasz generálását
    return StreamingResponse(
        generate_response_stream(query.query, session_id, session_data),
        media_type="text/event-stream"
    )
    
@app.get("/")
async def read_index():
    # Új session létrehozása
    session_id = create_session()

    # Fájl kiszolgálása session cookie-val
    response = FileResponse('/app/static/index.html')
    response.set_cookie(key="session_id", value=session_id, httponly=True, path="/")
    return response

# FastAPI endpoint to upload files
@app.get("/upload/")
async def get_upload_form():
    """
    GET endpoint to return the upload.html file for the user.
    """
    try:
        return HTMLResponse(content=open('/app/static/upload.html').read(), status_code=200)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="upload.html not found")

@app.post("/upload/")
async def upload_files(files: List[UploadFile]):
    """
    Endpoint to upload multiple documents and process them one by one.

    Args:
        files: List of uploaded files from the HTTP request.

    Returns:
        Success message with file details or raises error if processing fails.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    results = []
    for file in files:
        try:
            await rag.upload_document(file)
            results.append({"filename": file.filename, "status": "success"})
        except Exception as e:
            results.append({"filename": file.filename, "status": f"error: {str(e)}"})
    return {"results": results}        
 
         
if __name__ == "__main__":

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
