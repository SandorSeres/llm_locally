# Standard könyvtárak
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, UploadFile
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os

from contextlib import asynccontextmanager
from typing import Any, Dict, AsyncGenerator, Optional, List
from pydantic import BaseModel
from uuid import uuid4
from datetime import timedelta
from neo4j.exceptions import ServiceUnavailable

import httpx
import time
import logging
import dotenv
import os
import json
import asyncio

# Saját modulok
from neo4jrag import *

dotenv.load_dotenv("./.env")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Globális változók
rag = None
session_manager = None
neo4j_manager = None

class ModelManager:
    """
    ModelManager osztály arra, hogy a választott modell (OpenAI vagy Ollama) alapján
    generáljon streamelt választ.
    """
    def __init__(self, model_type: str, model_name: str, api_url: str = None, api_key: str = None):
        self.model_type = model_type
        self.model_name = model_name
        # Ollama url/key
        self.ollama_api_url = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/chat")
        
        # OpenAI url/key        
        self.openai_api_url = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        

    async def generate_stream(self, messages: List[Dict[str, str]]) -> AsyncGenerator[dict, None]:
        """
        A kiválasztott modellnek elküldi a kérést és aszinkron stream-elve adja vissza a válasz chunk-okat.
        """
        if self.model_type == "ollama":
            async for chunk in self._generate_ollama_stream(messages):
                yield chunk
        elif self.model_type == "openai":
            async for chunk in self._generate_openai_stream(messages):
                yield chunk
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    async def _generate_ollama_stream(self, messages: List[Dict[str, str]]) -> AsyncGenerator[dict, None]:
        "       Ollama modell streaming.     "
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        async with httpx.AsyncClient() as client:
            try:
                logger.info(f"model: {self.model_name}, prompt: {prompt}, 'stream': {True}")
                response = await client.post(
                    self.ollama_api_url,
                    json={
                        "model": self.model_name, 
                        "prompt": prompt, 
                        "options": {
                                "num_ctx": 4096
                        },
                        "stream": True},
                    timeout=None
                )
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning(f"Ollama válasz nem JSON: {line}")
                        continue
                    if data.get("done"):
                        break
                    yield data

            except httpx.RequestError as e:
                logger.error(f"Ollama request error: {e}")
                raise HTTPException(status_code=500, detail="Ollama API hiba.")

    async def _generate_openai_stream(self, messages: List[Dict[str, str]]) -> AsyncGenerator[dict, None]:

        headers={"Authorization": f"Bearer {self.openai_api_key}"}
        body = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            "temperature": 0.1,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        }
        logger.info(messages)
        logger.info(f"Body: {body} Header: {headers} URL: {self.openai_api_url}")
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", self.openai_api_url, headers=headers, json=body) as response:
                #if response.status_code != 200:
                #    raise HTTPException(status_code=response.status_code, detail="Error from OpenAI API")
                async for chunk in response.aiter_bytes():
                    #print(time.time_ns() ,chunk)
                    yield chunk


# FastAPI lifecycle események
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logika
    global rag, session_manager, neo4j_manager
    logger.info("App startup")
    try:
        time.sleep(10)
        neo4j_manager = Neo4jManager(
            url=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password"),
        )
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
model_options = {
    "openai": ["gpt-4o", "gpt-4o-mini"],
    "ollama": ["hf.co/QuantFactory/EuroLLM-9B-GGUF:Q4_0", "llama3.2", "mistral", "phi3"]
}



class QueryModel(BaseModel):
    query: str
    model_type: Optional[str] = None
    model_name: Optional[str] = None

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
        rag_context = rag.search(query, k=3)
    except Exception as e:
        rag_context = "No relevant context found in RAG database."
        logger.warning(f"RAG search failed: {e}")

    history = session_data.get("history", [])

    messages = history.copy()
    messages.append({
        "role": "system",
        "content": (
            "Answer the user question using the information in the context. "
            "If no context is available, use your own info.\n"
            f"Context:\n{rag_context}\n"
        )
    })
    messages.append({"role": "user", "content": query})

    local_model_manager = ModelManager(
        model_type=model_type or os.getenv("MODEL_TYPE", "ollama"),
        model_name=model_name or os.getenv("MODEL_NAME", "mistral")
    )

    full_response = ""

    async for chunk in local_model_manager.generate_stream(messages):
        if isinstance(chunk, bytes):
            decoded = chunk.decode("utf-8", errors="ignore")
            lines = decoded.strip().split("\n")
            logger.info(f"lines:{lines}")
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
    logger.info(f"Using session data: {session_data}")

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
        
@app.get("/upload/")
async def get_upload_form():
    """
    Visszaadja az upload.html fájlt.
    """
    try:
        return HTMLResponse(content=open('/app/static/upload.html').read(), status_code=200)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="upload.html not found")

@app.post("/upload/")
async def upload_files(files: List[UploadFile]):
    """
    Több dokumentum feltöltése és feldolgozása.
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

