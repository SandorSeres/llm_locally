#
# A választ streaming módban küldi a kliensnek, így chat módban használható
# https://platform.openai.com/docs/api-reference
#
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import Any, Dict
from typing import AsyncGenerator
import httpx
from starlette.responses import FileResponse 
from typing import List
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import Request, HTTPException
import time
import logging
import dotenv
import os
import openai
import json
from threading import Lock

from starlette.middleware.sessions import SessionMiddleware
from datetime import timedelta

dotenv.load_dotenv("./.env")
logger = logging.getLogger(__name__)

# Statikus fájlok könyvtárának csatolása

app = FastAPI()

app.mount("/static", StaticFiles(directory=os.path.join(os.getcwd(), "static")), name="static")
app.add_middleware(SessionMiddleware, secret_key="sas")

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

# Tároló a session-oknak
SESSION_FILE = "/app/sessions.json"
file_lock = Lock()

def create_session() -> str:
    """Új session létrehozása."""
    session_id = str(uuid4())
    print("create_session")
    sessions = load_sessions()
    print(sessions)
    sessions[session_id] = {"history": []}
    save_sessions(sessions)
    print("saved")
    return session_id

def get_session(session_id: str) -> dict:
    """Session adatainak lekérése."""
    print("get_session")
    sessions = load_sessions()
    print(sessions)
    return sessions.get(session_id, None)

def save_session(session_id: str, session_data: dict):
    """Session adatainak frissítése."""
    sessions = load_sessions()
    sessions[session_id] = session_data
    save_sessions(sessions)


def load_sessions() -> dict:
    """Session adatok betöltése a fájlból."""
    with file_lock:
        if not os.path.exists(SESSION_FILE):
            return {}
        with open(SESSION_FILE, "r", encoding="utf-8") as file:
            return json.load(file)

def save_sessions(sessions: dict):
    """Session adatok mentése a fájlba."""
    with file_lock:
        with open(SESSION_FILE, "w", encoding="utf-8") as file:
            json.dump(sessions, file, indent=4)



# Egy egyszerű modell a kérésekhez és válaszokhoz
class QueryModel(BaseModel):
    query: str

class ResponseModel(BaseModel):
    answer: str
    metadata: str

headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}


#response = openai.chat.completions.create(**body)

async def generate_response_stream(query: str, session_id : str, session_data: dict):
    # Az előző üzenetek hozzáadása az új üzenethez
    history = session_data.get("history", [])
    history.append({"role": "user", "content": query})
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
            if content:  # Csak akkor adjuk hozzá, ha nem None
                # A válasz hozzáadása a session history-hoz
                #history.append({"role": "assistant", "content": content})
                full_response += content
                yield f"data: {json.dumps({'choices': [{'delta': {'content': content}}]})}\n\n"

    # Iterátor vége: A teljes válasz hozzáadása a history-hoz
    if full_response.strip():  # Ellenőrizzük, hogy nem üres-e
        history.append({"role": "assistant", "content": full_response})

    # Frissített history mentése a session-be
    session_data["history"] = history    
    save_session(session_id, session_data)


@app.post("/generate")
async def generate(query: QueryModel, request: Request):
    # Cookie-ból session ID lekérése
    session_id = request.cookies.get("session_id")
    print(session_id)
    if not session_id:
        raise HTTPException(status_code=403, detail="Invalid or missing session")

    # Hozzáférés a session adatokhoz
    session_data = get_session(session_id)
    if not session_data:
        raise HTTPException(status_code=403, detail="Session not found")
    print(f"Using session data: {session_data}")

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
    response = FileResponse('/app/static/streaming.html')
    response.set_cookie(key="session_id", value=session_id, httponly=True, path="/")
    return response


if __name__ == "__main__":

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
