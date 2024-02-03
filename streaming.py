#
# A választ streaming módban küldi a kliensnek, így chat módban használható
# https://platform.openai.com/docs/api-reference
#
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import Any, Dict
import httpx
from starlette.responses import FileResponse 
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import time
import logging

logger = logging.getLogger(__name__)

# Statikus fájlok könyvtárának csatolása

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS middleware hozzáadása    
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Engedélyezett források (biztonsági okokból szűkítsd le!)
    allow_credentials=True,
    allow_methods=["*"],  # Engedélyezett metódusok
    allow_headers=["*"],  # Engedélyezett fejlécek
)

API_KEY = "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"  # your api key goes here
MODEL = "gpt-4-0125-preview"
MAX_TOKENS = 500

# Egy egyszerű modell a kérésekhez és válaszokhoz
class QueryModel(BaseModel):
    query: str

class ResponseModel(BaseModel):
    answer: str
    metadata: str


headers = {"Authorization": f"Bearer {API_KEY}"}

async def generate_response_stream(query: str):
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query},
        ],
        "max_tokens": MAX_TOKENS,
        "stream": True,
        "temperature": 0.1,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }
    async with httpx.AsyncClient() as client:
        async with client.stream("POST", "https://api.openai.com/v1/chat/completions", headers=headers, json=body) as response:
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail="Error from OpenAI API")
            async for chunk in response.aiter_bytes():
                # print(time.time_ns() ,chunk)
                yield chunk


@app.post("/generate", response_model=List[ResponseModel])
async def generate(query: QueryModel):
    return StreamingResponse(generate_response_stream(query.query), media_type="application/json") # "text/event-stream") # 

@app.get("/")
async def read_index():
    return FileResponse('./streaming.html')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)