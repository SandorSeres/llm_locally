import os  # Környezeti változók kezeléséhez
import json  # JSON válaszok feldolgozásához
from typing import List, Dict, AsyncGenerator  # Típusannotációkhoz
import httpx  # HTTP kérésekhez
from fastapi import HTTPException  # Hibakezeléshez (FastAPI projektek esetén)
import logging  # Naplózáshoz
logger = logging.getLogger(__name__)

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
        """
        Ollama modell streaming.
        A választ OpenAI-kompatibilis formátumra alakítjuk.
        """
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        async with httpx.AsyncClient(timeout=600) as client:
            try:
                logger.info(f"model: {self.model_name}, prompt: {prompt}, 'stream': {True}", exc_info=True)
                response = await client.post(
                    self.ollama_api_url,
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "options": {"num_ctx": 60000},
                        "stream": True
                    },
                    timeout=600
                )
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue

                    #logger.info(f"Streamed line: {line}", exc_info=True)  # Ellenőrzés

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning(f"Ollama válasz nem JSON: {line}", exc_info=True)
                        continue

                    # Stream végének kezelése
                    if data.get("done"):
                        yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}
                        break

                    # Tartalom lekérése az Ollama válaszból és átalakítás
                    content = data.get("response", "")
                    if content:
                        yield {
                            "choices": [
                                {
                                    "delta": {"content": content},
                                    "finish_reason": None
                                }
                            ]
                        }
            except httpx.RequestError as e:
                logger.error(f"Ollama request error: {e}", exc_info=True)
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
        logger.info(messages, exc_info=True)
        logger.info(f"Body: {body} Header: {headers} URL: {self.openai_api_url}", exc_info=True)
        async with httpx.AsyncClient(timeout=600) as client:
            async with client.stream("POST", self.openai_api_url, headers=headers, json=body) as response:
                #if response.status_code != 200:
                #    raise HTTPException(status_code=response.status_code, detail="Error from OpenAI API")
                async for chunk in response.aiter_bytes():
                    #print(time.time_ns() ,chunk)
                    yield chunk

    async def generate_complete(self, messages: List[Dict[str, str]]) -> str:
        """
        A kiválasztott modellnek elküldi a kérést és a teljes választ adja vissza egyben.
        """
        if self.model_type == "ollama":
            return await self._generate_ollama_complete(messages)
        elif self.model_type == "openai":
            return await self._generate_openai_complete(messages)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    async def _generate_ollama_complete(self, messages: List[Dict[str, str]]) -> str:
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        async with httpx.AsyncClient(timeout=600) as client:
            response = await client.post(
                self.ollama_api_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "options": {"num_ctx": 60000}
                },
                timeout=600
            )
            
            result = ""
            async for line in response.aiter_lines():
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    result += data.get("response", "")
                except json.JSONDecodeError:
                    logging.warning(f"Skipping non-JSON line: {line}")
            
            return result

    async def _generate_openai_complete(self, messages: List[Dict[str, str]]) -> str:
        """
        OpenAI modell teljes válasz generálása.
        """
        headers = {"Authorization": f"Bearer {self.openai_api_key}"}
        body = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.1,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        }
        async with httpx.AsyncClient(timeout=600) as client:
            response = await client.post(self.openai_api_url, headers=headers, json=body, timeout=None)
            response_data = response.json()
            if response.status_code != 200 or "choices" not in response_data:
                raise ValueError(f"OpenAI API hiba: {response.text}")
            return response_data["choices"][0]["message"]["content"]


