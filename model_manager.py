import os  # Környezeti változók kezeléséhez
import json  # JSON válaszok feldolgozásához
from typing import List, Dict, AsyncGenerator, Any  # Típusannotációkhoz
import httpx  # HTTP kérésekhez
from fastapi import HTTPException  # Hibakezeléshez 
import logging  # Naplózáshoz

logger = logging.getLogger(__name__)

class ModelManager:
    """
    ModelManager osztály arra, hogy a választott modell (OpenAI vagy Ollama) alapján
    generáljon streamelt választ, teljes választ, valamint kezelje a Tool Calling funkciókat.
    """
    def __init__(self, model_type: str, model_name: str, api_url: str = None, api_key: str = None):
        self.model_type = model_type
        self.model_name = model_name
        self.embedding_size = 768
        # Ollama url/key
        self.ollama_api_url = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/chat")
        
        # OpenAI url/key        
        self.openai_api_url = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")

    async def embed_query(self, text: str) -> List[float]:
        """
        Embed szöveg generálása a kiválasztott modell alapján.
        """
        if self.model_type == "ollama":
            url = "http://ollama:11434/api/embeddings"  # Állítsd be a megfelelő címet
            payload = {"model": "nomic-embed-text", "prompt": text}
            async with httpx.AsyncClient(timeout=600) as client:
                response = await client.post(url, json=payload)
                if response.status_code == 200:
                    result = response.json()
                    return result["embedding"]
                else:
                    raise Exception(f"Ollama embedding hiba: {response.status_code}, {response.text}")
        elif self.model_type == "openai":
            # OpenAI embedding logika
            raise NotImplementedError("OpenAI embedding még nincs implementálva.")
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

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

        headers = {"Authorization": f"Bearer {self.openai_api_key}"}
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
                async for chunk in response.aiter_bytes():
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
                    logger.warning(f"Skipping non-JSON line: {line}")
            
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

    # Metódus a Tool Calling funkcióhoz
    async def generate_tool_calling(self, messages: List[Dict[str, str]], functions_or_tools: List[Dict[str, Any]]) -> str:
        """
        A kiválasztott modellnek elküldi a kérést a Tool Calling funkcióval és a teljes választ adja vissza egyben.
        """
        if self.model_type == "ollama":
            return await self._generate_ollama_tool_calling(messages, functions_or_tools)
        elif self.model_type == "openai":
            return await self._generate_openai_tool_calling(messages, functions_or_tools)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    async def _generate_openai_tool_calling(self, messages: List[Dict[str, str]], functions: List[Dict[str, Any]]) -> str:
        """
        OpenAI modell Tool Calling (Function Calling) támogatása.
        """
        headers = {"Authorization": f"Bearer {self.openai_api_key}"}
        body = {
            "model": self.model_name,
            "messages": messages,
            "functions": functions,
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

            message = response_data["choices"][0]["message"]
            # Ellenőrizzük, hogy a modell hív-e funkciót
            if "function_call" in message:
                function_call = message["function_call"]
                function_name = function_call["name"]
                arguments = json.loads(function_call["arguments"])
                result = await self.call_tool(function_name, arguments)
                # A funkció eredményét visszaadjuk a felhasználónak
                return result
            else:
                # Ha nincs funkcióhívás, akkor a modell válaszát adjuk vissza
                return message.get("content", "")

    async def _generate_ollama_tool_calling(self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]]) -> str:
        """
        Ollama modell Tool Calling támogatása.
        """
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        async with httpx.AsyncClient(timeout=600) as client:
            response = await client.post(
                self.ollama_api_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "tools": tools,
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
                except json.JSONDecodeError:
                    logger.warning(f"Skipping non-JSON line: {line}")
                    continue

                # Ellenőrizzük, hogy van-e eszközhívás
                if 'tool_calls' in data:
                    for tool_call in data['tool_calls']:
                        tool_name = tool_call['function']['name']
                        arguments = tool_call['function']['arguments']
                        # Meghívjuk a megfelelő eszközt
                        result_from_tool = await self.call_tool(tool_name, arguments)
                        # Az eszköz eredményét hozzáadjuk a válaszhoz
                        result += result_from_tool
                else:
                    # Ha nincs eszközhívás, a modell válaszát adjuk hozzá
                    result += data.get("response", "")
            
            return result

    """
    async def _generate_ollama_tool_calling(self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]]) -> str:
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        # Strukturált kimeneti séma definiálása
        structured_prompt = f"{prompt}\n\nA kimenet formátuma:\n" + json.dumps({
            "response": {
                "type": "object",
                "properties": {
                    "tool_calls": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "arguments": {"type": "object"},
                            },
                        },
                    },
                    "text": {"type": "string"},
                },
            }
        }, indent=2)
        
        async with httpx.AsyncClient(timeout=600) as client:
            response = await client.post(
                self.ollama_api_url,
                json={
                    "model": self.model_name,
                    "prompt": structured_prompt,
                    "tools": tools,
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
                    result += json.dumps(data, indent=2)  # JSON formázás
                except json.JSONDecodeError:
                    logger.warning(f"Skipping non-JSON line: {line}")
                    continue
            
            return result

    """

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Meghívja a megfelelő eszközt a megadott névvel és argumentumokkal.
        """
        # Implementálja az eszközök funkcióit itt
        if tool_name == "get_current_weather":
            city = arguments.get("city", "")
            return await self.get_current_weather(city)
        # További eszközök hozzáadása szükség esetén
        else:
            logger.warning(f"Ismeretlen eszköz: {tool_name}")
            return ""

    async def get_current_weather(self, city: str) -> str:
        """
        Példa eszköz függvény, amely lekéri az aktuális időjárást egy adott városban.
        """
        # Itt implementálhatja a tényleges API hívást egy időjárás szolgáltatáshoz
        # Demonstrációként egy példa választ adunk vissza
        return f"A(z) {city} városban jelenleg napos az idő, 25°C hőmérséklettel."


"""
Használat:

Eszközök vagy funkciók definiálása:

OpenAI esetében:

functions = [
    {
        "name": "get_current_weather",
        "description": "Lekéri az aktuális időjárást egy adott városban.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "A város neve, amelynek időjárását le kell kérni.",
                },
            },
            "required": ["city"],
        },
    },
]
Ollama esetében:

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Lekéri az aktuális időjárást egy adott városban.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "A város neve, amelynek időjárását le kell kérni.",
                    },
                },
                "required": ["city"],
            },
        },
    },
]
Kérés küldése a Tool Calling metódushoz:

OpenAI esetében:

model_manager = ModelManager(model_type="openai", model_name="gpt-3.5-turbo")
result = await model_manager.generate_tool_calling(messages, functions)
Ollama esetében:

model_manager = ModelManager(model_type="ollama", model_name="your-ollama-model")
result = await model_manager.generate_tool_calling(messages, tools)
"""
