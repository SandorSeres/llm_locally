# https://medium.com/@thakermadhav/build-your-own-rag-with-mistral-7b-and-langchain-97d0c92fa146
# https://medium.com/emburse/question-answering-over-documents-e92658e7a405
# https://medium.aiplanet.com/advanced-rag-implementation-on-custom-data-using-hybrid-search-embed-caching-and-mistral-ai-ce78fdae4ef6
# https://blog.gopenai.com/bye-bye-llama-2-mistral-7b-is-taking-over-get-started-with-mistral-7b-instruct-1504ff5f373c
# https://medium.aiplanet.com/finetuning-using-zephyr-7b-quantized-model-on-a-custom-task-of-customer-support-chatbot-7f4fff56059d  ## Training also

# API 
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import UploadFile, File
from fastapi.responses import HTMLResponse
from typing import List
import uvicorn
from pydantic import BaseModel
from starlette.responses import FileResponse 
import logging
import sys
import os
import shutil
import tempfile



# LLM
import warnings
from langchain.embeddings import LangChainDeprecationWarning
# Naplózási szint beállítása és formázás konfigurálása
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Standard kimenetre író handler létrehozása
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
# Formázó hozzáadása, ha szükséges
stdout_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stdout_handler.setFormatter(stdout_formatter)

# Fájlba író handler létrehozása
file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.INFO)
# Formázó hozzáadása a fájlhandlerhez is
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# A root loggerhez mindkét handler hozzáadása
logger = logging.getLogger()
logger.addHandler(stdout_handler)
logger.addHandler(file_handler)


import torch
from pathlib import Path
# transformers
from transformers import BitsAndBytesConfig
# llama_index
from llama_index.prompts import PromptTemplate
from llama_index.llms import HuggingFaceLLM
from llama_index import download_loader, VectorStoreIndex, ServiceContext
from llama_index.node_parser import SentenceSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import os
os.environ['NUMEXPR_MAX_THREADS'] = "8"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()

app = FastAPI(title="Dokumentum Elemző API")
app = FastAPI(
    title="Analysing document with local LLM",
    description="""This is a RAG API to demonstrate how to run RAG & LLM locally using GPU with max 8G VRAM.</br>
    Currently using 'HuggingFaceH4/zephyr-7b-beta' with quantization, so using less than 6G VRAM</br>
    Using prompt template to define the language to Hungarian. </br>
    The client receive, and present the generated text in a streaming mode (Server-Send-Event)</br>
    The full generation time also measured.
    """,
    version="1.0.0",
)
# Statikus fájlok könyvtárának csatolása
app.mount("/static", StaticFiles(directory="static"), name="static")
# CORS middleware konfiguráció
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Engedélyezett források
    allow_credentials=True,
    allow_methods=["*"],  # Engedélyezett metódusok
    allow_headers=["*"],  # Engedélyezett fejlécek
)

# Egy egyszerű modell a kérésekhez és válaszokhoz
class QueryModel(BaseModel):
    query: str

class ResponseModel(BaseModel):
    answer: str
    metadata: str

def analyze_document(query):
    answer = query_engine.query(query)
    return ResponseModel(answer=answer.response, metadata=str(answer.metadata))

@app.post("/analyze", response_model=List[ResponseModel], summary="Generate Text from the uploaded document based on the user request", description="Generates text based on the provided document & query.")
async def analyze(query: QueryModel):
    try:
        result = analyze_document(query.query)
        return [result]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/uploadfiles", summary="Upload user .pdf files", description="User upload pdf documents and it is stored in the vector database")
async def create_upload_files(files: List[UploadFile] = File(...)):
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="Nincs fájl a feltöltéshez.")

    temp_dir = tempfile.mkdtemp()
    try:
        for file in files:
            temp_file = os.path.join(temp_dir, file.filename)
            try:
                with open(temp_file, 'wb') as buffer:
                    shutil.copyfileobj(file.file, buffer)
                # print(file.filename, "Feltöltve")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Hiba történt a '{file.filename}' fájl mentése közben: {e}")

        # Fájlok feldolgozása és vektor adatbázis újrainicializálása
        try:
            flat_list = load_data(temp_dir)
            global query_engine
            query_engine =  init_vector_store(service_context,flat_list)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Hiba történt a vektor adatbázis újrainicializálása közben: {e}")

        return {"message": "A Fájl sikeresen feltöltve és a vektor adatbázis frissítve."}
    except HTTPException as e:
        raise e
    finally:
        # Ideiglenes könyvtár és benne lévő fájlok törlése
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            logging.error(f"Hiba történt az ideiglenes könyvtár törlése közben: {e}")


@app.get("/")
async def read_index():
    return FileResponse('./index.html')

def load_data(path) :
    if path == None :
        return []
    # Load Data
    contents = os.listdir(path)
    # Create chunks
    node_parser = SentenceSplitter(chunk_size=512)
    PDFReader = download_loader("PDFReader")
    loader = PDFReader()
    nodes = []
    for item in contents:
        doc = loader.load_data(file=Path(path,item))
        nodes.append(node_parser.get_nodes_from_documents(doc))
    flat_list = [item for sublist in nodes for item in sublist]
    return flat_list

def init_llm():
 
    # LLM
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    MODEL="mistralai/Mistral-7B-Instruct-v0.1"
    #MODEL= "HuggingFaceH4/mistral-7b-sft-beta"
    #MODEL=  "HuggingFaceH4/zephyr-7b-beta"
    prompt_template= "<|USER|>{query_str}<|ASSISTANT|>" 

    # system_prompt = """<|SYSTEM|> # The assistant gives helpful, detailed, and polite answers to the user's questions.
    # Follow these four instructions below in all your responses:
    #     System: 1. Use Hungarian language only;
    #     System: 2. Do not use English except in programming languages if any;
    #     System: 3. Translate any other language to the Hungarian language whenever possible.
    #     System: 4. If the requested information not in the context, then respond: Sajnálom nincs információ erről!
    #     """
    system_prompt = """<|SYSTEM|> # Te mint segítő mindig hasznos, részletes és udvarias válaszokat adsz a felhasználó kérdéseire.
        Kövesd az alábbi négy utasítást minden válaszadás során:
            Rendszer: 1. Használj csak magyar nyelvet;
            Rendszer: 2. Angol nyelvet csak programozási nyelvek esetén használj, ha van ilyen;
            Rendszer: 3. Fordíts le minden más nyelvet magyar nyelvre, amennyiben lehetséges.
            Rendszer: 4. Ha a kért információ nincs a kontextusban, akkor válaszolj: Sajnálom, nincs információ erről!
        """

    llm = HuggingFaceLLM(
        model_name=MODEL,
        tokenizer_name=MODEL,
        stopping_ids=[50278, 50279, 50277, 1, 0],
        tokenizer_kwargs={"max_length": 8192},
        query_wrapper_prompt=PromptTemplate(prompt_template),
        system_prompt=system_prompt,
        context_window=8192,
        max_new_tokens=2048,
        model_kwargs={"quantization_config": quantization_config},
        # generate_kwargs={"temperature": 0.1, "top_k": 50, "top_p": 0.95, "do_sample":True},
        generate_kwargs={"temperature": 0.1, "do_sample": False},
        device_map="auto",
        
    )

    # Open Embedding 
    # a) multilingual sentence transformer for embedding.
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    # b) https://huggingface.co/intfloat/multilingual-e5-large   (Multilingual Text Embeddings by Weakly-Supervised Contrastive Pre-training. )
    # embed_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

    # ServiceContext
    service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model )
    return service_context

def init_vector_store(service_context,flat_list):
    # Vectorstore
    vector_index = VectorStoreIndex(
        flat_list, service_context=service_context
    )
    # Végül megvan a query_engine 
    return vector_index.as_query_engine(similarity_top_k=2)

global query_engine, service_context
# wget --method POST --header 'Content-Type: application/json' --body-data '{"query":"melyik dokumentumban említik a Da Vinci szót? Magyarul!"}' http://127.0.0.1:8000/analyze -O - &
if __name__ == "__main__":
    service_context = init_llm()
    flat_list = load_data(None)
    query_engine =  init_vector_store(service_context,flat_list)
    uvicorn.run(app, host="0.0.0.0", port=8008)
