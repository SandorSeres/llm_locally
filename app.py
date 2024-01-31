# API 
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List
import uvicorn
from pydantic import BaseModel
from starlette.responses import FileResponse 
import logging
import sys
import os
from fastapi import UploadFile, File
from fastapi.responses import HTMLResponse
import shutil
import tempfile


# LLM
import warnings
from langchain.embeddings import LangChainDeprecationWarning
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

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

@app.post("/analyze", response_model=List[ResponseModel])
async def analyze(query: QueryModel):
    try:
        result = analyze_document(query.query)
        return [result]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/uploadfiles")
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
    MODEL=  "HuggingFaceH4/zephyr-7b-beta" #  "HuggingFaceH4/zephyr-7b-alpha"  
    # "TheBloke/zephyr-7B-beta-GPTQ" # -max-input-length 3696 --max-total-tokens 4096 --max-batch-prefill-tokens 4096    
    prompt_template= "<|system|>Minden kérdésre CSAK magyarul válaszolj!\n\n<|user|>\n{query_str}\n<|assistant|>\n" # "HuggingFaceH4/zephyr-7b-beta"
    #############################################
   # 'NousResearch/Yarn-Mistral-7b-128k'/'TheBloke/Yarn-Mistral-7B-128k-AWQ' 
    #prompt_template="{query_str}" # 'TheBloke/Yarn-Mistral-7B-64k-AWQ' 'TheBloke/Yarn-Mistral-7B-128k-AWQ'
    #############################################

    llm = HuggingFaceLLM(
        model_name=MODEL,
        tokenizer_name=MODEL,
        query_wrapper_prompt=PromptTemplate(prompt_template),
        context_window=32000,
        max_new_tokens=2048,
        model_kwargs={"quantization_config": quantization_config},
        generate_kwargs={"temperature": 0.1, "top_k": 50, "top_p": 0.95, "do_sample":True},
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
    return vector_index.as_query_engine()

global query_engine, service_context
# wget --method POST --header 'Content-Type: application/json' --body-data '{"query":"melyik dokumentumban említik a Da Vinci szót? Magyarul!"}' http://127.0.0.1:8000/analyze -O - &
if __name__ == "__main__":
    service_context = init_llm()
    flat_list = load_data(None)
    query_engine =  init_vector_store(service_context,flat_list)
    uvicorn.run(app, host="0.0.0.0", port=8000)
