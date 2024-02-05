from transformers import    AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from transformers import pipeline
import time
import torch
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse 
import logging
import json
import torch.nn.functional as F


logger = logging.getLogger(__name__)
MAX_TOKEN = 1024
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

#MODEL= "TinyLlama/TinyLlama-1.1B-intermediate-step-480k-1T"
MODEL= "stabilityai/stablelm-2-1_6b-zephyr"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    load_in_8bit = False, 
    llm_int8_enable_fp32_cpu_offload= True,
    bnb_4bit_compute_dtype=torch.float16, #torch.float32,
    bnb_4bit_quant_type=  "nf4", # "fp4", #
    bnb_4bit_use_double_quant=True,
)


tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(  MODEL, device_map=DEVICE,  **{"quantization_config": quantization_config, 'trust_remote_code' : True} )

# Egy egyszerű modell a kérésekhez és válaszokhoz
class QueryModel(BaseModel):
    query: str

class ResponseModel(BaseModel):
    answer: str
    metadata: str

def stop_condition_met(predictions, tokenizer, threshold=0.5):
    # softmax alkalmazása a logits-ra az utolsó dimenzió mentén
    probabilities = F.softmax(predictions[:, -1, :], dim=-1)
    # az eos_token_id valószínűségének kikeresése
    eos_probability = probabilities[0, tokenizer.eos_token_id].item()  # `.item()` konvertálja a tensor értéket Python floattá
    
    # Rendezzük a valószínűségeket csökkenő sorrendben és keressük meg az eos_token_id indexét
    sorted_probs, indices = torch.sort(probabilities[0], descending=True)
    eos_position = (indices == tokenizer.eos_token_id).nonzero(as_tuple=True)[0].item() + 1  # +1, mert az indexelés 0-tól kezdődik, de mi 1-től szeretnénk számolni

    #print(f"EOS token valószínűsége a {eos_position}. legnagyobb.")

    return eos_position < 2

def generate_response_stream(input_text: str, threshold=0.5):
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(DEVICE)
    output_ids = input_ids
    model.eval()
    previous_text = ''  # Az előző szöveg tárolása.

    with torch.no_grad():
        for _ in range(MAX_TOKEN):
            outputs = model(output_ids)
            predictions = outputs.logits
            probabilities = F.softmax(predictions[:, -1, :], dim=-1)
            next_token_id = torch.argmax(probabilities, dim=-1, keepdim=True)

            # Itt biztosítjuk, hogy a next_token_id megfelelő dimenzióval rendelkezzen
            next_token_id = next_token_id.unsqueeze(0)  # Hozzáadjuk a batch dimenziót, ha szükséges

            if stop_condition_met(predictions, tokenizer, threshold):
                break

            # Az output_ids és a next_token_id dimenzióinak ellenőrzése és összefűzése
            if len(next_token_id.shape) == 3:  # Ha a next_token_id valóban 3 dimenziós lett
                next_token_id = next_token_id.squeeze(-1)  # Eltávolítjuk a felesleges dimenziót

            output_ids = torch.cat([output_ids, next_token_id], dim=-1)

            current_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)

            # Az előző szöveg frissítése és az új delta szöveg küldése hiányzik a bemutatott kódból
            # A következő sorokat szükséges hozzáadni a kódodhoz, hogy a 'previous_text' változót megfelelően kezeljük
            delta_text = current_text[len(previous_text):] if previous_text else current_text
            previous_text = current_text  # Frissítjük az előző szöveg változót az aktuális szövegre
            #print(delta_text)
            yield f"data: {json.dumps({'choices': [{'delta': {'content': delta_text}}]})}\n\n"

@app.post("/generate", response_model=List[ResponseModel])
async def generate(query: QueryModel):
    return StreamingResponse(generate_response_stream(query.query), media_type="application/json") # "text/event-stream") # 

@app.get("/")
async def read_index():
    return FileResponse('./streaming.html')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)