from transformers import    AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import torch
from fastapi import FastAPI
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
app = FastAPI(
    title="Local LLM API",
    description="""This is a sample API to demonstrate how to run LLM locally using GPU with max 8G VRAM.</br>
    Currently using 'HuggingFaceH4/zephyr-7b-beta' with quantization, so using less than 6G VRAM</br>
    Using prompt template to define the language to Hungarian. </br>
    The client receive, and present the generated text in a streaming mode (Server-Send-Event)</br>
    The full generation time also measured.
    """,
    version="1.0.0",
)
# Statikus fájlok könyvtárának csatolása (ha van image)
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS middleware hozzáadása    
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Engedélyezett források (biztonsági okokból szűkíthető!)
    allow_credentials=True,
    allow_methods=["*"],  # Engedélyezett metódusok
    allow_headers=["*"],  # Engedélyezett fejlécek
)

#MODEL= "TinyLlama/TinyLlama-1.1B-intermediate-step-480k-1T"
#MODEL= "stabilityai/stablelm-2-1_6b-zephyr"
MODEL= "HuggingFaceH4/zephyr-7b-beta"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# Kvantálás paraméterei, hogy beleférjen a GPU-ba
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    load_in_8bit = False, 
    llm_int8_enable_fp32_cpu_offload= True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type=  "nf4",
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(  MODEL, device_map=DEVICE,  **{"quantization_config": quantization_config, 'trust_remote_code' : True} )

# Egy egyszerű osztály definiálása a kérésekhez és válaszokhoz
class QueryModel(BaseModel):
    query: str

class ResponseModel(BaseModel):
    answer: str
    metadata: str
#
# A generált szöveg végének a megállapítása
#
def stop_condition_met(predictions, tokenizer, threshold=0.5):
    # softmax alkalmazása a logits-ra az utolsó dimenzió mentén
    probabilities = F.softmax(predictions[:, -1, :], dim=-1)
    # Rendezzük a valószínűségeket csökkenő sorrendben és keressük meg az eos_token_id indexét
    sorted_probs, indices = torch.sort(probabilities[0], descending=True)
    eos_position = (indices == tokenizer.eos_token_id).nonzero(as_tuple=True)[0].item() + 1  # +1, mert az indexelés 0-tól kezdődik, de mi 1-től szeretnénk számolni
    #print(f"EOS token valószínűsége a {eos_position}. legnagyobb.")
    return eos_position < 2

def generate_response_stream(input_text: str, threshold=0.5):
    # Prompt template definiálása. (Szükséges, mert különben nem működik  magyar nyelven!)
    prompt_template= "<|system|>Minden kérdésre CSAK magyarul válaszolj!\n<|user|>{query_str}\n<|assistant|>\n\n" 
    # Bemeneti szöveg előkészítése a template használatával
    formatted_input = prompt_template.format(query_str=input_text)  # A lekérdezés beillesztése a template-be
    input_ids = tokenizer.encode(formatted_input, return_tensors="pt").to(DEVICE)
    output_ids = input_ids
    model.eval()
    previous_text = ''  # Az előző szöveg tárolása.

    # Streamingelés miatt nem használhatjuk a model.generate() funkciót, hanem tokenenként kell generálni. (az is ezt csinálja, csak nem tudok közbenső adatokat elkapni)
    with torch.no_grad():
        for _ in range(MAX_TOKEN):
            # modell call metódusát hívjuk meg
            outputs = model(output_ids)
            predictions = outputs.logits
            probabilities = F.softmax(predictions[:, -1, :], dim=-1)
            next_token_id = torch.argmax(probabilities, dim=-1, keepdim=True)

            # Itt biztosítjuk, hogy a next_token_id megfelelő dimenzióval rendelkezzen
            next_token_id = next_token_id.unsqueeze(0)  # Hozzáadjuk a batch dimenziót, ha szükséges

            if stop_condition_met(predictions, tokenizer, threshold):
                print('---------------END-------------------------') # Itt van a generált szöveg vége.
                break

            # Az output_ids és a next_token_id dimenzióinak ellenőrzése és összefűzése
            if len(next_token_id.shape) == 3:  # Ha a next_token_id valóban 3 dimenziós lett
                next_token_id = next_token_id.squeeze(-1)  # Eltávolítjuk a felesleges dimenziót

            output_ids = torch.cat([output_ids, next_token_id], dim=-1)
            current_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)

            # Az előző szöveg frissítése és az új delta szöveg küldése
            delta_text = current_text[len(previous_text):] if previous_text else current_text
            previous_text = current_text  # Frissítjük az előző szöveg változót az aktuális szövegre
            # Csak az első <|assistant|>\n\n utáni szöveg küldése
            if '<|assistant|>\n\n' in delta_text:
                response_text = delta_text.split('\n\n', 1)[-1]  # A válaszszöveg megszűrése
                yield f"data: {json.dumps({'choices': [{'delta': {'content': response_text}}]})}\n\n"
            else:
                yield f"data: {json.dumps({'choices': [{'delta': {'content': delta_text}}]})}\n\n"


@app.post("/generate", response_model=List[ResponseModel], summary="Generate Text based on the user request", description="Generates text based on the provided query.")
async def generate(query: QueryModel):
    return StreamingResponse(generate_response_stream(query.query), media_type="application/json")

@app.get("/", summary="Send back the client UI.", description="This API call sends back the user UI in HTML all used Java Scripts and css included.</br>")
async def read_index():
    return FileResponse('./streaming.html')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)