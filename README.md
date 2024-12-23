### Chatbot Projekt Neo4j és Kis LLM Integrációval

Ez a projekt egy FastAPI-alapú chatbot alkalmazást valósít meg, amely integrálva van egy Neo4j adatbázissal a vektoros beágyazások (embeddings) tárolására és lekérdezésére. A chatbot kompatibilis OpenAI és Ollama konténerekkel, és támogatja a GPU-gyorsítást.

---

## Főbb Funkciók
- **Neo4j Integráció**: Dokumentumrészletek (chunk-ok) tárolása vektoros beágyazásokkal és azok keresése.
- **FastAPI Végpontok**:
  - `POST /upload`: Dokumentumok (PDF, DOCX, TXT, MD) feltöltése, feldolgozása chunk-okra és mentése a Neo4j adatbázisba.
  - `GET /`: Egy `index.html` sablont ad vissza, amely lehetővé teszi a felhasználói interakciókat.
- **Vektorindexelés**: Automatikus vektorindexek létrehozása Neo4j-ban a hasonlósági keresésekhez.
- **Grafikus Kapcsolatok**: Kapcsolatok (`BELONGS_TO`, `SIMILAR_TO`, `NEXT`) létrehozása a dokumentumrészletek között.

---

## Projekt Struktúrája
```
.
├── deploy
│   ├── build.sh
│   ├── deploy.sh
│   ├── Dockerfile
│   ├── neo4j.Dockerfile
│   └── remove.sh
├── Dockerfile
├── Dockerfile.ollama
├── neo4jrag.py
├── ollama_streaming.py
├── README.md
├── requirements.dock
├── run.sh
├── static
│   ├── hospitaly.png
│   ├── hourglass.gif
│   └── upload.html
└── templates
    └── index.html
```

---

## Előfeltételek
- **Docker és Docker Compose**
- **NVIDIA Docker** (GPU használatához)
- **Neo4j** hitelesítési adatok a `.env` fájlban:
  ```bash
  NEO4J_USERNAME=neo4j
  NEO4J_PASSWORD=your_password
  ```

---

## Telepítés és Beállítás

### 1. A Projekt Klónozása
```bash
git clone git@github.com:SandorSeres/llm_locally.git
cd llm_locally
```

### 2. Környezeti Változók
Hozz létre egy `.env` fájlt az alábbi tartalommal:
```bash
NEO4J_URI=bolt://neo4j:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
```

### 3. Build és Indítás
Az alkalmazás indítása Docker Compose segítségével:
```bash
docker-compose up --build
```
- **Neo4j böngésző**: [http://localhost:7474](http://localhost:7474)
- **FastAPI alkalmazás**: [http://localhost:8000](http://localhost:8000)
- **Feltöltési felület**: [http://localhost:8000/upload](http://localhost:8000/upload)

> **Megjegyzés**: Az új build előtt töröld a régi köteteket:
> ```bash
> docker-compose down -v
> ```

### 4. API Tesztelése
- **Dokumentum feltöltése**:
  ```bash
  curl -X POST "http://localhost:8000/upload" \
  -F "file=@example.pdf"
  ```
  vagy töltsd fel manuálisan a [http://localhost:8000/upload](http://localhost:8000/upload) felületen keresztül.

- **Főoldal elérése**:
  [http://localhost:8000](http://localhost:8000)

---

## Docker Compose Konfiguráció
A `docker-compose.yml` fájl konfigurálja a FastAPI alkalmazást és a Neo4j konténert:

```yaml
services:
  neo4j:
    image: neo4j:latest
    container_name: neo4j_vector
    ports:
      - "7474:7474"  # HTTP
      - "7687:7687"  # Bolt
    environment:
      - NEO4J_AUTH=${NEO4J_USERNAME}/${NEO4J_PASSWORD}
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
      - neo4j_import:/import
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:7474"]
      interval: 10s
      timeout: 10s
      retries: 5

  app:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: chatbot_app
    ports:
      - "8000:8000"  # FastAPI app HTTP
    depends_on:
      - neo4j
    volumes:
      - app_data:/app/data
    environment:
      - NVIDIA_VISIBLE_DEVICES=all  # GPU visibility
      - NVIDIA_DRIVER_CAPABILITIES=all
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]

volumes:
  neo4j_data:
  neo4j_logs:
  neo4j_import:
  app_data:
```

---

## Neo4j Beállítások
A `neo4jrag.py` fájl az alábbi feladatokat kezeli:
- Szövegrészletek mentése beágyazásokkal
- Vektorindexek létrehozása
- Hasonlósági keresések végrehajtása
- Kapcsolatok kiépítése a gráfban

### Kulcsfontosságú Módszerek

1. **`save_chunk`**:  
   Szövegrészletek (chunk-ok) mentése metaadatokkal és vektoros beágyazásokkal. Minden chunk egyedi azonosítót kap, amely lehetővé teszi a pontos nyomon követést és kapcsolatok építését.
2. **`create_vector_index`**:  
   Vektoros indexek automatikus létrehozása Neo4j-ben a szövegrészletek számára. Ez a funkció biztosítja a gyors és pontos hasonlósági keresést.
3. **`search_chunks`**:  
   Hasonlósági keresés végrehajtása a Neo4j adatbázisban tárolt beágyazások alapján. Az algoritmus figyelembe veszi a beágyazások közötti távolságot, hogy megtalálja a releváns chunk-okat.
4. **`create_document_relationships`**:  
   A szövegrészletek kapcsolása a forrásdokumentumokhoz. Ez a kapcsolat lehetővé teszi, hogy a rendszer azonosítsa, melyik chunk melyik dokumentumhoz tartozik, és megőrizze a dokumentumok szerkezeti logikáját.
5. **`create_similarity_relationships`**:  
   Hasonlósági kapcsolatok kiépítése a chunk-ok között a beágyazások közötti hasonlóság alapján. Ez a funkció automatikusan összekapcsolja a tartalmilag hasonló chunk-okat, segítve az összefüggő információk feltárását.
6. **`create_next_relationships`**:  
   Szomszédos kapcsolatok (`NEXT`) létrehozása a chunk-ok között, amelyek a dokumentum sorrendi logikáját tükrözik. Ez biztosítja, hogy a szöveg feldolgozása során a rendszer megőrizze a szöveg természetes sorrendjét.
7. **`generate_hypothetical_questions`**:  
   Hipotetikus kérdések generálása a chunk-ok tartalma alapján. Ezek a kérdések segítik a felhasználót a tartalom mélyebb megértésében, és javítják az interakciók minőségét.
8. **`summarize_document`**:  
   Dokumentumok automatikus összefoglalása, amely a szöveg darabolásán alapul. A funkció rövid, lényegre törő összefoglalót készít az összes feldolgozott chunk alapján.
9. **`link_chunk_to_topics`**:  
   Chunk-ok kapcsolása a releváns témákhoz (topikokhoz), amelyeket a rendszer automatikusan azonosít. Ez lehetővé teszi a téma-alapú kereséseket és az összefüggések jobb megértését.
10. **`advanced_search_with_topics`**:  
    Fejlett keresési funkció, amely ötvözi az embedding-alapú keresést, a gráfkapcsolatok elemzését, valamint a témák figyelembevételét. Ez a kombinált megközelítés biztosítja, hogy a keresési eredmények nemcsak relevánsak, hanem kontextusban gazdagok is legyenek.
11. **Topikok kezelése (`TopicManager`)**:  
  Témák betöltése fájlokból, mentése a Neo4j adatbázisba, és kapcsolatok kiépítése a releváns chunk-okkal.
12. **Kapcsolatok kombinációja a keresés során**:  
  A keresési algoritmus figyelembe veszi a dokumentumkapcsolatokat (`BELONGS_TO`), hasonlósági kapcsolatokat (`SIMILAR_TO`), és a szomszédos kapcsolatokat (`NEXT`), hogy pontos és kontextusban gazdag eredményeket biztosítson.
13. **Dokumentumok időszakos újrafeldolgozása**:  
  Lehetőség van a rendszer által tárolt adatok időszakos újrafeldolgozására, amely során frissülnek a hasonlósági kapcsolatok és a vektoros indexek.

---

### Összefoglalás

Ezek a kulcsfontosságú módszerek biztosítják, hogy a rendszer hatékonyan kezelje a dokumentumokat, azok szövegrészleteit és a hozzájuk kapcsolódó információkat. Az automatikus kapcsolatépítés és a fejlett keresési algoritmusok révén a felhasználók gyorsan és könnyen hozzáférhetnek a releváns információkhoz.
---

## GPU Támogatás
Az alkalmazás támogatja a GPU-gyorsítást az NVIDIA Docker konfigurációival. Ellenőrizd, hogy az NVIDIA driverek és a CUDA toolkit telepítve vannak-e a rendszeren.

---

## Függőségek
Fejlesztési környezetben telepíthető függőségek:
```bash
pip install -r requirements.txt
```

---

## Fontos Megjegyzések
- Támogatja az **OpenAI** és **Ollama** kis LLM-eket a beágyazások generálásához.
- Támogatott fájltípusok: `.pdf`, `.docx`, `.txt`, `.md`.
- Új build előtt futtasd:
  ```bash
  docker-compose down -v
  ```

---

## Licenc
Ez a projekt az MIT Licenc alatt érhető el.

---

## Szerző
**Sandor Seres** - [GitHub Profil](https://github.com/SandorSeres)
