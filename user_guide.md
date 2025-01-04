### Rendszer Működése és Felhasználói Útmutató

#### Összetevők és Funkciók

1. **./templates/index.html**:
   - Frontend oldal, amely lehetővé teszi a felhasználóknak, hogy kérdéseket tegyenek fel a rendszernek.
   - Modellválasztó felületet kínál, ahol az OpenAI vagy Ollama típusú modellek közül választható ki az aktuálisan használandó modell.
   - Valós idejű adatfolyamot biztosít a kérdés-válasz interakciók során.
   (http://localhost:8000)

2. **./static/upload.html**:
   - Feltöltési felület, amely támogatja dokumentumok és témák feltöltését.
   - A felhasználók választhatnak, hogy dokumentumokat vagy topikokat töltenek fel, és ennek megfelelően a rendszer a megfelelő API végpontra továbbítja a feltöltést.
   (http://localhost:8000/upload)

---

### Működés Részletesen

#### 1. **Felhasználói interakciók**
   - **index.html**:
     - A felhasználó kiválasztja a modell típusát és nevét.
     - Kérdéseket ír be a szövegmezőbe, amelyeket a rendszer streaming módban válaszol meg.
   - **upload.html**:
     - A felhasználó kiválasztja a feltöltés típusát (dokumentum vagy topik).
     - Dokumentumok esetén a rendszer automatikusan darabolja a szövegeket, és feltölti azokat a Neo4j adatbázisba.

#### 2. **Backend feldolgozás**
     - A kérdésekhez egy `generate` API végpontot biztosít, amely a megadott modell alapján választ generál.
         - Embedding alapú keresést valósít meg.
         - Fejlett keresést biztosít a Neo4j-ben lévő kapcsolatok alapján.
     - A `upload` és `upload_topics` API végpontok fogadják a fájlokat, és aszinkron módon dolgozzák fel azokat.
         - A dokumentumok darabokra bontását és feltöltését végzi.
         - Hozzáadja a topikokat, és a chunk-okat topikokhoz kapcsolja.
         - Generálja a Neo4j adatbázisban a hasonlósági kapcsolatokat.
         - Chunk-ok, dokumentumok és kapcsolatok tárolása a Neo4j-ben.
         - Támogatja a `SIMILAR_TO`, `NEXT`, és `RELATED_TO` kapcsolatok létrehozását.
         - Témák beolvasása és tárolása fájlokból.
     
   **További Funkciók**
   - **Összefoglalók generálása**:
     - Dokumentumok feltöltésekor automatikusan összefoglalót készít a rendszer.
   - **Hipotetikus kérdések generálása**:
     - A chunk-okhoz releváns kérdéseket generál, amelyeket ment a Neo4j-be.
   - **Kapcsolatok építése**:
     - Chunk-ok kapcsolódása a dokumentumokhoz és topikokhoz automatikusan megtörténik.

Ezzel a felhasználók könnyen kereshetnek, tölthetnek fel adatokat, és interakcióba léphetnek a rendszerrel a kérdések megválaszolása érdekében.

---

### Felhasználói Útmutató

#### **Főoldal Használata (index.html)**

1. Nyissa meg az alkalmazást egy böngészőben.
2. Válassza ki a modell típusát (`openai` vagy `ollama`).
3. Válassza ki a konkrét modell nevét a második lenyíló menüből.
4. Írja be a kérdését, és nyomja meg a "Kérdés" gombot.
5. A válasz valós időben fog megjelenni a chatboxban.

#### **Feltöltési Felület (upload.html)**

1. Nyissa meg a feltöltési oldalt.
2. Válassza ki a feltöltés típusát (`document` vagy `topic`).
3. Kattintson a "Fájlok kiválasztása" gombra, és töltse fel a fájlokat.
4. Nyomja meg a "Feltöltés" gombot.

---

### **1. Dokumentumok Feltöltése**

A rendszerbe feltöltött fájlok (pl. PDF, DOCX, TXT, PPTX, JPG/PNG, stb.) automatikusan feldolgozásra kerülnek:

1. **Feldarabolás (Chunk-okra bontás)**:
   A dokumentumot kisebb részekre (chunk-ok) osztjuk, hogy könnyebben lehessen keresni és tárolni. Minden chunk tartalmazza:
   - A szövegrészletet
   - A dokumentum nevét
   - A chunk indexét és pozícióját a dokumentumban

2. **Embedding létrehozása**:
   A chunk-okhoz egyedi, numerikus reprezentáció (embedding) generálódik, amely lehetővé teszi a hasonlósági kereséseket.

3. **Chunk mentése**:
   A feldolgozott chunk-okat a rendszer a Neo4j adatbázisba menti, ahol minden chunk egyedi azonosítóval rendelkezik.

4. **Hipotetikus kérdések generálása**:
   Minden chunk-hoz releváns kérdések generálódnak, amelyek segítik a felhasználót az adott szöveg megértésében.

---

### **2. Kapcsolatok Kiépítése**

A dokumentumok és chunk-ok közötti kapcsolatok kiépítése segíti a keresési folyamatokat és az összefüggések megértését:

1. **Kapcsolat a dokumentummal**:
   Minden chunk kapcsolatba kerül a saját dokumentumával, így könnyen azonosítható, hogy melyik chunk melyik dokumentumhoz tartozik.

2. **Témákhoz kapcsolás**:
   A rendszer azonosítja a chunk-hoz kapcsolódó témákat, és kapcsolatot hoz létre a chunk és a témák között. Például, ha egy chunk az „Adatbázis-kezelés” témát érinti, akkor ehhez a topikhoz kapcsolódik.

3. **Hasonlósági kapcsolatok**:
   A chunk-ok közötti hasonlóságok alapján automatikusan kapcsolatok jönnek létre. Ez lehetővé teszi, hogy a rendszer az egymáshoz tartalmilag közel álló chunk-okat összekapcsolja.

4. **Szomszédos kapcsolatok (NEXT)**:
   A dokumentumokban egymás után következő chunk-ok között szomszédos kapcsolatot hoz létre a rendszer, ami a dokumentum sorrendi logikáját tükrözi.

---

### **3. Keresési Funkciók**

A keresési folyamat az adatbázisban tárolt chunk-ok és kapcsolatok kombinációját használja, hogy pontos és releváns eredményeket adjon:

1. **Embedding-alapú keresés**:
   A keresés során a lekérdezéshez tartozó embedding alapján a rendszer megkeresi a tartalmilag leginkább releváns chunk-okat.

2. **Kapcsolati gráf-alapú keresés**:
   Az embedding-alapú keresést kiegészítve a rendszer figyelembe veszi a chunk-ok közötti kapcsolatokat, például a hasonlósági és szomszédos kapcsolatokat. Ez lehetővé teszi, hogy a keresés nemcsak a tartalmilag legközelebb álló chunk-okat találja meg, hanem az azokkal kapcsolódó chunk-okat is.

3. **Téma-alapú keresés**:
   A rendszer azonosítja a lekérdezéshez tartozó témákat, és prioritásként kezeli azokat a chunk-okat, amelyek ezekhez a témákhoz kapcsolódnak.

4. **Összefoglalók és kérdések**:
   A keresési eredmények tartalmazhatnak összefoglalókat vagy hipotetikus kérdéseket, amelyek megkönnyítik a tartalom megértését és navigációját.

---

### **Felhasználási Példa**

- **Dokumentum feltöltése**:
   - A feltöltött fájl automatikusan feldolgozásra kerül, a chunk-okhoz kapcsolódó témák és kapcsolatok létrejönnek.
   - A rendszer a háttérben építi ki a hasonlósági és szomszédos kapcsolatokat, de célszerű megvárni míg a dokumentumok betöltése megtörténik.

- **Keresés indítása**:
   - A felhasználó beír egy kérdést a keresőbe.
   - A rendszer először embedding-alapú keresést végez, majd figyelembe veszi a gráfkapcsolatokat és a témákat.
   - A keresési eredmények tartalmazzák a releváns chunk-okat, összefoglalókat és témákat.

- **Eredmények**:
   - A felhasználó könnyen navigálhat a kapcsolódó tartalmak között, miközben a rendszer releváns válaszokat és információkat biztosít.

---

### **Előnyök**

- **Releváns keresési eredmények**: A hasonlósági és kapcsolati gráfok kombinációja pontosabb találatokat eredményez.
- **Téma-alapú keresés**: A témák figyelembevételével a rendszer gyorsabban és pontosabban találja meg a releváns tartalmakat.
- **Automatizált kapcsolatépítés**: A dokumentum- és chunk-kapcsolatok automatikus kiépítése csökkenti a manuális munkát.
- **Támogatott formátumok**: A rendszer széles körben támogatott fájlformátumokat dolgoz fel (PDF, DOCX, TXT, MD, JPG/PNG).

