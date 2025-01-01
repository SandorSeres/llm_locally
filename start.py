import sys
import os
from datetime import datetime


# Környezeti változó beolvasása
exit_after_date_str = os.getenv("EXIT_AFTER_DATE")
if not exit_after_date_str:
    sys.exit("Program exitálva: Az EXIT_AFTER_DATE környezeti változó nincs beállítva.")

try:
    EXIT_AFTER_DATE = datetime.strptime(exit_after_date_str, "%Y-%m-%d")
except ValueError:
    sys.exit("Program exitálva: Az EXIT_AFTER_DATE értéke nem megfelelő formátumú (pl. YYYY-MM-DD).")

# Aktuális dátum ellenőrzése
current_date = datetime.now()
if current_date > EXIT_AFTER_DATE:
    sys.exit(f"Program exitálva: A dátum {EXIT_AFTER_DATE.strftime('%Y-%m-%d')} után van.")

# Python verzió meghatározása
py_version = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
pyc_file = f"/app/__pycache__/ollama_streaming.{py_version}.pyc"

# Ellenőrizzük, hogy létezik-e a megfelelő .pyc fájl
if os.path.exists(pyc_file):
    os.execv("/usr/bin/python3", ["python3", pyc_file])
else:
    raise FileNotFoundError(f"{pyc_file} not found")

