import sys
import os

# Python verzió meghatározása
py_version = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
pyc_file = f"/app/__pycache__/ollama_streaming.{py_version}.pyc"

# Ellenőrizzük, hogy létezik-e a megfelelő .pyc fájl
if os.path.exists(pyc_file):
    os.execv("/usr/bin/python3", ["python3", pyc_file])
else:
    raise FileNotFoundError(f"{pyc_file} not found")

