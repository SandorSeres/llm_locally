#!/bin/bash

# Ellenőrzi, hogy a /app mappa létezik-e
if [ ! -d "/app" ]; then
    echo "Error: /app directory not found!"
    exit 1
fi

# Fordítás a /app könyvtár összes .py fájljára
echo "Compiling Python files in ./"
python3 -m compileall ./

# Ellenőrzi, hogy van-e hiba a fordítás során
if [ $? -ne 0 ]; then
    echo "Error: Compilation failed!"
    exit 1
else
    echo "Compilation completed successfully!"
fi

