#!/bin/bash

# Ellenőrzi, hogy a /app mappa létezik-e
if [ ! -d "/app" ]; then
    echo "Error: /app directory not found!"
    exit 1
fi

# Ellenőrzi, hogy vannak-e .py fájlok a /app könyvtárban
if ! find /app -name "*.py" -type f | grep -q "."; then
    echo "Error: No .py files found in /app!"
    exit 1
fi

# Fordítás a /app könyvtár összes .py fájljára
echo "Compiling Python files in /app..."
python3 -m compileall /app
if [ $? -ne 0 ]; then
    echo "Error: Compilation failed!"
    exit 1
fi

# Ellenőrzi, hogy létrejött-e a __pycache__ könyvtár
if [ ! -d "/app/__pycache__" ]; then
    echo "Error: __pycache__ directory not created!"
    exit 1
else
    echo "__pycache__ directory created successfully!"
fi

# Az eredeti .py fájlok törlése
echo "Removing original Python source files..."
find /app -name "*.py" -type f -delete

