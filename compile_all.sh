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

# Kommentek és docstring-ek eltávolítása Python script segítségével
echo "Removing comments and docstrings from Python files..."
python3 - <<EOF
import ast
import os

def remove_comments_and_docstrings(source):
    """
    Removes comments and docstrings from Python source code.
    """
    class CommentAndDocstringRemover(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            # Remove docstrings from functions
            if (len(node.body) > 0 and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str)):
                node.body.pop(0)
            self.generic_visit(node)
            return node

        def visit_ClassDef(self, node):
            # Remove docstrings from classes
            if (len(node.body) > 0 and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str)):
                node.body.pop(0)
            self.generic_visit(node)
            return node

        def visit_Module(self, node):
            # Remove docstrings from modules
            if (len(node.body) > 0 and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str)):
                node.body.pop(0)
            self.generic_visit(node)
            return node

    tree = ast.parse(source)
    remover = CommentAndDocstringRemover()
    cleaned_tree = remover.visit(tree)
    return ast.unparse(cleaned_tree)

directory = "/app"
for root, _, files in os.walk(directory):
    for file in files:
        if file.endswith(".py"):
            file_path = os.path.join(root, file)
            with open(file_path, "r") as f:
                source = f.read()
            try:
                cleaned_source = remove_comments_and_docstrings(source)
                with open(file_path, "w") as f:
                    f.write(cleaned_source)
                print(f"Processed: {file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
EOF

# Ellenőrzi, hogy a pyarmor telepítve van-e
if ! command -v pyarmor &> /dev/null; then
    echo "Error: pyarmor is not installed! Please install it using 'pip install pyarmor'."
    exit 1
fi

# Obfuszkáció a .py fájlokra
echo "Obfuscating Python files in /app..."
find /app -name "*.py" -type f | while read -r file; do
    pyarmor obfuscate "$file"
    if [ $? -ne 0 ]; then
        echo "Error: Obfuscation failed for file $file!"
        exit 1
    fi
    # Az obfuszkált fájl az eredeti helyén marad, töröljük az eredetit
    mv "${file%.py}.py" "$file"
    echo "File obfuscated: $file"
done

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

echo "Obfuscation and compilation completed successfully!"

