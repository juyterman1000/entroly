import os
import ast
import json

def extract_python(filepath):
    output = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                docstring = ast.get_docstring(node) or ""
                output.append(f"Class: {node.name}\nDoc: {docstring.split(chr(10))[0] if docstring else 'None'}")
            elif isinstance(node, ast.FunctionDef):
                docstring = ast.get_docstring(node) or ""
                output.append(f"Function: {node.name}\nDoc: {docstring.split(chr(10))[0] if docstring else 'None'}")
    except Exception as e:
        output.append(f"Error parsing Python: {e}")
    return "\n".join(output)

def extract_rust(filepath):
    output = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    current_doc = []
    for line in lines:
        line = line.strip()
        if line.startswith("///") or line.startswith("//!"):
            current_doc.append(line.lstrip("/! "))
        elif line.startswith("pub struct") or line.startswith("struct"):
            output.append(f"Struct: {line.split('{')[0]}")
            if current_doc:
                output.append(f"Doc: {current_doc[0]}")
            current_doc = []
        elif line.startswith("pub fn") or line.startswith("fn"):
            output.append(f"Function: {line.split('{')[0]}")
            if current_doc:
                output.append(f"Doc: {current_doc[0]}")
            current_doc = []
        elif line.startswith("pub enum") or line.startswith("enum"):
            output.append(f"Enum: {line.split('{')[0]}")
            if current_doc:
                output.append(f"Doc: {current_doc[0]}")
            current_doc = []
        elif not line.startswith("#["):
            current_doc = []

    return "\n".join(output)

def main():
    base_dir = r"C:\Users\abhis\entroly"
    summary = []
    for root, dirs, files in os.walk(base_dir):
        if ".venv" in root or ".git" in root or "__pycache__" in root or "target" in root:
            continue
        for file in files:
            path = os.path.join(root, file)
            if file.endswith(".py"):
                summary.append(f"\n=== File: {path} ===")
                summary.append(extract_python(path))
            elif file.endswith(".rs"):
                summary.append(f"\n=== File: {path} ===")
                summary.append(extract_rust(path))

    with open("c:\\Users\\abhis\\entroly\\arch_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary))

if __name__ == "__main__":
    main()
