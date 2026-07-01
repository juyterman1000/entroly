import os
import ast

def extract_python(filepath):
    output = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    output.append("--- Snippet (First 10 lines) ---")
    output.append("\n".join(content.splitlines()[:10]))
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                doc = ast.get_docstring(node) or ""
                doc = doc.replace('\n', ' ')
                output.append(f"CLASS: {node.name} - {doc[:100]}")
            elif isinstance(node, ast.FunctionDef):
                doc = ast.get_docstring(node) or ""
                doc = doc.replace('\n', ' ')
                output.append(f"FUNC: {node.name} - {doc[:100]}")
    except Exception:
        pass
    return "\n".join(output)

def extract_rust(filepath):
    output = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    output.append("--- Snippet (First 10 lines) ---")
    output.append("".join(lines[:10]).strip())

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("pub struct") or stripped.startswith("struct"):
            output.append(f"STRUCT: {stripped.split('{')[0]}")
        elif stripped.startswith("pub fn") or stripped.startswith("fn"):
            output.append(f"FN: {stripped.split('{')[0]}")
        elif stripped.startswith("pub enum") or stripped.startswith("enum"):
            output.append(f"ENUM: {stripped.split('{')[0]}")
        elif stripped.startswith("impl"):
            output.append(f"IMPL: {stripped.split('{')[0]}")

    return "\n".join(output)

def extract_generic(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        return "".join(lines[:50])
    except:
        return ""

def main():
    dirs_to_scan = [
        r"C:\Users\abhis\entroly",
    ]
    summary = []
    processed_count = 0
    for base_dir in dirs_to_scan:
        if not os.path.exists(base_dir):
            continue
        for root, dirs, files in os.walk(base_dir):
            if any(skip in root for skip in [".git", "target", "__pycache__", ".venv", "node_modules"]):
                continue
            for file in files:
                path = os.path.join(root, file)
                summary.append("\n=======================")
                summary.append(f"FILE: {path}")
                if file.endswith(".py"):
                    summary.append(extract_python(path))
                elif file.endswith(".rs"):
                    summary.append(extract_rust(path))
                elif file.endswith(".toml") or file.endswith(".md"):
                    summary.append(extract_generic(path))
                else:
                    summary.append("Skipped full parse, generic file.")
                processed_count += 1

    with open(r"c:\Users\abhis\entroly\super_dump.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary))
    print(f"Processed {processed_count} files into super_dump.txt")

if __name__ == "__main__":
    main()
