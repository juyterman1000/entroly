import json
import re

with open("bench/looGLE_compare.py", "r", encoding="utf-8") as f:
    code = f.read()

# Replace the Lingua code block with Agentic Pruning in looGLE_compare.py
code = re.sub(
    r'def _get_lingua_compressor.*?def compress_pruner.*?return out\["compressed_prompt"\]',
    '''def compress_agentic_pruner(context: str, question: str, budget: int, method: str) -> str:
    """Agentic Context Pruning (2026 SOTA): uses LLM to extract relevant context."""
    current_tokens = _tokens(context)
    if current_tokens <= budget:
        return context
        
    from openai import OpenAI
    client = OpenAI()
    prompt = f"Extract only the exact facts from the following text that are relevant to this question: {question}\\n\\n{context}"
    
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=budget,
            temperature=0.0
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        # Fallback to truncation on API error during compression
        return context[:budget * 4]''',
    code,
    flags=re.DOTALL
)

# Fix references in the loop
code = re.sub(r'compress_pruner\(context, question, args\.budget, method\)', 'compress_agentic_pruner(context, question, args.budget, method)', code)
code = re.sub(r'args\.pruner_method', '"agentic_pruning"', code)
code = re.sub(r'--pruner-method.*?help=.*?TokenPruner comparison"\)', '--skip-pruner", action="store_true", help="skip agentic pruner")', code, flags=re.DOTALL)

with open("bench/looGLE_compare.py", "w", encoding="utf-8") as f:
    f.write(code)


with open("bench/colab_run.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if "source" in cell and len(cell["source"]) > 0:
        if "Load Token Pruner" in cell["source"][0] or "Load LLMLingua" in cell["source"][0]:
            cell["source"] = ["## 7. Initialize Agentic Context Pruner (2026 SOTA Baseline)\n"]
        elif "from llmlingua import PromptCompressor" in cell["source"][1] if len(cell["source"]) > 1 else False:
            cell["source"] = [
                "def compress_agentic_pruner(context: str, question: str, budget: int) -> str:\n",
                "    \"\"\"2026 Agentic Pruning: Uses an LLM to extract relevant context.\"\"\"\n",
                "    current_tokens = tokens(context)\n",
                "    if current_tokens <= budget:\n",
                "        return context\n",
                "    from openai import OpenAI\n",
                "    client = OpenAI()\n",
                "    prompt = f\"Extract only the facts from the following text relevant to this question: {question}\\n\\n{context}\"\n",
                "    resp = client.chat.completions.create(\n",
                "        model='gpt-4o-mini',\n",
                "        messages=[{'role': 'user', 'content': prompt}],\n",
                "        max_tokens=budget,\n",
                "    )\n",
                "    return resp.choices[0].message.content.strip()\n"
            ]
        else:
            for i, line in enumerate(cell["source"]):
                line = re.sub(r"compress_pruner\(", "compress_agentic_pruner(", line)
                line = re.sub(r"PRUNER_METHOD", "\"agentic_pruning\"", line)
                cell["source"][i] = line

with open("bench/colab_run.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
