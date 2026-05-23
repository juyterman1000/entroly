import json

with open('bench/colab_run.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if 'source' in cell and len(cell['source']) > 0:
        source_text = ''.join(cell['source'])
        if 'def compress_agentic_pruner' in source_text and 'lingua.compress_prompt' in source_text:
            cell['source'] = [
                'def compress_agentic_pruner(context: str, question: str, budget: int, method: str) -> str:\n',
                '    """Agentic Context Pruning (2026 SOTA): uses LLM to extract relevant context."""\n',
                '    current_tokens = tokens(context)\n',
                '    if current_tokens <= budget:\n',
                '        return context\n',
                '        \n',
                '    from openai import OpenAI\n',
                '    client = OpenAI()\n',
                '    prompt = f"Extract only the exact facts from the following text that are relevant to this question: {question}\\n\\n{context}"\n',
                '    \n',
                '    try:\n',
                '        resp = client.chat.completions.create(\n',
                '            model="gpt-4o-mini",\n',
                '            messages=[{"role": "user", "content": prompt}],\n',
                '            max_tokens=budget,\n',
                '            temperature=0.0\n',
                '        )\n',
                '        return resp.choices[0].message.content.strip()\n',
                '    except Exception:\n',
                '        # Fallback to truncation on API error during compression\n',
                '        return context[:budget * 4]\n'
            ]

with open('bench/colab_run.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
