import json
import re

with open("bench/colab_run.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if "source" in cell:
        for i, line in enumerate(cell["source"]):
            line = re.sub(r"from token_pruner import PromptCompressor", "from llmlingua import PromptCompressor", line)
            line = re.sub(r"microsoft/token_pruner-2", "microsoft/llmlingua-2", line)
            line = re.sub(r"use_token_pruner2=True", "use_llmlingua2=True", line)
            line = re.sub(r"rank_method=\"longtoken_pruner\"", "rank_method=\"longllmlingua\"", line)
            line = re.sub(r"rank_method=\'longtoken_pruner\'", "rank_method=\'longllmlingua\'", line)
            cell["source"][i] = line

with open("bench/colab_run.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
