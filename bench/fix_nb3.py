import json
import re

with open("bench/colab_run.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if "source" in cell:
        for i, line in enumerate(cell["source"]):
            if '"agentic_pruning" = \'llmlingua2\'' in line:
                cell["source"][i] = "PRUNER_METHOD = 'agentic_pruning'    # 2026 SOTA Agentic baseline\n"
            elif 'lingua={"agentic_pruning"}' in line:
                cell["source"][i] = 'print(f"subset={SUBSET}  samples={SAMPLES}  budget={BUDGET}  answer={ANSWER_MODEL}  pruner={PRUNER_METHOD}")\n'

with open("bench/colab_run.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
