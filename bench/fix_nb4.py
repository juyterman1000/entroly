import json
import re

with open("bench/looGLE_compare.py", "r", encoding="utf-8") as f:
    code = f.read()

# Extract run_benchmark and print_report from looGLE_compare.py
run_benchmark_code = re.search(r'def run_benchmark\(args\) -> dict:.*?(?=\n# ══════════════════════════════════════════════════════════════════════\n# Reporting)', code, re.DOTALL).group(0)
print_report_code = re.search(r'def print_report\(report: dict\) -> None:.*?(?=\n# ══════════════════════════════════════════════════════════════════════\n# Main)', code, re.DOTALL).group(0)

with open("bench/colab_run.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if "source" in cell and len(cell["source"]) > 0:
        source_text = "".join(cell["source"])
        if "def run_benchmark" in source_text:
            cell["source"] = [line + "\n" for line in run_benchmark_code.split("\n")[:-1]]
        elif "def print_report" in source_text:
            cell["source"] = [line + "\n" for line in print_report_code.split("\n")[:-1]]

with open("bench/colab_run.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
