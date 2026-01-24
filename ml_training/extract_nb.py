import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

nb = json.load(open('trocr-fine-tuning.ipynb', encoding='utf-8'))
for i, c in enumerate(nb['cells']):
    cell_type = c['cell_type']
    source = ''.join(c['source']) if isinstance(c['source'], list) else c['source']
    print(f"=== Cell {i} ({cell_type}) ===")
    print(source)
    print()
