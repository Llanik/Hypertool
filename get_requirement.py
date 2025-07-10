import os
import re
from pathlib import Path

PROJECT_DIR = Path(".")  # Adapt si besoin
imports = set()

for pyfile in PROJECT_DIR.rglob("*.py"):
    with open(pyfile, encoding="utf-8") as f:
        for line in f:
            match = re.match(r'^\s*(?:from|import)\s+([a-zA-Z0-9_\.]+)', line)
            if match:
                top_level = match.group(1).split('.')[0]
                if not top_level.startswith('_'):
                    imports.add(top_level)

print("📦 Modules détectés dans le projet :")
for imp in sorted(imports):
    print(imp)
