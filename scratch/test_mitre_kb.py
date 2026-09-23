import ast
import sys
from pathlib import Path

# Add repo root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ast.parse(open("backend/mitre_kb.py", encoding="utf-8").read())
print("AST check: OK")

from backend.mitre_kb import get_mitre_kb
kb = get_mitre_kb()
meta = kb.get_corpus_metadata()
print("Corpus metadata:", meta)
print("Total tactics:", len(kb.tactics_by_id))
print("Total techniques:", len(kb.techniques_by_id))

# Verify the 6 technique IDs
targets = ["T1110.001", "T1498.001", "T1071.001", "T1190", "T1189", "T1046"]
print("\nTarget technique verification:")
for tid in targets:
    t = kb.get_technique(tid)
    if t:
        print(f"  {tid}: name='{t['name']}', full_name='{t['full_name']}', parent='{t['parent_name']}', url='{t['url']}'")
    else:
        print(f"  {tid}: NOT FOUND!")

# Test resolve_stage
for stage in ["Credential Access", "Lateral Movement", "Impact", "Unknown/Other"]:
    res = kb.resolve_stage(stage)
    print(f"\nStage '{stage}':")
    print(f"  Tactic: {res['tactic_id']} - {res['tactic_name']} ({res['url']})")
    print(f"  Technique: {res['technique_id']} - {res['technique_full_name']} ({res['technique_url']})")
    print(f"  Desc: {res['technique_description'][:90]}...")
