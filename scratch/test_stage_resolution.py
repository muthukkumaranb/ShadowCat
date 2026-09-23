import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.mitre_kb import get_mitre_kb

kb = get_mitre_kb()

stages = ["Reconnaissance", "Initial Access", "Credential Access", "Lateral Movement", "Impact", "Command and Control", "Discovery"]

for s in stages:
    res = kb.resolve_stage(s)
    print(f"\n--- Stage: {s} ---")
    print(f"Tactic: {res['tactic_id']} - {res['tactic_name']} ({res['url']})")
    print(f"Technique: {res['technique_id']} - {res['technique_full_name']} ({res['technique_url']})")
    print(f"Description: {res['technique_description'][:160]}...")
