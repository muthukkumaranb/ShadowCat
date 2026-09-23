import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.mitre_kb import get_mitre_kb
kb = get_mitre_kb()

stages_config = [
    {
        "stage": "Reconnaissance",
        "status": "Historical",
        "confidence": 0.89,
    },
    {
        "stage": "Initial Access",
        "status": "Active (Current)",
        "confidence": 0.78,
    },
    {
        "stage": "Credential Access",
        "status": "Forecast (t+1)",
        "confidence": 0.67,
    },
    {
        "stage": "Lateral Movement",
        "status": "Forecast (t+2)",
        "confidence": 0.54,
    },
    {
        "stage": "Impact",
        "status": "Forecast (t+4)",
        "confidence": 0.38,
    },
]

output = []
for sc in stages_config:
    res = kb.resolve_stage(sc["stage"])
    entry = {
        "stage": sc["stage"],
        "id": res["technique_id"],
        "tactic_id": res["tactic_id"],
        "tactic_name": res["tactic_name"],
        "tactic_url": res["url"],
        "technique_id": res["technique_id"],
        "technique_name": res["technique_name"],
        "technique_full_name": res["technique_full_name"],
        "technique_url": res["technique_url"],
        "description": res["technique_description"],
        "confidence": sc["confidence"],
        "status": sc["status"],
        "is_mock": False,
    }
    output.append(entry)

out_path = Path("frontend/models/mitre_mapper.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print("Generated frontend/models/mitre_mapper.json successfully!")
for item in output:
    print(f"  {item['stage']}: {item['technique_id']} ({item['technique_name']}) - {item['technique_url']}")
