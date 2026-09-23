import json
import os

filepath = r"data-engineering\data\mitre\enterprise-attack.json"
with open(filepath, "r", encoding="utf-8") as f:
    bundle = json.load(f)

print("Bundle type:", bundle.get("type"))
print("Bundle id:", bundle.get("id"))
print("Spec version:", bundle.get("spec_version"))
objects = bundle.get("objects", [])
print("Total objects:", len(objects))

type_counts = {}
for obj in objects:
    t = obj.get("type")
    type_counts[t] = type_counts.get(t, 0) + 1

for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
    print(f"  {t}: {c}")

# Find version / metadata
for obj in objects:
    if obj.get("type") == "x-mitre-collection":
        print("\nCollection object found:")
        print("  Name:", obj.get("name"))
        print("  Version:", obj.get("x_mitre_version"))
        print("  Modified:", obj.get("modified"))
        print("  Description:", (obj.get("description", "")[:120] + "..."))

# Check tactics
tactics = [obj for obj in objects if obj.get("type") == "x-mitre-tactic"]
print(f"\nTotal x-mitre-tactic objects: {len(tactics)}")
for tac in tactics:
    tac_id = None
    for ref in tac.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            tac_id = ref.get("external_id")
            break
    print(f"  {tac_id}: {tac.get('name')} (shortname: {tac.get('x_mitre_shortname')})")

# Check attack-patterns (techniques)
techniques = [obj for obj in objects if obj.get("type") == "attack-pattern"]
active_techniques = [t for t in techniques if not t.get("revoked", False) and not t.get("x_mitre_deprecated", False)]
print(f"\nTotal attack-patterns: {len(techniques)} (Active non-revoked: {len(active_techniques)})")

# Check the 6 techniques specifically
target_techs = ["T1110.001", "T1498.001", "T1071.001", "T1190", "T1189", "T1046"]
tech_map = {}
for tech in techniques:
    tech_id = None
    url = None
    for ref in tech.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            tech_id = ref.get("external_id")
            url = ref.get("url")
            break
    if tech_id:
        tech_map[tech_id] = {
            "name": tech.get("name"),
            "url": url,
            "kill_chain_phases": tech.get("kill_chain_phases", []),
            "revoked": tech.get("revoked", False),
            "deprecated": tech.get("x_mitre_deprecated", False),
            "is_subtechnique": tech.get("x_mitre_is_subtechnique", False),
            "description": tech.get("description", "")[:120] + "..."
        }

print("\nTarget 6 Techniques Verification:")
for t_id in target_techs:
    if t_id in tech_map:
        info = tech_map[t_id]
        print(f"  [FOUND] {t_id} -> {info['name']}")
        print(f"          URL: {info['url']}")
        print(f"          Subtechnique: {info['is_subtechnique']}")
        print(f"          Phases: {[p.get('phase_name') for p in info['kill_chain_phases']]}")
    else:
        print(f"  [MISSING] {t_id}")
