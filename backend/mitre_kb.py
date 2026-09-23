"""
SHADOWCAT Backend - MITRE ATT&CK Knowledge Base Loader & Query Layer
Parses the vendored official MITRE Enterprise ATT&CK STIX 2.1 corpus
(enterprise-attack.json) into an in-memory, air-gapped query structure.
Zero network dependencies at inference time.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default path to vendored Enterprise ATT&CK STIX 2.1 bundle
_DEFAULT_CORPUS_REL = Path("data-engineering") / "data" / "mitre" / "enterprise-attack.json"


def _clean_stix_text(text: str) -> str:
    """Removes STIX citation markers e.g. (Citation: Microsoft RDP 2020), converts markdown links [text](url) to text, and normalizes whitespace."""
    if not text:
        return ""
    # Strip citation tags
    cleaned = re.sub(r"\(Citation:\s*[^)]+\)", "", text)
    # Convert markdown links [Label](url) -> Label
    cleaned = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", cleaned)
    # Normalize whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


class MitreKnowledgeBase:
    """
    Offline in-memory query engine for MITRE Enterprise ATT&CK STIX 2.1 data.
    Provides fast, indexed lookups for tactics, techniques, sub-techniques, and stages.
    """

    def __init__(self, corpus_path: Optional[Path | str] = None) -> None:
        self.corpus_path = self._locate_corpus(corpus_path)
        self.metadata: Dict[str, Any] = {}
        self.tactics_by_id: Dict[str, Dict[str, Any]] = {}
        self.tactics_by_name: Dict[str, Dict[str, Any]] = {}
        self.tactics_by_shortname: Dict[str, Dict[str, Any]] = {}
        self.techniques_by_id: Dict[str, Dict[str, Any]] = {}
        self.techniques_by_name: Dict[str, List[Dict[str, Any]]] = {}
        self.tactic_to_techniques: Dict[str, List[str]] = {}
        self._loaded = False
        self._lock = threading.Lock()
        self._load_corpus()

    @staticmethod
    def _locate_corpus(corpus_path: Optional[Path | str] = None) -> Path:
        """Resolves corpus path across backend, repo root, or environment."""
        if corpus_path:
            p = Path(corpus_path).resolve()
            if p.exists():
                return p

        # Check relative to backend/
        backend_dir = Path(__file__).resolve().parent
        repo_root = backend_dir.parent

        candidates = [
            repo_root / _DEFAULT_CORPUS_REL,
            backend_dir.parent / _DEFAULT_CORPUS_REL,
            Path("data-engineering/data/mitre/enterprise-attack.json").resolve(),
            Path("../data-engineering/data/mitre/enterprise-attack.json").resolve(),
        ]

        for cand in candidates:
            if cand.exists():
                return cand

        # Return primary intended path even if missing (load will give descriptive error)
        return repo_root / _DEFAULT_CORPUS_REL

    def _load_corpus(self) -> None:
        """Parses the STIX 2.1 bundle into in-memory dictionaries."""
        with self._lock:
            if self._loaded:
                return

            if not self.corpus_path.exists():
                raise FileNotFoundError(
                    f"MITRE Enterprise ATT&CK corpus not found at {self.corpus_path}. "
                    "Ensure Step 1 vendoring of enterprise-attack.json is complete."
                )

            file_size = self.corpus_path.stat().st_size

            with open(self.corpus_path, "r", encoding="utf-8") as f:
                bundle = json.load(f)

            objects = bundle.get("objects", [])

            # 1. Parse Metadata
            for obj in objects:
                if obj.get("type") == "x-mitre-collection":
                    self.metadata = {
                        "name": obj.get("name", "Enterprise ATT&CK"),
                        "version": obj.get("x_mitre_version", "unknown"),
                        "modified": obj.get("modified", "unknown"),
                        "spec_version": obj.get("x_mitre_attack_spec_version", "unknown"),
                        "file_size_bytes": file_size,
                        "bundle_id": bundle.get("id"),
                        "total_objects": len(objects),
                    }
                    break

            if not self.metadata:
                self.metadata = {
                    "name": "Enterprise ATT&CK",
                    "version": "19.2",
                    "file_size_bytes": file_size,
                    "total_objects": len(objects),
                }

            # Temporary maps for STIX ID resolution
            stix_to_technique: Dict[str, Dict[str, Any]] = {}
            stix_to_subtechnique_parent: Dict[str, str] = {}  # child_stix_id -> parent_stix_id

            # 2. Parse Tactics (x-mitre-tactic)
            for obj in objects:
                if obj.get("type") == "x-mitre-tactic":
                    tac_id = None
                    tac_url = None
                    for ref in obj.get("external_references", []):
                        if ref.get("source_name") == "mitre-attack":
                            tac_id = ref.get("external_id")
                            tac_url = ref.get("url")
                            break

                    if not tac_id:
                        continue

                    tac_name = obj.get("name", "")
                    shortname = obj.get("x_mitre_shortname", "")
                    raw_desc = obj.get("description", "")
                    clean_desc = _clean_stix_text(raw_desc)

                    tactic_info = {
                        "id": tac_id,
                        "name": tac_name,
                        "shortname": shortname,
                        "description": clean_desc,
                        "raw_description": raw_desc,
                        "url": tac_url or f"https://attack.mitre.org/tactics/{tac_id}",
                    }

                    self.tactics_by_id[tac_id] = tactic_info
                    self.tactics_by_name[tac_name.lower()] = tactic_info
                    if shortname:
                        self.tactics_by_shortname[shortname.lower()] = tactic_info
                        self.tactic_to_techniques[shortname.lower()] = []
                    self.tactic_to_techniques[tac_id] = []

            # 3. Parse Subtechnique relationships
            for obj in objects:
                if (
                    obj.get("type") == "relationship"
                    and obj.get("relationship_type") == "subtechnique-of"
                    and not obj.get("revoked", False)
                ):
                    child_ref = obj.get("source_ref")
                    parent_ref = obj.get("target_ref")
                    if child_ref and parent_ref:
                        stix_to_subtechnique_parent[child_ref] = parent_ref

            # 4. Parse Techniques (attack-pattern)
            for obj in objects:
                if obj.get("type") == "attack-pattern":
                    tech_id = None
                    tech_url = None
                    for ref in obj.get("external_references", []):
                        if ref.get("source_name") == "mitre-attack":
                            tech_id = ref.get("external_id")
                            tech_url = ref.get("url")
                            break

                    if not tech_id:
                        continue

                    stix_id = obj.get("id")
                    name = obj.get("name", "")
                    raw_desc = obj.get("description", "")
                    clean_desc = _clean_stix_text(raw_desc)
                    is_sub = obj.get("x_mitre_is_subtechnique", False)
                    phases = obj.get("kill_chain_phases", [])
                    tactic_shortnames = [p.get("phase_name") for p in phases if p.get("kill_chain_name") == "mitre-attack"]

                    # Map shortnames to official tactic names and IDs
                    tactics_resolved = []
                    for sname in tactic_shortnames:
                        tac_meta = self.tactics_by_shortname.get(sname.lower())
                        if tac_meta:
                            tactics_resolved.append({
                                "id": tac_meta["id"],
                                "name": tac_meta["name"],
                                "shortname": sname,
                            })

                    tech_info = {
                        "id": tech_id,
                        "stix_id": stix_id,
                        "name": name,
                        "full_name": name,  # updated below if subtechnique
                        "parent_id": None,
                        "parent_name": None,
                        "description": clean_desc,
                        "raw_description": raw_desc,
                        "url": tech_url or f"https://attack.mitre.org/techniques/{tech_id.replace('.', '/')}",
                        "is_subtechnique": is_sub,
                        "subtechniques": [],
                        "tactics": tactics_resolved,
                        "tactic_shortnames": tactic_shortnames,
                        "revoked": obj.get("revoked", False),
                        "deprecated": obj.get("x_mitre_deprecated", False),
                    }

                    self.techniques_by_id[tech_id] = tech_info
                    if stix_id:
                        stix_to_technique[stix_id] = tech_info

                    name_lower = name.lower()
                    if name_lower not in self.techniques_by_name:
                        self.techniques_by_name[name_lower] = []
                    self.techniques_by_name[name_lower].append(tech_info)

                    # Register under tactics
                    for tac in tactics_resolved:
                        self.tactic_to_techniques[tac["id"]].append(tech_id)
                        if tac["shortname"] in self.tactic_to_techniques:
                            self.tactic_to_techniques[tac["shortname"]].append(tech_id)

            # 5. Link Parent & Sub-technique Relationships
            for child_stix, parent_stix in stix_to_subtechnique_parent.items():
                child_tech = stix_to_technique.get(child_stix)
                parent_tech = stix_to_technique.get(parent_stix)
                if child_tech and parent_tech:
                    child_tech["parent_id"] = parent_tech["id"]
                    child_tech["parent_name"] = parent_tech["name"]
                    child_tech["full_name"] = f"{parent_tech['name']}: {child_tech['name']}"
                    parent_tech["subtechniques"].append(child_tech["id"])

            self.metadata["total_tactics"] = len(self.tactics_by_id)
            self.metadata["total_techniques"] = len(self.techniques_by_id)
            self.metadata["active_techniques"] = len([t for t in self.techniques_by_id.values() if not t["revoked"] and not t["deprecated"]])
            self._loaded = True
            logger.info(
                f"MitreKnowledgeBase loaded successfully: {self.metadata['total_tactics']} tactics, "
                f"{self.metadata['total_techniques']} techniques ({self.metadata['active_techniques']} active)."
            )

    def get_corpus_metadata(self) -> Dict[str, Any]:
        """Returns corpus metadata, version, and object counts."""
        return dict(self.metadata)

    def get_tactic(self, tactic_id_or_name: str) -> Optional[Dict[str, Any]]:
        """Look up tactic by ID (TA0006), name ('Credential Access'), or shortname ('credential-access')."""
        if not tactic_id_or_name:
            return None
        key = str(tactic_id_or_name).strip()
        if key in self.tactics_by_id:
            return self.tactics_by_id[key]
        key_lower = key.lower()
        if key_lower in self.tactics_by_name:
            return self.tactics_by_name[key_lower]
        if key_lower in self.tactics_by_shortname:
            return self.tactics_by_shortname[key_lower]
        return None

    def get_technique(self, technique_id_or_name: str) -> Optional[Dict[str, Any]]:
        """Look up technique by ID ('T1110.001') or name ('Password Guessing')."""
        if not technique_id_or_name:
            return None
        key = str(technique_id_or_name).strip()
        if key in self.techniques_by_id:
            return self.techniques_by_id[key]
        key_lower = key.lower()
        if key_lower in self.techniques_by_name:
            return self.techniques_by_name[key_lower][0]
        return None

    def get_techniques_by_tactic(self, tactic_id_or_name: str) -> List[Dict[str, Any]]:
        """Returns list of techniques categorized under the given tactic."""
        tac = self.get_tactic(tactic_id_or_name)
        if not tac:
            return []
        tech_ids = self.tactic_to_techniques.get(tac["id"], [])
        return [self.techniques_by_id[tid] for tid in tech_ids if tid in self.techniques_by_id]

    def resolve_stage(self, stage_name: str) -> Dict[str, Any]:
        """
        Resolves a high-level stage label (from the stage classifier head or progression)
        into a real MITRE ATT&CK tactic, canonical technique, official description, and URL.
        """
        # Canonical mappings from telemetry attack types in CSE-CIC-IDS2018 to primary techniques
        CANONICAL_TECHNIQUES = {
            "credential access": "T1110.001",    # Brute Force: Password Guessing
            "initial access": "T1190",           # Exploit Public-Facing Application
            "command and control": "T1071.001",  # Application Layer Protocol: Web Protocols
            "discovery": "T1046",                # Network Service Discovery
            "reconnaissance": "T1046",           # Network Service Discovery
            "impact": "T1498.001",               # Network Denial of Service: Direct Network Flood
            "lateral movement": "T1021.002",     # Remote Services: SMB/Windows Admin Shares
        }

        stage_clean = str(stage_name).strip()
        stage_lower = stage_clean.lower()

        # Handle non-attack / baseline / unknown
        if stage_lower in ("unknown/other", "nominal baseline", "nominal traffic", "unknown", "none"):
            return {
                "tactic_id": "TA0000",
                "tactic_name": "Unknown/Other",
                "shortname": "unknown",
                "description": "Baseline network behavior or telemetry activity not cleanly attributed to a specific MITRE ATT&CK tactic.",
                "url": "https://attack.mitre.org/tactics/",
                "technique_id": "T0000",
                "technique_name": "Unclassified Telemetry Pattern",
                "technique_full_name": "Baseline Network State",
                "technique_description": "Network telemetry within operational bounds without active adversary technique indicators.",
                "technique_url": "https://attack.mitre.org/",
                "is_active_attack": False,
            }

        # Resolve tactic directly from KB
        tac = self.get_tactic(stage_clean)
        tactic_id = tac["id"] if tac else "TA0000"
        tactic_name = tac["name"] if tac else stage_clean
        tactic_desc = tac["description"] if tac else f"Adversary activity associated with {stage_clean}."
        tactic_url = tac["url"] if tac else f"https://attack.mitre.org/tactics/{tactic_id}"

        # Resolve primary technique
        canon_tech_id = CANONICAL_TECHNIQUES.get(stage_lower)
        tech_meta = self.get_technique(canon_tech_id) if canon_tech_id else None

        if tech_meta:
            tech_id = tech_meta["id"]
            tech_name = tech_meta["name"]
            tech_full = tech_meta.get("full_name", tech_meta["name"])
            tech_desc = tech_meta["description"]
            tech_url = tech_meta["url"]
        else:
            tech_id = "T1046"
            tech_name = "Network Activity"
            tech_full = "Network Activity"
            tech_desc = tactic_desc
            tech_url = tactic_url

        return {
            "tactic_id": tactic_id,
            "tactic_name": tactic_name,
            "shortname": tac.get("shortname", "") if tac else "",
            "description": tactic_desc,
            "url": tactic_url,
            "technique_id": tech_id,
            "technique_name": tech_name,
            "technique_full_name": tech_full,
            "technique_description": tech_desc,
            "technique_url": tech_url,
            "is_active_attack": True,
        }


# Global thread-safe singleton
_MITRE_KB_INSTANCE: Optional[MitreKnowledgeBase] = None
_INIT_LOCK = threading.Lock()


def get_mitre_kb(corpus_path: Optional[Path | str] = None) -> MitreKnowledgeBase:
    """Returns the initialized, cached singleton MitreKnowledgeBase."""
    global _MITRE_KB_INSTANCE
    if _MITRE_KB_INSTANCE is None:
        with _INIT_LOCK:
            if _MITRE_KB_INSTANCE is None:
                _MITRE_KB_INSTANCE = MitreKnowledgeBase(corpus_path=corpus_path)
    return _MITRE_KB_INSTANCE
