import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from typing import Optional

# Safe UTF-8 console output for Windows CLI environments
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

CHAIN_PATH = os.path.join(os.path.dirname(__file__), "audit_chain.json")


def _sha256_of_file(filepath: str) -> str:
    """
    Hash a file's actual content. For text files (.md, .json, .yaml, .yml, .txt, .csv, .py),
    normalize CRLF to LF so the hash is invariant across OS line-ending conventions
    and git checkout settings (core.autocrlf). For binary files, hash raw bytes directly.
    """
    ext = os.path.splitext(filepath)[1].lower()
    text_extensions = {".md", ".json", ".yaml", ".yml", ".txt", ".csv", ".py"}

    if ext in text_extensions:
        with open(filepath, "rb") as f:
            content = f.read()
        canonical = content.replace(b"\r\n", b"\n")
        return hashlib.sha256(canonical).hexdigest()
    else:
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()


def _resolve_path(stored_path: str) -> str:
    """
    Resolve an artifact path relative to current working dir, backend dir,
    or repository root.
    """
    if os.path.isabs(stored_path) and os.path.exists(stored_path):
        return stored_path
    if os.path.exists(stored_path):
        return stored_path
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    cand1 = os.path.normpath(os.path.join(backend_dir, stored_path))
    if os.path.exists(cand1):
        return cand1
    repo_root = os.path.normpath(os.path.join(backend_dir, ".."))
    cand2 = os.path.normpath(os.path.join(repo_root, stored_path))
    if os.path.exists(cand2):
        return cand2
    return cand1


def _load_chain(chain_path: Optional[str] = None) -> list:
    path = chain_path or CHAIN_PATH
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_chain(chain: list, chain_path: Optional[str] = None) -> None:
    path = chain_path or CHAIN_PATH
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chain, f, indent=2)


def append_entry(
    artifact_path: str,
    artifact_type: str,
    description: str,
    chain_path: Optional[str] = None,
) -> dict:
    """
    Add a new entry to the chain, hashing the ACTUAL current bytes of
    artifact_path and linking to the previous entry's hash.
    """
    chain = _load_chain(chain_path)
    prev_hash = chain[-1]["entry_hash"] if chain else "0" * 64

    resolved_path = _resolve_path(artifact_path)
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Artifact file not found: {artifact_path} (resolved: {resolved_path})")

    artifact_hash = _sha256_of_file(resolved_path)
    timestamp = datetime.now(timezone.utc).isoformat()

    # Normalize relative path representation with forward slashes
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    rel_path = os.path.relpath(resolved_path, start=backend_dir).replace("\\", "/")

    entry_content = {
        "index": len(chain),
        "timestamp": timestamp,
        "artifact_path": rel_path,
        "artifact_type": artifact_type,
        "description": description,
        "artifact_hash": artifact_hash,
        "prev_entry_hash": prev_hash,
    }
    # Hash the entry itself (including prev_hash) to produce the link
    entry_str = json.dumps(entry_content, sort_keys=True)
    entry_hash = hashlib.sha256(entry_str.encode()).hexdigest()
    entry_content["entry_hash"] = entry_hash

    chain.append(entry_content)
    _save_chain(chain, chain_path)
    return entry_content


def verify_chain(chain_path: Optional[str] = None) -> tuple:
    """
    Walk the entire chain and verify:
    1. Each entry's prev_entry_hash matches the actual previous entry's
       entry_hash (chain integrity).
    2. Each entry's entry_hash matches the SHA-256 of its content (block integrity).
    3. Each entry's artifact_hash still matches the CURRENT bytes of the
       artifact file on disk (tamper detection).
    Returns (is_valid: bool, issues: list[str]).
    """
    chain = _load_chain(chain_path)
    if not chain:
        return (False, ["Audit chain is empty or not found."])

    issues = []

    for i, entry in enumerate(chain):
        expected_prev = chain[i - 1]["entry_hash"] if i > 0 else "0" * 64
        if entry.get("prev_entry_hash") != expected_prev:
            issues.append(
                f"Entry {i}: chain link broken (prev_entry_hash mismatch)"
            )

        # Verify entry block integrity
        entry_copy = {k: v for k, v in entry.items() if k != "entry_hash"}
        entry_str = json.dumps(entry_copy, sort_keys=True)
        computed_entry_hash = hashlib.sha256(entry_str.encode()).hexdigest()
        if computed_entry_hash != entry.get("entry_hash"):
            issues.append(
                f"Entry {i}: entry hash corrupted/mismatched "
                f"(computed {computed_entry_hash[:12]}... vs recorded {str(entry.get('entry_hash'))[:12]}...)"
            )

        resolved = _resolve_path(entry.get("artifact_path", ""))
        if os.path.exists(resolved):
            current_hash = _sha256_of_file(resolved)
            if current_hash != entry.get("artifact_hash"):
                issues.append(
                    f"Entry {i} ({entry.get('artifact_path')}): "
                    f"artifact has been modified since chaining "
                    f"(hash mismatch — possible tampering)"
                )
        else:
            issues.append(
                f"Entry {i} ({entry.get('artifact_path')}): file no longer exists"
            )

    return (len(issues) == 0, issues)
