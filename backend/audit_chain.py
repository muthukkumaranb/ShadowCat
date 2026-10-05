import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from typing import Optional
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

# Safe UTF-8 console output for Windows CLI environments
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Runtime chain lives outside tracked sources: <repo>/runtime/ (gitignored),
# overridable with SHADOWCAT_RUNTIME_DIR.
RUNTIME_DIR = os.environ.get(
    "SHADOWCAT_RUNTIME_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runtime")
)
CHAIN_PATH = os.path.join(RUNTIME_DIR, "audit_chain.json")


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
    runtime_dir = os.environ.get("SHADOWCAT_RUNTIME_DIR")
    if runtime_dir:
        cand3 = os.path.normpath(os.path.join(runtime_dir, stored_path))
        if os.path.exists(cand3):
            return cand3
    return cand1


def _get_chain_path(chain_path: Optional[str] = None) -> str:
    if chain_path:
        return chain_path
    runtime_dir = os.environ.get(
        "SHADOWCAT_RUNTIME_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runtime")
    )
    return os.path.join(runtime_dir, "audit_chain.json")


def _load_chain(chain_path: Optional[str] = None) -> list:
    path = _get_chain_path(chain_path)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_chain(chain: list, chain_path: Optional[str] = None) -> None:
    path = _get_chain_path(chain_path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
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
    try:
        rel_path = os.path.relpath(resolved_path, start=backend_dir).replace("\\", "/")
    except ValueError:  # runtime dir on another drive (Windows): keep the absolute path
        rel_path = os.path.abspath(resolved_path).replace("\\", "/")

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
    
    priv_key_path = os.environ.get("SHADOWCAT_SIGNING_KEY")
    if priv_key_path and os.path.exists(priv_key_path):
        with open(priv_key_path, "rb") as f:
            key_data = f.read()
        try:
            priv_key = serialization.load_pem_private_key(key_data, password=None)
            sig = priv_key.sign(entry_hash.encode("utf-8"))
            entry_content["signature"] = sig.hex()
        except Exception as e:
            pass

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
        entry_copy = {k: v for k, v in entry.items() if k not in ("entry_hash", "signature")}
        entry_str = json.dumps(entry_copy, sort_keys=True)
        computed_entry_hash = hashlib.sha256(entry_str.encode()).hexdigest()
        if computed_entry_hash != entry.get("entry_hash"):
            issues.append(
                f"Entry {i}: entry hash corrupted/mismatched "
                f"(computed {computed_entry_hash[:12]}... vs recorded {str(entry.get('entry_hash'))[:12]}...)"
            )
            
        if "signature" in entry:
            pub_key_path = os.path.join(os.path.dirname(__file__), "admin_public_key.pem")
            if not os.path.exists(pub_key_path):
                issues.append(f"Entry {i}: signature present but public key not found")
            else:
                with open(pub_key_path, "rb") as f:
                    pub_key_bytes = f.read()
                try:
                    pub_key = serialization.load_pem_public_key(pub_key_bytes)
                    sig = bytes.fromhex(entry["signature"])
                    pub_key.verify(sig, entry["entry_hash"].encode("utf-8"))
                except Exception:
                    issues.append(f"Entry {i}: invalid signature")

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

    issues.extend(_verify_merkle_checkpoint(chain, chain_path))
    return (len(issues) == 0, issues)


# ----------------------------------------------------------------------------- Merkle root
# The chain links every entry to the previous one. On top of it, a Merkle tree over the entry hashes gives one
# 32-byte root that commits to the whole chain, and an O(log n) inclusion proof for any single entry (for example
# one forecast), checkable without the rest of the chain. Hashing follows RFC 6962 (Certificate Transparency):
#   leaf = SHA-256(0x00 || entry_hash), node = SHA-256(0x01 || left || right),
# and an unpaired last node is carried up unchanged, so a leaf can never be passed off as an internal node.

def _leaf(entry_hash_hex: str) -> bytes:
    return hashlib.sha256(b"\x00" + bytes.fromhex(entry_hash_hex)).digest()


def _node(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(b"\x01" + left + right).digest()


def merkle_root(entry_hashes: list) -> str:
    """Merkle root (hex) over a list of entry hashes (hex). Empty list -> SHA-256 of the empty string."""
    if not entry_hashes:
        return hashlib.sha256(b"").hexdigest()
    level = [_leaf(h) for h in entry_hashes]
    while len(level) > 1:
        nxt = [_node(level[i], level[i + 1]) for i in range(0, len(level) - 1, 2)]
        if len(level) % 2:
            nxt.append(level[-1])
        level = nxt
    return level[0].hex()


def merkle_proof(entry_hashes: list, index: int) -> list:
    """Inclusion proof for entry `index`: list of {"sibling": hex, "side": "left"|"right"} from leaf to root."""
    if not 0 <= index < len(entry_hashes):
        raise IndexError(f"index {index} outside chain of {len(entry_hashes)} entries")
    level = [_leaf(h) for h in entry_hashes]
    proof, i = [], index
    while len(level) > 1:
        sib = i ^ 1
        if sib < len(level):
            proof.append({"sibling": level[sib].hex(), "side": "left" if sib < i else "right"})
        nxt = [_node(level[j], level[j + 1]) for j in range(0, len(level) - 1, 2)]
        if len(level) % 2:
            nxt.append(level[-1])
        level, i = nxt, i // 2
    return proof


def verify_merkle_proof(entry_hash: str, proof: list, root: str) -> bool:
    """True if `entry_hash` is included under `root` according to `proof`."""
    h = _leaf(entry_hash)
    for step in proof:
        sib = bytes.fromhex(step["sibling"])
        h = _node(sib, h) if step["side"] == "left" else _node(h, sib)
    return h.hex() == root


def _checkpoint_path(chain_path: Optional[str] = None) -> str:
    return os.path.splitext(_get_chain_path(chain_path))[0] + ".merkle.json"


def chain_merkle_root(chain_path: Optional[str] = None) -> dict:
    """Current Merkle root over every entry of the chain."""
    chain = _load_chain(chain_path)
    return {"merkle_root": merkle_root([e["entry_hash"] for e in chain]), "n_entries": len(chain)}


def write_merkle_checkpoint(chain_path: Optional[str] = None) -> dict:
    """Record the chain's current Merkle root in <chain>.merkle.json, Ed25519-signed when a signing key is set."""
    cp = chain_merkle_root(chain_path)
    cp["timestamp"] = datetime.now(timezone.utc).isoformat()
    priv_key_path = os.environ.get("SHADOWCAT_SIGNING_KEY")
    if priv_key_path and os.path.exists(priv_key_path):
        with open(priv_key_path, "rb") as f:
            priv_key = serialization.load_pem_private_key(f.read(), password=None)
        cp["signature"] = priv_key.sign(cp["merkle_root"].encode("utf-8")).hex()
    path = _checkpoint_path(chain_path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cp, f, indent=2)
    return cp


def _verify_merkle_checkpoint(chain: list, chain_path: Optional[str] = None) -> list:
    """If a checkpoint exists, the root over its first n entries must still match (and its signature verify)."""
    path = _checkpoint_path(chain_path)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        cp = json.load(f)
    n = int(cp.get("n_entries", 0))
    if n > len(chain):
        return [f"Merkle checkpoint covers {n} entries but the chain has only {len(chain)} (entries removed)"]
    issues = []
    if merkle_root([e["entry_hash"] for e in chain[:n]]) != cp.get("merkle_root"):
        issues.append(f"Merkle root mismatch over the first {n} entries (chain rewritten since the checkpoint)")
    if "signature" in cp:
        pub_key_path = os.path.join(os.path.dirname(__file__), "admin_public_key.pem")
        try:
            with open(pub_key_path, "rb") as f:
                pub_key = serialization.load_pem_public_key(f.read())
            pub_key.verify(bytes.fromhex(cp["signature"]), cp["merkle_root"].encode("utf-8"))
        except Exception:
            issues.append("Merkle checkpoint: invalid signature")
    return issues
