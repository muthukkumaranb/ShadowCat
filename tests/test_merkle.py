"""Merkle root over the audit chain: known values, inclusion proofs, and tamper detection via the checkpoint."""
import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import audit_chain as ac  # noqa: E402


def _h(i):
    return hashlib.sha256(f"entry-{i}".encode()).hexdigest()


def test_root_matches_rfc6962_construction():
    a, b, c = _h(0), _h(1), _h(2)
    leaf = lambda x: hashlib.sha256(b"\x00" + bytes.fromhex(x)).digest()
    node = lambda l, r: hashlib.sha256(b"\x01" + l + r).digest()
    assert ac.merkle_root([a]) == leaf(a).hex()
    assert ac.merkle_root([a, b]) == node(leaf(a), leaf(b)).hex()
    # unpaired last node is carried up unchanged
    assert ac.merkle_root([a, b, c]) == node(node(leaf(a), leaf(b)), leaf(c)).hex()


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 13])
def test_every_entry_has_a_valid_inclusion_proof(n):
    hashes = [_h(i) for i in range(n)]
    root = ac.merkle_root(hashes)
    for i, h in enumerate(hashes):
        proof = ac.merkle_proof(hashes, i)
        assert ac.verify_merkle_proof(h, proof, root)
        assert len(proof) <= max(1, (n - 1).bit_length())
        assert not ac.verify_merkle_proof(_h(999), proof, root)


def test_checkpoint_detects_a_rewritten_chain(tmp_path, monkeypatch):
    monkeypatch.delenv("SHADOWCAT_SIGNING_KEY", raising=False)
    chain_path = str(tmp_path / "audit_chain.json")
    for i in range(4):
        art = tmp_path / f"a{i}.txt"
        art.write_text(f"artifact {i}")
        ac.append_entry(str(art), "test", f"artifact {i}", chain_path=chain_path)
    cp = ac.write_merkle_checkpoint(chain_path)
    assert cp["n_entries"] == 4
    assert ac.verify_chain(chain_path)[0]

    # Appending keeps the checkpoint valid (it covers the first 4 entries).
    art = tmp_path / "a4.txt"; art.write_text("artifact 4")
    ac.append_entry(str(art), "test", "artifact 4", chain_path=chain_path)
    assert ac.verify_chain(chain_path)[0]

    # Rebuilding the chain with a different history, links recomputed, is caught by the root.
    chain = ac._load_chain(chain_path)
    chain[1]["description"] = "rewritten"
    import json
    prev = chain[0]["entry_hash"]
    for e in chain[1:]:
        e["prev_entry_hash"] = prev
        body = {k: v for k, v in e.items() if k not in ("entry_hash", "signature")}
        e["entry_hash"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        prev = e["entry_hash"]
    ac._save_chain(chain, chain_path)
    ok, issues = ac.verify_chain(chain_path)
    assert not ok and any("Merkle root mismatch" in i for i in issues)
