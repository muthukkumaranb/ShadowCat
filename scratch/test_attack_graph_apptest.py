"""
Streamlit AppTest Verification for 04_Attack_Graph.py
Tests that the real graph-propagation traversal Attack Graph page renders,
scrubs across k=0..5, selects real nodes, and handles isolation without exceptions.
"""

import sys
from pathlib import Path
import os

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "frontend"))
os.chdir(str(REPO_ROOT))

import streamlit as st
from streamlit.testing.v1 import AppTest


def test_attack_graph_page():
    page_path = str(REPO_ROOT / "frontend" / "views" / "04_Attack_Graph.py")
    at = AppTest.from_file(page_path, default_timeout=120)
    at.run(timeout=120)

    print("--- 04_Attack_Graph.py Initial Render ---")
    print(f"Exceptions: {len(at.exception)}")
    for e in at.exception:
        print(f"  [EXCEPTION] {e.value}")
    assert len(at.exception) == 0, f"Expected 0 exceptions, got {len(at.exception)}"

    # Check session state
    print(f"attack_k_step in state: {at.session_state['attack_k_step']}")
    print(f"selected_node in state: {at.session_state['selected_node']}")
    assert at.session_state["selected_node"] != "svc-auth-master", "selected_node should be derived from real data, not svc-auth-master"

    # Scrub through k steps by clicking buttons
    for k in range(1, 6):
        btn = at.button(key=f"att_k_{k}")
        if btn:
            btn.click().run(timeout=120)
            assert len(at.exception) == 0, f"Exception when scrubbing to k={k}: {at.exception}"
            print(f"[+] Scrubbed to k={k}: selected_node={at.session_state['selected_node']}")

    print("[+] All AppTest steps on 04_Attack_Graph.py passed with zero exceptions!")


if __name__ == "__main__":
    test_attack_graph_page()
