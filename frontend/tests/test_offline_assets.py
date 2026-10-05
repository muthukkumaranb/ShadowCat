"""The dashboard's attack graph must render without internet: Cytoscape.js ships with the repo and is inlined."""
import hashlib
import sys
from pathlib import Path

FRONTEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FRONTEND))

from components import cytoscape_attack_graph as cag  # noqa: E402

VENDORED_SHA256 = "92d752b48ea949720675865197fd2a0001c95bc5888545e990af60321712d4c6"  # cytoscape 3.28.1 dist (npm)


def test_cytoscape_is_vendored_and_inlined():
    js = FRONTEND / "static" / "vendor" / "cytoscape-3.28.1.min.js"
    assert hashlib.sha256(js.read_bytes()).hexdigest() == VENDORED_SHA256
    assert (FRONTEND / "static" / "vendor" / "cytoscape-LICENSE.txt").exists()
    tag = cag._cytoscape_script_tag()
    assert tag.startswith("<script>") and "cdnjs" not in tag and len(tag) > 300_000
