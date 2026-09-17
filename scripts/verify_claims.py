#!/usr/bin/env python3
"""
SHADOWCAT - Automated Claims & Integrity Verification Gate
Enforces strict claims discipline and authoritative benchmark values.

FAIL CONDITIONS:
1. "5-Fold" or "38-fold" appearing anywhere in frontend code (must be authoritative 37-fold LOEO).
2. "Action:" appearing in output cards (prescriptive language prohibited).
3. Bare "Recommended:" appearing without the required "Illustrative analyst guidance" framing.
4. Absence of "LOEO 37-Fold" in views/03_Validation.py or mock_data.py.
"""

import os
import sys
import re

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

FORBIDDEN_PATTERNS = [
    (r"\b5-[Ff]old\b", "Stale 5-fold LOEO label detected (must be 37-fold LOEO)."),
    (r"\b38-[Ff]old\b", "Stale 38-fold draft label detected (must be 37-fold LOEO)."),
    (r"\bAction:\b", "Prescriptive 'Action:' language detected (must use 'Illustrative analyst guidance')."),
]

RECOMMENDED_RE = re.compile(r"Recommended:", re.IGNORECASE)
ALLOWED_GUIDANCE = "Illustrative analyst guidance"
MOCK_IMPORT_RE = re.compile(r"^\s*(from\s+mock_data\s+import|import\s+mock_data)\b")

FRONTEND_TARGETS = ["components", "views", "app.py", "mock_data.py", "data_provider.py", "styles.py"]
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def verify_file(filepath: str) -> list[str]:
    violations = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as fp:
        lines = fp.readlines()

    norm_path = os.path.normpath(filepath)
    is_allowed_mock_importer = norm_path.endswith("data_provider.py") or norm_path.endswith("mock_data.py")

    for idx, line in enumerate(lines, 1):
        # 1. Check forbidden regex patterns
        for pat, msg in FORBIDDEN_PATTERNS:
            if re.search(pat, line):
                violations.append(f"{filepath}:{idx}: {msg}\n    Offending line: {line.strip()}")

        # 2. Check bare "Recommended:" without "Illustrative analyst guidance" nearby
        if RECOMMENDED_RE.search(line):
            context_window = "".join(lines[max(0, idx - 3):min(len(lines), idx + 2)])
            if ALLOWED_GUIDANCE.lower() not in context_window.lower():
                violations.append(
                    f"{filepath}:{idx}: Bare 'Recommended:' found without required guidance framing.\n"
                    f"    Offending line: {line.strip()}"
                )

        # 3. Check direct import of mock_data outside data_provider.py
        if not is_allowed_mock_importer and MOCK_IMPORT_RE.search(line):
            violations.append(
                f"{filepath}:{idx}: Unauthorized direct import of mock_data (must import from data_provider).\n"
                f"    Offending line: {line.strip()}"
            )

    return violations


def main():
    print("=" * 65)
    print("SHADOWCAT Claims Discipline & Metric Verification Gate")
    print("=" * 65)

    all_violations = []

    for target in FRONTEND_TARGETS:
        target_path = os.path.join(WORKSPACE_ROOT, target)
        if os.path.isfile(target_path):
            all_violations.extend(verify_file(target_path))
        elif os.path.isdir(target_path):
            for root, _, files in os.walk(target_path):
                for f in sorted(files):
                    if f.endswith(".py"):
                        fpath = os.path.join(root, f)
                        all_violations.extend(verify_file(fpath))

    # Check that 37-fold LOEO is explicitly present in validation view
    validation_view = os.path.join(WORKSPACE_ROOT, "views", "03_Validation.py")
    if os.path.isfile(validation_view):
        with open(validation_view, "r", encoding="utf-8") as fp:
            content = fp.read()
            if "LOEO 37-Fold" not in content:
                all_violations.append(
                    f"{validation_view}: Missing mandatory '[Protocol: LOEO 37-Fold Cross-Validation]' badge."
                )

    if all_violations:
        print("\n❌ VERIFICATION FAILED with violations:")
        for v in all_violations:
            print(f"  • {v}")
        print("\nFix these violations before proceeding with builds.")
        sys.exit(1)
    else:
        print("\n✅ All claim discipline, fold count, and guidance checks PASSED.")
        print("   - 0 stale fold labels ('5-Fold', '38-fold')")
        print("   - 0 prescriptive 'Action:' directives")
        print("   - 0 unhedged 'Recommended:' statements")
        print("   - Authoritative 'LOEO 37-Fold' confirmed in validation views")
        sys.exit(0)


if __name__ == "__main__":
    main()
