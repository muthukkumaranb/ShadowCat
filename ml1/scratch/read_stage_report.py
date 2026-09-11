import os

input_path = "artifacts/lstm/stage_head/stage_head_report.md"

with open(input_path, "r", encoding="latin1") as f:
    text = f.read()

print("--- Original stage_head_report.md content ---")
print(text)
