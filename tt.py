import json

# Load JSON file
with open("rllm/data/test/code/livecodebench.json", "r") as f:
    data = json.load(f)

# Print first 5 rows
for row in data[:20]:
    print(row["starter_code"])
    print(row["starter_code"].strip() == "")
