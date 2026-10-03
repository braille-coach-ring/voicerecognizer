import json
from collections import Counter
from pathlib import Path

decisions_path = Path("evaluation_results/review_decisions.json")
if not decisions_path.exists():
    print("review_decisions.json not found!")
    exit(1)

with open(decisions_path, encoding="utf-8") as f:
    data = json.load(f)

decisions = data.get("decisions", [])
print(f"Total decisions recorded: {len(decisions)}")

counts = Counter()
relabel_details = []
other_details = []
maybe_details = []
keep_details = []

for d in decisions:
    dec = d.get("decision")
    new_lbl = d.get("new_label", "")
    if dec == "other" or new_lbl == "other":
        counts["other"] += 1
        other_details.append(d)
    elif dec == "relabel":
        counts["relabel"] += 1
        relabel_details.append(d)
    elif dec == "keep":
        counts["keep"] += 1
        keep_details.append(d)
    elif dec == "maybe":
        counts["maybe"] += 1
        maybe_details.append(d)
    else:
        counts[dec] += 1

print("\n--- Summary counts ---")
for k, v in counts.items():
    print(f"  {k}: {v}")

print(f"\n--- Relabel details ({len(relabel_details)} items) ---")
for r in relabel_details:
    print(f"  {r.get('filepath')}: {r.get('label')} -> {r.get('new_label')} (predicted: {r.get('prediction')})")

print(f"\n--- Other (noise) details ({len(other_details)} items) ---")
for o in other_details[:30]:
    print(f"  {o.get('filepath')}: label={o.get('label')}, pred={o.get('prediction')}")
if len(other_details) > 30:
    print(f"  ... and {len(other_details) - 30} more other items")

print(f"\n--- Maybe details ({len(maybe_details)} items) ---")
for m in maybe_details[:30]:
    print(f"  {m.get('filepath')}: label={m.get('label')}, pred={m.get('prediction')}")
