import json

with open("evaluation_results/benchmark_si_test.json", encoding="utf-8") as f:
    data = json.load(f)

overall = data["overall"]
print("=== Overall Benchmark Results ===")
print(f"Accuracy:    {overall['accuracy']*100:.2f}%")
print(f"Macro F1:    {overall['macro_f1']:.4f}")
print(f"Weighted F1: {overall['weighted_f1']:.4f}")
print(f"Total:       {overall['total_samples']} samples")

print("\n=== Speaker Breakdown ===")
sp_metrics = data.get("speaker_metrics", {})
for sp, m in sp_metrics.items():
    acc = m["accuracy"] * 100
    corr = m["correct_samples"]
    tot = m["total_samples"]
    print(f"  {sp:12s}: Accuracy = {acc:6.2f}% ({corr:3d} / {tot:3d})")
