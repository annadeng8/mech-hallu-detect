import json
import numpy as np

def analyze_triage_errors(results_path):
    with open(results_path, 'r') as f:
        data = json.load(f)

    # Aggregators for per-token distributions
    metrics = {
        "correct": {"t1": [], "t2": [], "t3": [], "count": 0},
        "incorrect": {"t1": [], "t2": [], "t3": [], "count": 0}
    }

    print(f"--- Processing {len(data)} Samples ---")

    for entry in data:
        # Map the boolean 'correct' to our aggregator keys
        label = "correct" if entry["correct"] else "incorrect"
        stats = entry["stats"]
        
        metrics[label]["t1"].append(stats.get("tier1_frac", 0))
        metrics[label]["t2"].append(stats.get("tier2_frac", 0))
        metrics[label]["t3"].append(stats.get("tier3_frac", 0))
        metrics[label]["count"] += 1

    print("\n--- Triage Efficiency Report (Mean Token Fractions) ---")
    
    for label in ["correct", "incorrect"]:
        count = metrics[label]["count"]
        if count == 0:
            print(f"\n[{label.upper()}] No samples found.")
            continue
        
        avg_t1 = np.mean(metrics[label]["t1"])
        avg_t2 = np.mean(metrics[label]["t2"])
        avg_t3 = np.mean(metrics[label]["t3"])
        
        print(f"\n[{label.upper()}] (N={count})")
        print(f"  Avg Tier 1 (Greedy):  {avg_t1:.2%}")
        print(f"  Avg Tier 2 (DoLa):    {avg_t2:.2%}")
        print(f"  Avg Tier 3 (Resample): {avg_t3:.2%}")

    # --- ACTIONABLE ADVICE ---
    print("\n--- Diagnostic Advice ---")
    
    # 1. Check for Over-Skepticism (The likely culprit for your 59% accuracy)
    avg_t1_correct = np.mean(metrics["correct"]["t1"]) if metrics["correct"]["count"] > 0 else 1
    if avg_t1_correct < 0.20:
        print("🚩 PROBLEM: System is over-thinking. Even correct answers are forced into DoLa/Resampling.")
        print("   FIX: Lower your weights (w1/w2) or decrease tau1. Allow more 'obvious' tokens to pass.")

    # 2. Check for Hallucination Leakage
    if metrics["incorrect"]["count"] > 0:
        avg_t1_incorrect = np.mean(metrics["incorrect"]["t1"])
        if avg_t1_incorrect > 0.20:
            print("🚩 PROBLEM: Hallucinations are leaking through Greedy decoding.")
            print("   FIX: Increase tau1 or increase w2 (TSV weight) to be more sensitive.")

    # 3. Check for Tier 3 Under-utilization
    if metrics["incorrect"]["count"] > 0:
        avg_t3_incorrect = np.mean(metrics["incorrect"]["t3"])
        if avg_t3_incorrect < 0.05:
            print("🚩 PROBLEM: Tier 3 (Resampling) is rarely triggered on incorrect answers.")
            print("   FIX: Lower tau2 to trigger the 'emergency' mode more easily.")

def main():
    # Ensure this path matches your actual output file
    analyze_triage_errors("outputs/dataset_results.json")

if __name__ == "__main__":
    main()