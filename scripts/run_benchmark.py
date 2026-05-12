import json
from pathlib import Path


def main():
    results = {
        "placeholder": True,
        "note": "Integrate TruthfulQA/HaluEval loaders here.",
    }

    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Saved benchmark results")


if __name__ == "__main__":
    main()