import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.models.tsv_train import train_tsv
from src.models.tsv import TSVProbe
from src.utils.config import load_config
from src.data.loaders import load_truthfulqa
import argparse

def _answer_span_hidden(outputs, prefix_len, layer_idx=-1):
    return outputs.hidden_states[layer_idx][0, prefix_len:, :]


def extract_answer_representation(
    model,
    tokenizer,
    question,
    answer,
    device,
    pooling="answer_layeravg",
    layer_avg_k=4,
):
    prefix = f"Q: {question} A:"
    text = f"{prefix} {answer}"

    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
    prefix_len = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

        answer_tokens = _answer_span_hidden(outputs, prefix_len, layer_idx=-1)

        if pooling == "last":
            last_hidden = answer_tokens[-1]

        elif pooling == "mean":
            last_hidden = answer_tokens.mean(dim=0)

        elif pooling == "layeravg":
            layers = outputs.hidden_states[-layer_avg_k:]
            pooled_layers = [layer[0, prefix_len:, :].mean(dim=0) for layer in layers]
            last_hidden = torch.stack(pooled_layers, dim=0).mean(dim=0)

        elif pooling == "answer_layeravg":
            layers = outputs.hidden_states[-layer_avg_k:]
            pooled_layers = [layer[0, prefix_len:, :].mean(dim=0) for layer in layers]
            last_hidden = torch.cat(pooled_layers, dim=-1)

        else:
            raise ValueError("Unknown pooling: %s" % pooling)

    return last_hidden

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pooling", choices=["last", "mean", "layeravg", "answer_layeravg"], default="answer_layeravg")
    parser.add_argument("--layer-avg-k", type=int, default=4)
    parser.add_argument("--iters", type=int, default=250)
    parser.add_argument("--init", choices=["random", "kmeans", "supervised"], default="supervised")
    parser.add_argument("--baseline", action="store_true", help="Train logistic baseline on extracted features")
    args = parser.parse_args()

    cfg = load_config()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], torch_dtype=torch.float32).to(device)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    print("Extracting hidden states...")
    dataset = load_truthfulqa()
    subset = dataset['validation'] #.select(range(200))

    all_states, all_labels = [], []
    for row in subset:
        all_states.append(
            extract_answer_representation(
                model,
                tokenizer,
                row['question'],
                row['best_answer'],
                device,
                pooling=args.pooling,
                layer_avg_k=args.layer_avg_k,
            )
        )
        all_labels.append(1)

        all_states.append(
            extract_answer_representation(
                model,
                tokenizer,
                row['question'],
                row['incorrect_answers'][0],
                device,
                pooling=args.pooling,
                layer_avg_k=args.layer_avg_k,
            )
        )
        all_labels.append(0)

    states = torch.stack(all_states)
    labels = np.array(all_labels)

    # Split data
    train_idx, test_idx = train_test_split(np.arange(len(labels)), test_size=0.2, random_state=42)

    # Optionally run a quick logistic baseline
    if args.baseline:
        print("Running logistic baseline...")
        X = states.cpu().numpy()
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        clf = LogisticRegression(class_weight='balanced', max_iter=2000)
        clf.fit(X_train, y_train)
        probs = clf.decision_function(X_test)
        preds = (probs > 0).astype(int)

        print("\n--- Logistic Baseline Metrics ---")
        print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
        print(f"AUC Score: {roc_auc_score(y_test, probs):.4f}")

    # Train Centroids on the training set only
    print(f"Training centroids on {len(train_idx)} states with init={args.init}...")
    trainer = train_tsv(
        states[train_idx].to(device),
        labels=labels[train_idx],
        iters=args.iters,
        init=args.init,
    )

    # Initialize Probe
    probe = TSVProbe(states.shape[-1]).to(device)
    with torch.no_grad():
        probe.truth_centroid.copy_(trainer.mu_truth)
        probe.halluc_centroid.copy_(trainer.mu_hall)

    # Evaluate on Test Set
    print("Evaluating TSV Probe...")
    test_states = states[test_idx].to(device)
    test_labels = labels[test_idx]

    with torch.no_grad():
        # probe.score() returns (truth_sim - hall_sim)
        scores = probe.score(test_states).cpu().numpy()
        preds = (scores > 0).astype(int)

    print("\n--- TSV Probe Metrics ---")
    print(f"Accuracy: {accuracy_score(test_labels, preds):.4f}")
    print(f"AUC Score: {roc_auc_score(test_labels, scores):.4f}")

    torch.save(probe.state_dict(), "outputs/tsv_real.pt")

if __name__ == "__main__":
    main()