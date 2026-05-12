"""Run end-to-end Diagnostic-Driven Contrastive Decoding (D-DCD).

This script loads:
- A pretrained causal LM.
- A trained Truthfulness Separator Vector (TSV) probe.
- A trained Minimal Sparse Autoencoder (SAE).
- A trained Lookback Lens classifier.
- DoLa contrastive decoder.
- Tree-based resampling module.

It then performs token-by-token generation using the full D-DCD controller.
"""

import argparse
import json
from pathlib import Path

import joblib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import tqdm

from src.utils.config import load_config
from src.models.tsv import TSVProbe
from src.models.sae import MinimalSAE
# from src.models.lookback import LookbackLens
from src.decoding.dola import DoLaDecoder
from src.decoding.tree_resample import TreeResampler
from src.decoding.d_dcd import DDCDDecoder

from src.eval.metrics import summarize_logs
from src.eval.stats import bootstrap_ci
from src.eval.checker import TruthChecker

from src.data.loaders import load_truthfulqa


# ---------------------------------------------------------------------
# Loading Utilities
# ---------------------------------------------------------------------


def resolve_device(config_device: str) -> torch.device:
    if config_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if config_device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")



def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer



def load_model(model_name: str, device: torch.device):
    kwargs = {
        "torch_dtype": torch.float32,
        "attn_implementation": "eager"  
    }

    if device.type == "cuda":
        kwargs["torch_dtype"] = torch.float16
        kwargs["device_map"] = "auto"
    
    print(f"Loading {model_name} to {device}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    # Manually move to device if not already handled by device_map
    if device.type == "mps" or device.type == "cpu":
        model = model.to(device)

    model.eval()
    return model



def load_tsv(path: str, hidden_size: int, device: torch.device):
    """
    Loads a TSV Probe. 
    If pooling='answer_layeravg' was used, the hidden_size passed here 
    should be (model_hidden_size * layer_avg_k).
    """
    if not Path(path).exists():
        print(f"Warning: TSV file not found at {path}. Initializing random probe.")
        return TSVProbe(hidden_size).to(device)

    # Load the state file
    state = torch.load(path, map_location=device)
    
    # 1. Determine the actual dimension of the saved weights
    # We check the shape of the 'truth_centroid' in the saved file
    if isinstance(state, dict) and "truth_centroid" in state:
        saved_dim = state["truth_centroid"].shape[-1]
    elif hasattr(state, "mu_truth"):
        saved_dim = state.mu_truth.shape[-1]
    else:
        saved_dim = hidden_size

    print(f"Initializing TSV Probe with dimension: {saved_dim}")
    tsv = TSVProbe(saved_dim).to(device)

    # 2. Load the data into the probe
    if isinstance(state, dict):
        # strict=False allows loading even if there are slight version mismatches
        tsv.load_state_dict(state, strict=False)
    else:
        # This handles cases where saved the Trainer object directly
        with torch.no_grad():
            if hasattr(state, "mu_truth"):
                tsv.truth_centroid.copy_(state.mu_truth)
            if hasattr(state, "mu_hall"):
                tsv.halluc_centroid.copy_(state.mu_hall)

    tsv.eval()
    return tsv


def load_sae(path: str, hidden_size: int, n_features: int, device: torch.device):
    sae = MinimalSAE(hidden_size, n_features)

    if Path(path).exists():
        state = torch.load(path, map_location=device)
        sae.load_state_dict(state, strict=False)

    sae = sae.to(device)
    sae.eval()
    return sae



def load_lookback(path: str):
    if not Path(path).exists():
        return None
    try:
        # If saved via torch.save(clf, ...)
        obj = torch.load(path, map_location="cpu")
        return obj
    except Exception:
        try:
            # If saved via joblib.dump(clf, ...)
            return joblib.load(path)
        except Exception as e:
            print(f"Error loading Lookback Lens: {e}")
            return None
        

class LLMJudge:
    def __init__(self, model_name="prometheus-eval/prometheus2-7b-v2", device="mps"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="mps", 
            torch_dtype=torch.bfloat32)
        # Use the same device as the model

    def check(self, question, gold_options, generated_answer):
        formatted_gold = "\n".join([f"- {opt}" for opt in gold_options])
        
        # This prompt is designed to be concise for 8B models
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a factual correctness judge. Compare the generated answer to the reference options. 
If the generated answer is factually consistent with any reference, output 'Yes'. 
If it contradicts the references, output 'No'.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Question: {question}
References:
{formatted_gold}

Generated Answer: {generated_answer}

Decision (Yes/No):<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs, 
                max_new_tokens=2, 
                temperature=0.01,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the newly generated tokens
        full_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        decision_text = full_text.split("Decision (Yes/No):")[-1].strip().lower()
        
        return 1.0 if "yes" in decision_text else 0.0

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="D-DCD Generation & Evaluation")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt", type=str, help="Single prompt mode")
    group.add_argument("--eval-dataset", action="store_true", help="Dataset evaluation mode")

    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--tsv-path", default="outputs/tsv_real.pt")
    parser.add_argument("--sae-path", default="outputs/sae_real.pt")
    parser.add_argument("--lookback-path", default="outputs/lookback_real.pt")
    parser.add_argument("--num-samples", type=int, default=50)
    
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg.get("device", "mps"))

    tokenizer = load_tokenizer(cfg["model_name"])
    model = load_model(cfg["model_name"], device)
    
    k = cfg.get("layer_avg_k", 4)
    expected_dim = model.config.hidden_size * k if cfg.get("pooling") == "answer_layeravg" else model.config.hidden_size
    
    tsv = load_tsv(args.tsv_path, expected_dim, device)
    sae = load_sae(args.sae_path, model.config.hidden_size, cfg["sae"]["n_features"], device)
    lookback = load_lookback(args.lookback_path)

    dola = DoLaDecoder(model=model, premature_layers=cfg["layers"]["dola_candidates"])
    resampler = TreeResampler(tokenizer=tokenizer, k=cfg["resampling"]["k"])
    decoder = DDCDDecoder(model, tokenizer, tsv, sae, lookback, dola, resampler, cfg)

    system_message = (
        "You are a factual research assistant. Answer concisely and accurately. "
        "Stop after one sentence."
    )

    if args.eval_dataset:
        checker = TruthChecker(device=device)
        # checker = LLMJudge(model, tokenizer)
        print(f"Starting Dataset Evaluation on TruthfulQA ({args.num_samples} samples)...")
        dataset = load_truthfulqa()['validation'].select(range(args.num_samples))
        
        all_results, all_latencies, scores = [], [], []
        
        for row in tqdm.tqdm(dataset):
            # Formatted Prompt to avoid rambling
            full_prompt = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\nQ: {row['question']} A:<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
            
            # Generate using D-DCD Controller
            answer_only = decoder.generate(full_prompt)

            # Check against all correct answers
            is_correct = checker.check(row['question'], row['correct_answers'], answer_only)
            scores.append(is_correct)
            
            step_stats = summarize_logs(decoder.logs)
            all_latencies.append(step_stats["mean_latency_ms"])
            
            all_results.append({
                "question": row['question'],
                "generated": answer_only,
                "gold_options": row['correct_answers'],
                "correct": bool(is_correct),
                "stats": step_stats
            })
            decoder.logs = [] 
            if device.type == "mps":
                torch.mps.empty_cache()

        ci_lat = bootstrap_ci(all_latencies)
        final_acc = (sum(scores) / len(scores)) * 100
        
        print("\n" + "="*30)
        print(f"EVALUATION COMPLETE")
        print(f"FINAL D-DCD ACCURACY: {final_acc:.2f}%")
        print(f"Mean Latency: {np.mean(all_latencies):.2f}ms (95% CI: {ci_lat[0]:.2f}-{ci_lat[1]:.2f})")
        print("="*30)
        
        with open("outputs/dataset_results.json", "w") as f:
            json.dump(all_results, f, indent=2)

    else:
        # Single Prompt with System Message
        full_prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\nQ: {args.prompt} A:<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        print(f"Generating for: {args.prompt}")
        output = decoder.generate(full_prompt)
        answer_only = output.replace(full_prompt, "").strip()
        print(f"\nResponse: {answer_only}")
        
        stats = summarize_logs(decoder.logs)
        print(f"\nDiagnostic Summary:")
        print(f"- Tier 1 (Greedy): {stats['tier1_frac']*100:.1f}%")
        print(f"- Tier 2 (DoLa):   {stats['tier2_frac']*100:.1f}%")
        print(f"- Tier 3 (Resample): {stats['tier3_frac']*100:.1f}%")

if __name__ == "__main__":
    main()