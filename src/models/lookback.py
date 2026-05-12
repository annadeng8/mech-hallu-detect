import torch
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils.config import load_config
from src.data.loaders import load_truthfulqa  # Import your loader

def extract_attention(model, tokenizer, text, device):
    # Ensure inputs are on the correct device
    inp = tokenizer(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        out = model(**inp, output_attentions=True)
    
    # If attn_implementation wasn't 'eager', out.attentions is None
    if out.attentions is None:
        raise ValueError("Attention weights were not captured. Ensure 'attn_implementation=eager' is set.")
    
    # out.attentions is a tuple (one per layer), stack them into a single tensor
    attn = torch.stack(out.attentions) # [layers, batch, heads, seq, seq]
    
    # Return mean across heads for the last token's attention
    return attn.mean(dim=2).squeeze(1).cpu().numpy()

def build_features(attn, context_len):
    features = []
    # attn shape: [layers, seq_len, seq_len]
    for layer in attn:
        # Get attention weights of the very last token
        last_token_attn = layer[-1, :] 
        
        # Mean attention to context tokens vs generated/other tokens
        ctx_val = last_token_attn[:context_len].mean()
        gen_val = last_token_attn[context_len:].mean() if len(last_token_attn) > context_len else 1e-6
        
        features.append(ctx_val / (gen_val + 1e-6))
    return np.array(features)

def main():
    cfg = load_config()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        torch_dtype=torch.float32,
        attn_implementation="eager"
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    X, y = [], []

    # 1. Load the real dataset
    print("Loading TruthfulQA...")
    dataset = load_truthfulqa()
    
    # 2. Process a subset (e.g., first 50 samples) to test
    # TruthfulQA 'generation' split has 'question' and 'best_answer' (True) 
    # and 'incorrect_answers' (False)
    subset_limit = 50 
    samples = dataset['validation'].select(range(subset_limit))

    print(f"Extracting features from {subset_limit} samples...")
    for i, row in enumerate(samples):
        question = row['question']
        
        # We need a positive example and a negative example for the classifier
        # True example
        true_text = f"Question: {question} Answer: {row['best_answer']}"
        attn_true = extract_attention(model, tokenizer, true_text, device)
        X.append(build_features(attn_true, context_len=len(tokenizer.encode(question))))
        y.append(1)

        # False example (pick the first incorrect answer)
        false_text = f"Question: {question} Answer: {row['incorrect_answers'][0]}"
        attn_false = extract_attention(model, tokenizer, false_text, device)
        X.append(build_features(attn_false, context_len=len(tokenizer.encode(question))))
        y.append(0)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{subset_limit} questions...")

    # 3. Train
    print("Training Logistic Regression...")
    X_train = np.array(X)
    y_train = np.array(y)
    
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # 4. Save
    output_path = Path("outputs")
    output_path.mkdir(exist_ok=True)
    torch.save(clf, output_path / "lookback_real.pt")
    print("Done!")

if __name__ == "__main__":
    main()