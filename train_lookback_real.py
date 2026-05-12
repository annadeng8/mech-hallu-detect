import torch
import numpy as np
import joblib  # Standard for Sklearn
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils.config import load_config
from src.data.loaders import load_truthfulqa

def extract_attention(model, tokenizer, text, device):
    # Explicitly use eager implementation for attention capture
    inp = tokenizer(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        out = model(**inp, output_attentions=True)
    
    if out.attentions is None:
        raise ValueError("Attention weights were not captured. Ensure 'attn_implementation=eager' is set.")
    
    # Stack layers: [layers, batch, heads, seq, seq]
    attn = torch.stack(out.attentions) 
    # Mean across heads: [layers, seq, seq]
    return attn.mean(dim=2).squeeze(1).cpu().numpy() 

def build_features(attn, context_len):
    """
    attn should be shape: [num_layers, seq_len, seq_len]
    Returns: A 1D array of length [num_layers]
    """
    # Ensure attn is a 3D numpy array [Layers, Seq, Seq]
    if isinstance(attn, torch.Tensor):
        attn = attn.cpu().numpy()
        
    if attn.ndim == 2:
        # If somehow only one layer was passed, we need to handle it,
        # but for Llama-3, we expect 32 layers.
        attn = np.expand_dims(attn, axis=0)

    features = []
    for layer_idx in range(attn.shape[0]):
        layer_attn = attn[layer_idx] # [seq_len, seq_len]
        
        # Last token's attention row
        last_token_row = layer_attn[-1, :] 
        
        # Context vs Generated logic
        ctx_val = np.mean(last_token_row[:context_len]) + 1e-9
        gen_tokens = last_token_row[context_len:]
        
        if len(gen_tokens) == 0:
            gen_val = 1e-9
        else:
            gen_val = np.mean(gen_tokens) + 1e-9
            
        features.append(np.log(ctx_val / gen_val))
        
    return np.array(features) # Should be length 32

def main():
    cfg = load_config()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Ensure eager implementation is set here
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        torch_dtype=torch.float32,
        attn_implementation="eager" 
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    X, y = [], []
    dataset = load_truthfulqa()
    subset = dataset['validation'] # .select(range(200))

    print(f"Extracting Lookback features...")
    for row in subset:
        question = f"Q: {row['question']} A:"
        # Get actual token length of the prefix
        q_len = len(tokenizer.encode(question))
        
        # True Pair
        t_text = f"{question} {row['best_answer']}"
        X.append(build_features(extract_attention(model, tokenizer, t_text, device), q_len))
        y.append(1)

        # False Pair
        f_text = f"{question} {row['incorrect_answers'][0]}"
        X.append(build_features(extract_attention(model, tokenizer, f_text, device), q_len))
        y.append(0)

    X, y = np.array(X), np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Logistic Regression...")
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_train, y_train)

    # Stats
    y_probs = clf.predict_proba(X_test)[:, 1]
    print(f"\nAUC Score: {roc_auc_score(y_test, y_probs):.4f}")

    # THE CRITICAL FIX: Save using joblib with a modern protocol
    Path("outputs").mkdir(exist_ok=True)
    joblib.dump(clf, "outputs/lookback_real.pt")
    print("Model saved to outputs/lookback_real.pt using joblib.")

if __name__ == "__main__":
    main()