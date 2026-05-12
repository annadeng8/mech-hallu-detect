import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.models.sae import MinimalSAE
from src.utils.config import load_config
from src.data.loaders import load_truthfulqa

def extract_hidden(model, tokenizer, texts, device):
    outs = []
    for t in texts:
        # Use with torch.no_grad() to save memory on your Mac
        with torch.no_grad():
            inp = tokenizer(t, return_tensors="pt").to(device)
            out = model(**inp, output_hidden_states=True)
            # Last layer, last token
            h = out.hidden_states[-1][:, -1, :]
            outs.append(h.detach())
    return torch.cat(outs, dim=0)

def main():
    cfg = load_config()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"], 
        torch_dtype=torch.float32
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    # 1. Load Dataset and prepare texts
    print("Loading TruthfulQA for SAE activations...")
    dataset = load_truthfulqa()
    # SAEs need a variety of data; taking 100 samples (True and False)
    subset = dataset['validation'] # .select(range(100))
    
    training_texts = []
    for row in subset:
        training_texts.append(f"Q: {row['question']} A: {row['best_answer']}")
        training_texts.append(f"Q: {row['question']} A: {row['incorrect_answers'][0]}")

    # 2. Extract activations
    print(f"Extracting hidden states for {len(training_texts)} examples...")
    hidden = extract_hidden(model, tokenizer, training_texts, device)

    # 3. Initialize SAE on MPS
    sae = MinimalSAE(hidden.shape[-1], cfg["sae"]["n_features"]).to(device).to(torch.float32)
    
    # Force LR to float to avoid the previous string error
    lr = float(cfg["training"]["lr"])
    opt = torch.optim.Adam(sae.parameters(), lr=lr)

    # 4. Training Loop
    print("Beginning SAE training...")
    num_epochs = 500 # Increased for SAE complexity
    for epoch in range(num_epochs):
        loss, _ = sae.loss(hidden)
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {loss.item():.4f}")

    torch.save(sae.state_dict(), "outputs/sae_real.pt")
    print("SAE training complete.")

if __name__ == "__main__":
    main()