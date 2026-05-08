"""
hallucination_research_pipeline.py

FULL SIMPLE PIPELINE FOR:
1. Collecting transformer activations
2. Training an SAE (Sparse Autoencoder)
3. Detecting hallucination-related features
4. Running causal interventions
5. Evaluating hallucination mitigation

===========================================================
INSTALL:
pip install torch transformers datasets tqdm scikit-learn

OPTIONAL:
pip install matplotlib pandas

===========================================================
THIS IS A RESEARCH SKELETON.

You will still need:
- larger models
- better datasets
- cleaner evaluation
- proper experiment tracking

But this gives you a complete starting point.
===========================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# ===========================================================
# CONFIG
# ===========================================================

MODEL_NAME = "gpt2"

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "mps"

LAYER_IDX = 5

LATENT_DIM = 2048

N_SAMPLES = 200

MAX_TOKENS = 64

SAE_EPOCHS = 5

BATCH_SIZE = 256

L1_COEFF = 1e-3

INTERVENTION_STRENGTH = 1.0


# ===========================================================
# LOAD MODEL
# ===========================================================

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    output_hidden_states=True,
    output_attentions=True,
    torch_dtype=torch.float32
).to(DEVICE)

model.eval()

HIDDEN_DIM = model.config.n_embd

print("Hidden dim:", HIDDEN_DIM)


# ===========================================================
# LOAD DATASET
# ===========================================================

print("Loading dataset...")

dataset = load_dataset(
    "truthful_qa",
    "generation"
)["validation"]


# ===========================================================
# STEP 1:
# COLLECT ACTIVATIONS
# ===========================================================

def collect_activations():
    """
    Extract hidden states from transformer layer.
    """

    activations = []

    texts = []

    print("Collecting activations...")

    for i in tqdm(range(N_SAMPLES)):

        question = dataset[i]["question"]

        inputs = tokenizer(
            question,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_TOKENS
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)

        # hidden states from chosen layer
        h = outputs.hidden_states[LAYER_IDX]

        # shape:
        # [batch, tokens, hidden_dim]

        h = h.squeeze(0)

        activations.append(h.cpu())

        # save matching tokens
        token_ids = inputs["input_ids"][0]

        tokens = tokenizer.convert_ids_to_tokens(token_ids)

        texts.extend(tokens)

    activations = torch.cat(activations, dim=0)

    return activations, texts


# ===========================================================
# STEP 2:
# SPARSE AUTOENCODER
# ===========================================================

class SparseAutoencoder(nn.Module):

    def __init__(self, input_dim, latent_dim):
        super().__init__()

        self.encoder = nn.Linear(
            input_dim,
            latent_dim,
            bias=False
        )

        self.decoder = nn.Linear(
            latent_dim,
            input_dim,
            bias=False
        )

    def forward(self, x):

        # normalize inputs
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-6)

        # encode
        z = self.encoder(x)

        # sparse activation
        z = F.relu(z)

        # reconstruct
        recon = self.decoder(z)

        return z, recon


# ===========================================================
# STEP 3:
# TRAIN SAE
# ===========================================================

def train_sae(sae, data):

    print("Training SAE...")

    sae.train()

    optimizer = torch.optim.Adam(
        sae.parameters(),
        lr=1e-3
    )

    data = data.to(DEVICE)

    for epoch in range(SAE_EPOCHS):

        perm = torch.randperm(data.size(0))

        total_loss = 0

        for i in range(0, data.size(0), BATCH_SIZE):

            idx = perm[i:i+BATCH_SIZE]

            batch = data[idx]

            z, recon = sae(batch)

            # reconstruction loss
            recon_loss = F.mse_loss(recon, batch)

            # sparsity penalty
            sparsity_loss = z.abs().mean()

            loss = recon_loss + L1_COEFF * sparsity_loss

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch} | Loss {total_loss:.4f}")


# ===========================================================
# STEP 4:
# GET LATENTS
# ===========================================================

def extract_latents(sae, activations):

    sae.eval()

    latents = []

    with torch.no_grad():

        for i in range(0, activations.size(0), BATCH_SIZE):

            batch = activations[i:i+BATCH_SIZE].to(DEVICE)

            z, _ = sae(batch)

            latents.append(z.cpu())

    latents = torch.cat(latents, dim=0)

    return latents


# ===========================================================
# STEP 5:
# INTERPRET FEATURES
# ===========================================================

def show_top_tokens(latents, tokens, top_k=10):
    """
    Show which tokens activate features most.
    """

    print("\n=== TOP TOKENS PER FEATURE ===\n")

    for feature_idx in range(5):

        values = latents[:, feature_idx]

        top_idx = torch.topk(values, k=top_k).indices

        top_tokens = [tokens[i] for i in top_idx]

        print(f"Feature {feature_idx}")
        print(top_tokens)
        print()


# ===========================================================
# STEP 6:
# CREATE HALLUCINATION LABELS
# ===========================================================

def create_fake_labels(n):
    """
    TEMPORARY PLACEHOLDER.

    Replace with real hallucination labels later.
    """

    labels = np.random.randint(0, 2, size=n)

    return labels


# ===========================================================
# STEP 7:
# FIND HALLUCINATION FEATURES
# ===========================================================

def find_hallucination_features(latents, labels):

    print("Finding hallucination-correlated features...")

    latents_np = latents.numpy()

    scores = []

    for i in range(latents_np.shape[1]):

        feature_values = latents_np[:, i]

        corr = np.corrcoef(
            feature_values,
            labels
        )[0, 1]

        if np.isnan(corr):
            corr = 0

        scores.append(corr)

    scores = np.array(scores)

    top_features = np.argsort(scores)[-10:]

    print("Top hallucination features:")
    print(top_features)

    return top_features


# ===========================================================
# STEP 8:
# CAUSAL INTERVENTION HOOK
# ===========================================================

def make_intervention_hook(
    sae,
    hallucination_features,
    strength=1.0
):
    """
    Suppress hallucination features during generation.
    """

    def hook(module, input, output):

        h = output

        original_shape = h.shape

        h_flat = h.reshape(-1, h.shape[-1])

        z, _ = sae(h_flat)

        # suppress chosen features
        z[:, hallucination_features] *= (
            1.0 - strength
        )

        new_h = sae.decoder(z)

        new_h = new_h.reshape(original_shape)

        return new_h

    return hook


# ===========================================================
# STEP 9:
# GENERATE TEXT
# ===========================================================

def generate(prompt):

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False
        )

    text = tokenizer.decode(outputs[0])

    return text


# ===========================================================
# STEP 10:
# RUN EXPERIMENT
# ===========================================================

def run_experiment():

    # -----------------------------------
    # collect activations
    # -----------------------------------

    activations, tokens = collect_activations()

    print("Activation shape:", activations.shape)

    # -----------------------------------
    # train SAE
    # -----------------------------------

    sae = SparseAutoencoder(
        input_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM
    ).to(DEVICE)

    train_sae(sae, activations)

    # -----------------------------------
    # extract latents
    # -----------------------------------

    latents = extract_latents(
        sae,
        activations
    )

    print("Latent shape:", latents.shape)

    # -----------------------------------
    # interpret features
    # -----------------------------------

    show_top_tokens(
        latents,
        tokens
    )

    # -----------------------------------
    # create labels
    # -----------------------------------

    labels = create_fake_labels(
        latents.shape[0]
    )

    # -----------------------------------
    # find hallucination features
    # -----------------------------------

    hallucination_features = find_hallucination_features(
        latents,
        labels
    )

    # -----------------------------------
    # baseline generation
    # -----------------------------------

    prompt = (
        "Who won the Nobel Prize "
        "in Physics in 1800?"
    )

    print("\n=== BASELINE ===\n")

    baseline = generate(prompt)

    print(baseline)

    # -----------------------------------
    # intervention
    # -----------------------------------

    print("\n=== INTERVENTION ===\n")

    hook = model.transformer.h[
        LAYER_IDX
    ].register_forward_hook(

        make_intervention_hook(
            sae,
            hallucination_features,
            strength=INTERVENTION_STRENGTH
        )
    )

    intervened = generate(prompt)

    print(intervened)

    hook.remove()

    print("\nDone.")


# ===========================================================
# MAIN
# ===========================================================

if __name__ == "__main__":

    run_experiment()