import torch
from src.models.sae import MinimalSAE


def main():
    model = MinimalSAE(4096, 256)
    torch.save(model.state_dict(), "outputs/sae.pt")
    print("Saved SAE model")


if __name__ == "__main__":
    main()