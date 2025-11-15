import argparse
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_attention(attention_weights, save_path):
    """
    attention_weights: List[List[Tensor]] -> first batch -> layer list -> (heads averaged)
    """
    first_batch = attention_weights[0]   # first batch
    attn_per_layer = []

    for layer_attn in first_batch:
        # layer_attn: (batch, heads, seq, seq)
        averaged = layer_attn.mean(dim=1)[0]  # first example mean heads -> (seq, seq)
        attn_per_layer.append(averaged.cpu())

    fig, axes = plt.subplots(1, len(attn_per_layer), figsize=(5*len(attn_per_layer), 4))
    if len(attn_per_layer) == 1:
        axes = [axes]

    for i, attn in enumerate(attn_per_layer):
        sns.heatmap(attn.numpy(), ax=axes[i])
        axes[i].set_title(f"Layer {i} CLS Attention")
        axes[i].set_xlabel("Token Index")
        axes[i].set_ylabel("Token Index")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[VIS] Saved attention heatmap -> {save_path}")


def plot_embedding(embedding_tensor, save_path):
    """
    embedding_tensor: first batch -> combined_embeddings -> shape (seq, dim)
    """
    seq, dim = embedding_tensor.shape

    plt.figure(figsize=(6, 8))
    sns.heatmap(embedding_tensor.numpy(), cmap="viridis")
    plt.title("Embedding Heatmap (Seq x Hidden)")
    plt.xlabel("Hidden Dim")
    plt.ylabel("Token Index")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[VIS] Saved embedding heatmap -> {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True, help="embedding_snapshot.pt")
    parser.add_argument("--attn", required=True, help="attention_weights.pt")
    parser.add_argument("--outdir", default="vis")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    emb = torch.load(args.snapshot)["batches"][0]["combined_embeddings"][0]  # (seq, dim)
    attn = torch.load(args.attn)  # list

    plot_embedding(emb, os.path.join(args.outdir, "embedding.png"))
    plot_attention(attn, os.path.join(args.outdir, "attention.png"))


if __name__ == "__main__":
    main()
