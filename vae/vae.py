import pickle
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import torch
import wandb
from rdkit import Chem
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.decomposition import PCA

latent_dim = 256
hidden_dim = 1024
emb_dim = 64
batch_size = 128
learning_rate = 1e-4
total_epochs = 30
data_frac = 0.2

temperature = 0.5

device = "cuda" if torch.cuda.is_available() else "mps"


wandb.init(
    project="smiles-vae",
    config={
        "latent_dim": latent_dim,
        "hidden_dim": hidden_dim,
        "emb_dim": emb_dim,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": total_epochs,
        "data_frac": data_frac,
        "temperature": temperature,
    },
)


df = pd.read_csv("../data/train.csv")
smiles_list = df["SMILES"].sample(frac=data_frac).tolist()


def tokenize(smiles):
    return list(smiles)  # character-level


tokens = [token for s in smiles_list for token in tokenize(s)]
vocab = ["<pad>", "<bos>", "<eos>", "<unk>"] + sorted(set(tokens))
stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for ch, i in stoi.items()}

MAX_LEN = 60


def encode(smiles):
    tokens = ["<bos>"] + tokenize(smiles) + ["<eos>"]
    idxs = [stoi.get(t, stoi["<unk>"]) for t in tokens]
    idxs = idxs[:MAX_LEN] + [stoi["<pad>"]] * (MAX_LEN - len(idxs))
    return idxs


input_tensor = torch.tensor([encode(s) for s in smiles_list])


import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, latent_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(
            emb_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        h = h[-1]  # last layer hidden state
        return self.fc_mu(h), self.fc_logvar(h)


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=1, batch_first=True)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, z, x):
        h = torch.tanh(self.fc(z)).unsqueeze(0).repeat(self.lstm.num_layers, 1, 1)
        c = torch.zeros_like(h)
        x = self.embedding(x)
        output, _ = self.lstm(x, (h, c))
        return self.out(output)


class VAE(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, hidden_dim=1024, latent_dim=256):
        super().__init__()
        self.encoder = Encoder(vocab_size, emb_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(vocab_size, emb_dim, hidden_dim, latent_dim)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        x_recon = self.decoder(z, x[:, :-1])  # teacher forcing
        return x_recon, mu, logvar


def vae_loss(epoch, recon_logits, x, mu, logvar):
    recon_loss = nn.CrossEntropyLoss(ignore_index=stoi["<pad>"])(
        recon_logits.view(-1, recon_logits.size(-1)), x[:, 1:].contiguous().view(-1)
    )
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    kl_weight = min(1.0, epoch / 10)

    kl_loss = kl_weight * kl_loss
    recon_loss = recon_loss * 500

    wandb.log(
        {
            "kl_loss": kl_loss.item(),
            "recon_loss": recon_loss.item(),
            "total_loss": (recon_loss + kl_loss).item(),
            "mu_mean": mu.mean().item(),
            "logvar_mean": logvar.mean().item(),
            "mu_hist": wandb.Histogram(mu.detach().cpu()),
            "logvar_hist": wandb.Histogram(logvar.detach().cpu())
        }
    )

    return recon_loss + kl_loss


model = VAE(
    len(vocab), latent_dim=latent_dim, emb_dim=emb_dim, hidden_dim=hidden_dim
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

dataset = TensorDataset(input_tensor)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(total_epochs):
    model.train()
    total_loss = 0
    for (batch,) in tqdm(loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        recon_logits, mu, logvar = model(batch)
        loss = vae_loss(epoch, recon_logits, batch, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}: Loss = {total_loss / len(loader):.4f}")


model.eval()


def generate_smiles(i):
    with torch.no_grad():
        z = torch.randn(1, latent_dim).to(device)
        start_token = torch.tensor([[stoi["<bos>"]]]).to(device)
        generated = [start_token]

        for _ in range(MAX_LEN):
            inp = torch.cat(generated, dim=1)
            logits = model.decoder(z, inp)
            probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated.append(next_token)
            if next_token.item() == stoi["<eos>"]:
                break

        decoded = "".join(
            [
                itos[t.item()]
                for t in torch.cat(generated, dim=1)[0]
                if t.item() not in [stoi["<bos>"], stoi["<eos>"], stoi["<pad>"]]
            ]
        )
        print("Generated SMILES:", decoded)

        mol = Chem.MolFromSmiles(decoded)
        if mol:
            print("Valid molecule!")
        else:
            print("Invalid SMILES.")
        
        wandb.log({
            "generated_smiles": decoded,
            "temperature": temperature,
        }, step=i)


SMILES_TO_GENERATE = 5
for i in range(SMILES_TO_GENERATE):
    generate_smiles(i)


# Save the model
save_path = f"vae_model.pt"

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "stoi": stoi,
        "itos": itos,
        "config": {
            "vocab_size": len(vocab),
            "emb_dim": emb_dim,
            "hidden_dim": hidden_dim,
            "latent_dim": latent_dim,
        },
    },
    save_path,
)

print(f"Model saved to {save_path}")

artifact = wandb.Artifact(
    name="vae-model",
    type="model",
    metadata={
        "epochs": total_epochs,
        "latent_dim": latent_dim,
        "hidden_dim": hidden_dim,
        "emb_dim": emb_dim,
        "learning_rate": learning_rate
    }
)
artifact.add_file(save_path)
wandb.log_artifact(artifact)



model.eval()
latents = []
labels = []  # optional — e.g., molecular weight, scaffold class, etc.

with torch.no_grad():
    for (batch,) in DataLoader(dataset, batch_size=64):
        batch = batch.to(device)
        mu, logvar = model.encoder(batch)
        z = mu  # or reparameterize(mu, logvar)
        latents.append(z.cpu())
        # optionally add a property
        labels.extend(
            [len(s[s != stoi["<pad>"]]) for s in batch.cpu()]
        )  # e.g., length of SMILES

latents = torch.cat(latents).numpy()

pca = PCA(n_components=2)
latents_2d = pca.fit_transform(latents)

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(latents_2d[:, 0], latents_2d[:, 1], c=labels, cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label="SMILES Length" if labels else "")
ax.set_title("2D PCA of Molecular Latent Space")
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.grid(True)
wandb.log({"latent_space": wandb.Image(fig)})
plt.close()
