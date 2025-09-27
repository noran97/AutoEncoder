import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ============================
# Parameters
# ============================
LATENT_DIM = 20
IMAGE_SIZE = 28 * 28
EPOCHS = 10
BATCH_SIZE = 128
LR = 1e-3

# ============================
# DataLoader
# ============================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # normalize to [-1, 1]
])

train_dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ============================
# Encoder
# ============================
class Encoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(IMAGE_SIZE, 256)
        self.fc_mu = nn.Linear(256, latent_dim)      # mean
        self.fc_logvar = nn.Linear(256, latent_dim)  # log variance
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, IMAGE_SIZE)   # flatten
        h = self.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

# ============================
# Reparameterization Trick
# ============================
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

# ============================
# Decoder
# ============================
class Decoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, IMAGE_SIZE)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()  # match input normalization [-1, 1]

    def forward(self, z):
        h = self.relu(self.fc1(z))
        x_recon = self.tanh(self.fc2(h))
        return x_recon.view(-1, 1, 28, 28)

# ============================
# VAE Model
# ============================
class VAE(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

# ============================
# Loss Function
# ============================
def vae_loss(x_recon, x, mu, logvar):
    recon_loss = nn.MSELoss(reduction="sum")(x_recon, x)  # reconstruction
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # KL divergence
    return (recon_loss + kl_loss) / x.size(0)

# ============================
# Training Loop
# ============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE(LATENT_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(1, EPOCHS+1):
    model.train()
    train_loss = 0
    for x, _ in train_loader:
        x = x.to(device)
        optimizer.zero_grad()
        x_recon, mu, logvar = model(x)
        loss = vae_loss(x_recon, x, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print(f"Epoch [{epoch}/{EPOCHS}], Loss: {train_loss/len(train_loader):.4f}")

    # Quick visualization every 5 epochs
    if epoch % 5 == 0:
        model.eval()
        with torch.no_grad():
            x, _ = next(iter(test_loader))
            x = x.to(device)
            x_recon, _, _ = model(x)
            fig, axes = plt.subplots(2, 8, figsize=(12, 3))
            for i in range(8):
                axes[0, i].imshow(x[i].cpu().squeeze(), cmap="gray")
                axes[0, i].axis("off")
                axes[1, i].imshow(x_recon[i].cpu().squeeze(), cmap="gray")
                axes[1, i].axis("off")
            plt.suptitle(f"Epoch {epoch}: Top=Original, Bottom=Reconstructed")
            plt.show()
