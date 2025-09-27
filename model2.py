import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import get_data
import matplotlib.pyplot as plt

LATENT_DIM = 64
IMAGE_SIZE = 28 * 28
EPOCHS = 10
BATCH_SIZE = 128


def plot_reconstructions(images_tensor, title="Reconstruction Results"):
    num_images = images_tensor.shape[0]
    num_rows, num_cols = 2, 4
    plt.figure(figsize=(12, 5))
    plt.suptitle(title, fontsize=14)
    for i in range(num_images):
        plt.subplot(num_rows, num_cols, i + 1)
        img = (images_tensor[i] * 0.5) + 0.5
        plt.imshow(img.squeeze().numpy(), cmap="gray")
        plt.axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# ------------------ Variational Autoencoder ------------------

class VAE_Encoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super(VAE_Encoder, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(IMAGE_SIZE, 256)
        self.relu = nn.ReLU()
        # بدلاً من إخراج واحد → نطلع mean و logvar
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x):
        x = self.flatten(x)
        h = self.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class VAE_Decoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super(VAE_Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, IMAGE_SIZE)
        self.tanh = nn.Tanh()
        self.unflatten = nn.Unflatten(1, (1, 28, 28))

    def forward(self, z):
        h = self.relu(self.fc1(z))
        x_hat = self.tanh(self.fc2(h))
        return self.unflatten(x_hat)


class VAE(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super(VAE, self).__init__()
        self.encoder = VAE_Encoder(latent_dim)
        self.decoder = VAE_Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar


# ------------------ Loss Function ------------------

def vae_loss(recon_x, x, mu, logvar):
    # إعادة البناء (MSE)
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction="sum")
    # KL Divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss


# ------------------ Training ------------------

def train_vae():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_data(batch_size=BATCH_SIZE)
    model = VAE(latent_dim=LATENT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print(model)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0

        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = vae_loss(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        avg_loss = train_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch}/{EPOCHS}] Loss: {avg_loss:.6f}")

        if epoch % 5 ==== 0:
            model.eval()
            with torch.no_grad():
                images, _ = next(iter(test_loader))
                images = images.to(device)
                recon_images, _, _ = model(images)
                combined = torch.cat([images[:4].cpu(), recon_images[:4].cpu()], dim=0)
                plot_reconstructions(combined, title=f"Epoch {epoch} VAE Reconstructions")


if __name__ == "__main__":
    train_vae()
