# ============================
# CONFIG
# ============================
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

LATENT_DIM = 64
IMAGE_SIZE = 28 * 28
BATCH_SIZE = 128
EPOCHS = 10
LR = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Using device:", device)

# ============================
# STEP 1: Data Preprocessing
# ============================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # [0,1] -> [-1,1]
])

train_data = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
test_data  = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")

# ============================
# STEP 2: Autoencoder Model
# ============================
class Autoencoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(IMAGE_SIZE, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, IMAGE_SIZE),
            nn.Tanh(),
            nn.Unflatten(1, (1, 28, 28))
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

# ============================
# STEP 3: Setup Training
# ============================
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ============================
# STEP 4: Training Loop
# ============================
train_losses = []
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for imgs, _ in train_loader:
        imgs = imgs.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, imgs)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f}")

# ============================
# STEP 5: Plot Training Loss
# ============================
plt.plot(range(1, EPOCHS+1), train_losses, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Autoencoder Training Loss (MSE)")
plt.show()

# ============================
# STEP 6: Visualize Reconstructions
# ============================
model.eval()
with torch.no_grad():
    imgs, _ = next(iter(test_loader))
    imgs = imgs.to(device)
    outputs = model(imgs)

    # de-normalize [-1,1] -> [0,1]
    imgs_vis = (imgs.cpu() * 0.5) + 0.5
    outputs_vis = (outputs.cpu() * 0.5) + 0.5

    fig, axes = plt.subplots(2, 10, figsize=(15, 3))
    for i in range(10):
        axes[0, i].imshow(imgs_vis[i].squeeze(), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(outputs_vis[i].squeeze(), cmap="gray")
        axes[1, i].axis("off")
    plt.suptitle("Top: Original | Bottom: Reconstructed", fontsize=14)
    plt.show()
