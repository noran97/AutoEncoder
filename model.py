
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# Removed visualize_batch import as we are defining the plotting function internally
from data import get_data
import matplotlib.pyplot as plt

# Define the dimensions and parameters (matching the user's simple encoder)
LATENT_DIM = 64
IMAGE_SIZE = 28 * 28
EPOCHS = 10
BATCH_SIZE = 128


def plot_reconstructions(images_tensor, title="Reconstruction Results"):
    """
    Displays original and reconstructed images from a concatenated tensor.
    The tensor is expected to contain 4 original images followed by 4 reconstructed images.
    """
    # Assuming we pass 8 images total (4 original, 4 reconstructed)
    num_images = images_tensor.shape[0]

    # Calculate rows (2 rows: Original and Reconstructed) and 4 columns
    num_rows = 2
    num_cols = 4

    plt.figure(figsize=(12, 5))
    plt.suptitle(title, fontsize=14)

    for i in range(num_images):
        plt.subplot(num_rows, num_cols, i + 1)

        # Denormalize: Images are in [-1, 1] range, convert back to [0, 1] for display
        img = (images_tensor[i] * 0.5) + 0.5

        # Display the image, converting PyTorch tensor to NumPy array
        plt.imshow(img.squeeze().numpy(), cmap="gray")

        # Set titles based on position (0-3 are originals, 4-7 are reconstructions)
        if i < num_cols:
            plt.title("Original", fontsize=10)
        else:
            plt.title("Reconstructed", fontsize=10)

        plt.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for suptitle
    plt.show()


class Encoder(nn.Module):
    """
    Encoder part of the Autoencoder, provided by the user.
    It uses only fully connected layers.
    (1, 28, 28) -> Flatten (784) -> Linear (256) -> ReLU -> Linear (64)
    """

    def __init__(self, latent_dim=LATENT_DIM):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),  # تحويل الصورة إلى vector
            nn.Linear(IMAGE_SIZE, 256),  # طبقة fully connected
            nn.ReLU(),  # تفعيل ReLU
            nn.Linear(256, latent_dim)  # إخراج latent vector
        )

    def forward(self, x):
        z = self.encoder(x)
        return z


class Decoder(nn.Module):
    """
    Symmetric Decoder to reconstruct the image from the latent vector.
    It reverses the fully connected layers of the Encoder.
    (64) -> Linear (256) -> ReLU -> Linear (784) -> Unflatten (1, 28, 28)
    """

    def __init__(self, latent_dim=LATENT_DIM):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, IMAGE_SIZE),
            # Tanh is used because the input data was normalized to the range [-1, 1]
            nn.Tanh(),
            nn.Unflatten(1, (1, 28, 28))  # Unflatten back to image dimensions (C, H, W)
        )

    def forward(self, z):
        x_reconstructed = self.decoder(z)
        return x_reconstructed


class Autoencoder(nn.Module):
    """
    Combines the Encoder and Decoder.
    """

    def __init__(self, latent_dim=LATENT_DIM):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed


def train_model():
    """
    Sets up the training environment, loads data, and runs the training loop.
    """
    # 1. Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Loading (using the function from data.py)
    train_loader, test_loader = get_data(batch_size=BATCH_SIZE)
    print(f"Data loaders created with batch size {BATCH_SIZE}.")

    # 3. Model, Loss, and Optimizer Setup
    model = Autoencoder(latent_dim=LATENT_DIM).to(device)
    # Mean Squared Error (MSE) is standard for image reconstruction loss
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("\n--- Model Summary ---")
    print(model)

    # 4. Training Loop
    print(f"\nStarting training for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for batch_idx, (data, _) in enumerate(train_loader):
            # The input data is the target output (x_in = x_out)
            data = data.to(device)

            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, data)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch}/{EPOCHS}], Loss: {avg_loss:.6f}")

        # 5. Simple Evaluation/Visualization
        if epoch % 5 == 0:
            print("Visualizing reconstructions...")
            model.eval()
            with torch.no_grad():
                # Get a batch from the test loader
                # Note: We reset the test_loader iterator here to grab a fresh batch
                images, _ = next(iter(test_loader))
                images = images.to(device)

                # Reconstruct the images
                reconstructed = model(images).cpu()

                # Stack original and reconstructed images for visualization (4 pairs)
                original_and_reconstructed = torch.cat(
                    [images[:4].cpu(), reconstructed[:4]], dim=0
                )

                # Display the results using the internal plotting function
                plot_reconstructions(
                    original_and_reconstructed,
                    title=f"Epoch {epoch} Reconstruction (Top 4: Original, Bottom 4: Reconstructed)"
                )


if __name__ == '__main__':
    # Add a title argument to visualize_batch for better plotting (assuming it's in data.py)
    # You might need to add 'title=None' as a default argument to visualize_batch in data.py
    # if it doesn't support it already.
    train_model()
=======
# model.py
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, latent_dim=64):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),              # تحويل الصورة إلى vector
            nn.Linear(28*28, 256),     # طبقة fully connected
            nn.ReLU(),                 # تفعيل ReLU
            nn.Linear(256, latent_dim) # إخراج latent vector
        )
    
    def forward(self, x):
        z = self.encoder(x)
        return z

