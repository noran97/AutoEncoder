import numpy as np
import matplotlib.pyplot as plt
import sys

# Force the system's standard output to use UTF-8
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
# Import the specific dataset loader directly from the standard source (e.g., Keras/TensorFlow)
# Assuming you are using Keras/TensorFlow, as datasets.fashion_mnist.load_data() is common there.
from tensorflow.keras import datasets


def load_and_preprocess_data():
    """
    Loads the Fashion MNIST dataset, normalizes, pads, and reshapes it
    for use in a convolutional autoencoder/encoder model.

    Returns:
        tuple: (x_train, x_test) the preprocessed training and testing images.
    """
    # Load the data
    print("Loading Fashion MNIST data...")
    (x_train, _), (x_test, _) = datasets.fashion_mnist.load_data()

    # Note: Labels (y_train, y_test) are typically not needed for autoencoders/unsupervised encoders.

    def preprocess(imgs):
        """
        Normalize, pad, and reshape the images for a CNN input.
        Input shape: (N, 28, 28)
        Output shape: (N, 32, 32, 1)
        """
        # 1. Normalize: Convert to float32 and scale to [0, 1]
        imgs = imgs.astype("float32") / 255.0

        # 2. Padding: Pad 2 pixels on all sides to change (28x28) to (32x32)
        # This is often done to allow for specific convolutional layers (e.g., stride 2)
        # and to match common input sizes.
        imgs = np.pad(imgs, ((0, 0), (2, 2), (2, 2)), constant_values=0.0)

        # 3. Reshape: Add the channel dimension (1 for grayscale)
        # Shape goes from (N, 32, 32) to (N, 32, 32, 1)
        imgs = np.expand_dims(imgs, -1)

        return imgs

    print("Preprocessing data...")
    x_train = preprocess(x_train)
    x_test = preprocess(x_test)
    print(f"Train set shape: {x_train.shape}")
    print(f"Test set shape: {x_test.shape}")
    print("Preprocessing complete.")

    return x_train, x_test


def display_images(
        images, n=10, size=(20, 3), cmap="gray_r", as_type="float32", save_to=None, title="Sample Images"
):
    """
    Displays n random images from the supplied array. Handles normalization for display.

    Args:
        images (np.ndarray): The array of images (e.g., (N, H, W, C) or (N, H, W)).
        n (int): Number of images to display.
        size (tuple): Size of the matplotlib figure.
        cmap (str): Colormap to use (e.g., 'gray' or 'gray_r').
        as_type (str): Datatype for plotting (e.g., 'float32').
        save_to (str, optional): File path to save the plot. Defaults to None.
        title (str): Title for the figure.
    """
    if images is None or len(images) == 0:
        print("Error: No images supplied for display.")
        return

    # Select up to n random indices, but not more than the total number of images
    num_images = min(n, len(images))
    indices = np.random.choice(len(images), num_images, replace=False)

    # Prepare images for display:
    display_images_array = images[indices]

    # Handle channel dimension: remove if it's 1 (e.g., (N, 32, 32, 1) -> (N, 32, 32))
    if display_images_array.ndim == 4 and display_images_array.shape[-1] == 1:
        display_images_array = display_images_array.squeeze(axis=-1)

    # Normalization/Denormalization for display
    # Assumes images are in [0, 1] or occasionally [-1, 1]
    if display_images_array.max() > 1.0:
        # Scale down if values are, e.g., [0, 255]
        display_images_array = display_images_array / 255.0
    elif display_images_array.min() < 0.0:
        # Rescale if values are, e.g., [-1, 1]
        display_images_array = (display_images_array + 1.0) / 2.0

    plt.figure(figsize=size)
    plt.suptitle(title, fontsize=16)  # Add a figure title
    for i in range(num_images):
        # Use num_images instead of n to ensure correct subplot indexing
        _ = plt.subplot(1, num_images, i + 1)
        # Use the pre-selected array
        plt.imshow(display_images_array[i].astype(as_type), cmap=cmap)
        plt.axis("off")

    if save_to:
        plt.savefig(save_to)
        print(f"\nSaved to {save_to}")

    plt.show()


# --- Execution Block ---
# Only run data loading and preprocessing if the script is executed directly
if __name__ == "__main__":
    # Load and preprocess the data
    X_train, X_test = load_and_preprocess_data()

    # Display a sample of the training data
    display_images(X_train, title="Sample of Preprocessed Fashion MNIST Images")

    # Display a sample of the test data (optional)
    # display_images(X_test, title="Sample of Test Fashion MNIST Images")