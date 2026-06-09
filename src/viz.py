import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_generated_images(epoch, generator, latent_dim, out_dir="figures", examples=16):
    """
    Generates a grid of sample images from the generator and saves to disk.

    Args:
        epoch (int): Current epoch number; used in the output filename.
        generator: Trained Keras generator model.
        latent_dim (int): Dimensionality of the latent noise vector.
        out_dir (str): Directory to write the image grid.
        examples (int): Number of images to generate (must be a perfect square).
    """
    os.makedirs(out_dir, exist_ok=True)
    noise = np.random.normal(0, 1, (examples, latent_dim))
    gen_imgs = generator.predict(noise, verbose=0)

    rows = cols = int(np.sqrt(examples))
    plt.figure(figsize=(rows, cols))
    for i in range(examples):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(gen_imgs[i, :, :, 0], cmap="gray", vmin=0, vmax=1)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"mnist_epoch_{epoch}.png"))
    plt.close()


def plot_losses(d_losses, g_losses, out_path="figures/gan_losses.png"):
    """
    Plots discriminator and generator loss curves and saves to disk.

    Args:
        d_losses (list[float]): Per-epoch discriminator losses.
        g_losses (list[float]): Per-epoch generator losses.
        out_path (str): Output file path for the loss plot.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.figure()
    plt.plot(d_losses, label="Discriminator loss")
    plt.plot(g_losses, label="Generator (adversarial) loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GAN Training on MNIST")
    plt.legend()
    plt.savefig(out_path)
    plt.close()
    print(f"Loss curves saved to {out_path}")
