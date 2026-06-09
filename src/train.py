import numpy as np
from src.data import sample_real_batch, sample_noise
from src.viz import save_generated_images


def train_gan(
    generator,
    discriminator,
    combined,
    x_train,
    latent_dim,
    epochs=600,
    batch_size=128,
    save_interval=None,
    figures_dir="figures",
):
    """
    Adversarial training loop.

    Alternates between:
      1. Training the discriminator on real and generated batches.
      2. Freezing the discriminator and training the generator via the
         combined model.

    Args:
        generator: Keras generator model.
        discriminator: Compiled Keras discriminator model.
        combined: Combined (generator + frozen discriminator) model.
        x_train (np.ndarray): Preprocessed MNIST images, shape (N, 28, 28, 1).
        latent_dim (int): Latent vector dimensionality.
        epochs (int): Number of training epochs.
        batch_size (int): Mini-batch size.
        save_interval (list[int]): Epochs at which to save generated image grids.
        figures_dir (str): Directory for saving image checkpoints.

    Returns:
        d_losses (list[float]): Per-epoch mean discriminator loss.
        g_losses (list[float]): Per-epoch mean generator loss.
    """
    if save_interval is None:
        save_interval = [1, 50, 100, 200, 400, 600]

    num_samples = x_train.shape[0]
    batches_per_epoch = num_samples // batch_size

    valid_label = np.ones((batch_size, 1), dtype=np.float32)
    fake_label = np.zeros((batch_size, 1), dtype=np.float32)
    valid_for_g = np.ones((batch_size, 1), dtype=np.float32)

    d_losses, g_losses = [], []

    print(f"Training: {epochs} epochs | {batches_per_epoch} batches/epoch | batch size {batch_size}")
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        epoch_d, epoch_g = [], []

        for batch in range(batches_per_epoch):
            # --- Discriminator step ---
            discriminator.trainable = True
            real_imgs = sample_real_batch(x_train, batch_size)
            noise = sample_noise(batch_size, latent_dim)
            fake_imgs = generator(noise, training=False)

            d_loss_real = discriminator.train_on_batch(real_imgs, valid_label)
            d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_label)
            d_loss = 0.5 * (d_loss_real[0] + d_loss_fake[0])

            # --- Generator step ---
            discriminator.trainable = False
            noise = sample_noise(batch_size, latent_dim)
            g_loss = combined.train_on_batch(noise, valid_for_g)

            epoch_d.append(d_loss)
            epoch_g.append(g_loss)

            if batch % 50 == 0:
                print(
                    f"Epoch {epoch}/{epochs} | Batch {batch}/{batches_per_epoch} "
                    f"| D loss: {d_loss:.4f} | G loss: {g_loss:.4f}",
                    end="\r",
                )

        d_losses.append(float(np.mean(epoch_d)))
        g_losses.append(float(np.mean(epoch_g)))
        print(
            f">>> Epoch {epoch}/{epochs} COMPLETE "
            f"| Avg D loss: {d_losses[-1]:.4f} | Avg G loss: {g_losses[-1]:.4f}"
        )
        print("=" * 80)

        if epoch in save_interval:
            save_generated_images(epoch, generator, latent_dim, out_dir=figures_dir)
            print(f"*** Saved images for epoch {epoch} ***\n")

    return d_losses, g_losses
