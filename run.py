import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow.keras.optimizers import Adam
from src import (
    build_discriminator,
    build_generator,
    build_combined,
    compile_discriminator,
    load_mnist,
    plot_losses,
)
from src.train import train_gan

# ── Hyperparameters ─────────────────────────────────────────────────────────
LATENT_DIM   = 100
BATCH_SIZE   = 128
EPOCHS       = 600
SAVE_INTERVAL = [1, 50, 100, 200, 400, 600]
FIGURES_DIR  = "figures"
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    x_train = load_mnist()
    print(f"Loaded {x_train.shape[0]} MNIST images, shape: {x_train.shape}")

    d_optimizer = Adam(0.0002, 0.5)
    g_optimizer = Adam(0.0002, 0.5)

    discriminator = build_discriminator()
    discriminator = compile_discriminator(discriminator, d_optimizer)

    generator = build_generator()

    combined = build_combined(generator, discriminator, g_optimizer)

    d_losses, g_losses = train_gan(
        generator=generator,
        discriminator=discriminator,
        combined=combined,
        x_train=x_train,
        latent_dim=LATENT_DIM,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        save_interval=SAVE_INTERVAL,
        figures_dir=FIGURES_DIR,
    )

    plot_losses(d_losses, g_losses, out_path=f"{FIGURES_DIR}/gan_losses.png")
    print("Training complete.")
