import numpy as np
from tensorflow.keras.datasets import mnist


def load_mnist():
    """
    Loads and preprocesses the MNIST training set.

    Returns:
        x_train (np.ndarray): shape (60000, 28, 28, 1), values in [0, 1].
    """
    (x_train, _), (_, _) = mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, axis=-1)
    return x_train


def sample_real_batch(x_train, batch_size):
    """Randomly samples a batch of real images."""
    idx = np.random.randint(0, x_train.shape[0], batch_size)
    return x_train[idx]


def sample_noise(batch_size, latent_dim):
    """Samples a batch of random latent vectors from N(0, 1)."""
    return np.random.normal(0, 1, (batch_size, latent_dim)).astype(np.float32)
