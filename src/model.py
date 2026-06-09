import numpy as np
from tensorflow.keras.layers import (Input, Dense, Reshape, Flatten,
                                     Dropout, BatchNormalization,
                                     LeakyReLU, Conv2D, Conv2DTranspose)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

IMG_SHAPE = (28, 28, 1)
LATENT_DIM = 100


def build_discriminator():
    """
    Convolutional discriminator: four Conv2D blocks (LeakyReLU + Dropout),
    flattened to a sigmoid scalar validity score.
    """
    model = Sequential()

    model.add(Conv2D(32, kernel_size=5, strides=(2, 2), padding="same",
                     input_shape=IMG_SHAPE))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))

    model.add(Conv2D(64, kernel_size=5, strides=(2, 2), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))

    model.add(Conv2D(128, kernel_size=5, strides=(2, 2), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))

    model.add(Conv2D(256, kernel_size=5, strides=(2, 2), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid"))

    img = Input(shape=IMG_SHAPE)
    validity = model(img)
    return Model(img, validity)


def build_generator():
    """
    Transposed-convolution generator: projects a latent vector into a
    7x7x192 tensor, upsamples twice to 28x28, outputs a single-channel
    grayscale image in [0, 1].
    """
    model = Sequential()

    model.add(Dense(7 * 7 * 192, input_dim=LATENT_DIM))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((7, 7, 192)))

    model.add(Conv2DTranspose(96, kernel_size=5, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(48, kernel_size=5, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(24, kernel_size=5, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(1, kernel_size=5, padding="same", activation="sigmoid"))

    noise = Input(shape=(LATENT_DIM,))
    img = model(noise)
    return Model(noise, img)


def build_combined(generator, discriminator, optimizer):
    """
    Stacks generator and frozen discriminator into a single trainable model
    for generator updates.
    """
    discriminator.trainable = False
    z = Input(shape=(LATENT_DIM,))
    img = generator(z)
    validity = discriminator(img)
    combined = Model(z, validity)
    combined.compile(loss="binary_crossentropy", optimizer=optimizer)
    return combined


def compile_discriminator(discriminator, optimizer):
    discriminator.compile(
        loss="binary_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    return discriminator
