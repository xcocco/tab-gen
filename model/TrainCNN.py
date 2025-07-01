import numpy as np
import tensorflow as tf

from model.CustomDataset import CustomDataset


def build_model(shape) -> tf.keras.Model:
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(shape=shape),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])
    return model

def main():
    filename = '../data/spec_ann/00_BN1-129-Eb_comp_mic.npz'
    npzfile = np.load(filename)
    spectrogram = npzfile['spectrogram']
    labels = npzfile['labels']
    print("Spectrogram shape: ", spectrogram.shape)
    print("Labels shape: ", labels.shape)

    full_x = np.pad(spectrogram, [(4, 4), (0, 0)], mode='constant')
    print("full_x shape: ", full_x.shape)
    sample_x = full_x[0: 0 + 9]
    print("smaple_x shape: ", sample_x.shape)
    X = np.expand_dims(np.swapaxes(sample_x, 0, 1), -1)
    print("X shape: ", X.shape)

if __name__ == '__main__':
    main()