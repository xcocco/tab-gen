import math
import os

import keras.utils
import numpy as np
from tensorflow.python.ops.gen_dataset_ops import concatenate_dataset


class CustomDataset(keras.utils.PyDataset):
    def __init__(
            self,
            guitarists_ids,
            data_path='../data/spec_ann',
            batch_size=128,
            con_win_size=1,
            shuffle=True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.guitarists_ids = guitarists_ids
        self.data_path = data_path
        self.batch_size = batch_size
        self.con_win_size = con_win_size
        self.half_win = con_win_size // 2
        self.shuffle = shuffle

        self.X_dim = (self.batch_size, 128, self.con_win_size, 1)
        self.y_dim = (self.batch_size, 6, 21)

        self.spectrograms = []
        self.labels = []
        self._load_data()

        self.on_epoch_end()

    def _load_data(self):
        listdir = os.listdir(self.data_path)
        listdir.sort()
        for idx in self.guitarists_ids:
            start = idx * 60
            end = start + 60
            files_to_read = listdir[start:end]
            for filename in files_to_read:
                npz = np.load(os.path.join(self.data_path, filename))
                spectrogram = npz['spectrogram']
                label = npz['labels']
                spectrogram = np.pad(spectrogram, [])
                self.spectrograms.append(spectrogram)
                self.labels.append(label)
        self.concatenated = np.concatenate(self.spectrograms, axis=0)

    def __len__(self):
        data_size = 0
        for item in self.spectrograms:
            data_size += len(item)
        return math.ceil(data_size / self.batch_size)

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        X = np.empty(self.X_dim)
        y = np.empty(self.y_dim)
        counter = 0
        for index in indexes:
            window = self.concatenated[index : index + self.con_win_size]
            X[counter,] = np.expand_dims(np.swapaxes(window, 0, 1), -1)
            y[counter,] = self.labels[index]
            counter += 1

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.concatenated))
        if self.shuffle:
            np.random.shuffle(self.indexes)

dio = CustomDataset([0])
print(dio.concatenated.shape)