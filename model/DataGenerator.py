import math
import os

import keras
import numpy as np


class DataGenerator(keras.utils.PyDataset):
    def __init__(self,
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
        self.shuffle = shuffle
        self.half_win = con_win_size // 2
        self.indexes = None

        # init empty lists for spectrograms and labels
        self.spectrograms = []
        self.labels = []

        # init data info
        self.__load_data()

        if self.batch_size <= 0:
            for item in self.spectrograms:
                self.batch_size += len(item)
        # TODO make shape dynamic, pass from constructor
        self.X_shape = (self.batch_size, 128, self.con_win_size, 1)
        self.y_shape = (self.batch_size, 6, 21)

        self.on_epoch_end()

    def __load_data(self):
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
                spectrogram = np.pad(spectrogram, [(self.half_win, self.half_win), (0, 0)], mode='constant')
                self.spectrograms.append(spectrogram)
                self.labels.append(label)

    def __len__(self):
        data_size = self._calculate_dataset_length()
        return math.ceil(data_size / self.batch_size)

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]

        X, y = self.__data_generation(indexes)
        return X, y

    def __data_generation(self, indexes):
        X = np.empty(self.X_shape)
        y = np.empty(self.y_shape)

        for i, abs_idx in enumerate(indexes):
            spec_idx, rel_idx = self.__get_relative_index(abs_idx)
            spectrogram = self.spectrograms[spec_idx]
            sample_x = spectrogram[rel_idx : rel_idx + self.con_win_size]

            X[i,] = np.expand_dims(np.swapaxes(sample_x, 0, 1), -1)
            y[i,] = self.labels[spec_idx][rel_idx]

        return X, y

    def __get_relative_index(self, index):
        cumulative_size = 0
        for i, spectrogram in enumerate(self.spectrograms):
            spectrogram_size = len(spectrogram) - (self.half_win * 2)
            cumulative_size += spectrogram_size
            if index < cumulative_size:
                return i, index - (cumulative_size - spectrogram_size)

        return None

    def on_epoch_end(self):
        # update after each epoch
        self.indexes = np.arange(self._calculate_dataset_length())
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _calculate_dataset_length(self) -> int:
        # calculate total size of the dataset
        # (total number of frames)
        data_size = 0
        for item in self.spectrograms:
            # remove padding from the size
            data_size += len(item) - (self.half_win * 2)
        return data_size

#dio = DataGenerator([0,1,2,3,4], con_win_size=9)
#print(dio._calculate_dataset_length())