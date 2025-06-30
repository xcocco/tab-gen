import math
import os

import keras.utils
import numpy as np


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

        self.spectrograms = []
        self.labels = []
        self._load_data()

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
                self.spectrograms.append(spectrogram)
                self.labels.append(label)

    def __len__(self):
        data_size = 0
        for item in self.spectrograms:
            data_size += len(item)
        return math.ceil(data_size / self.batch_size)

    def __getitem__(self, idx):
        low = idx * self.batch_size
        high = min(low + self.batch_size, len(self.filenames_list))
        batch_x = self.filenames_list[low:high]
        batch_y = self.labels_list[low:high]