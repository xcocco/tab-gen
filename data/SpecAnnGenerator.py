import os

import librosa
import librosa.feature
import jams
import numpy as np

class SpecAnnGenerator:
    def __init__(self,
                 audio_path,
                 annotations_path,
                 n_fft=2048,
                 hop_length=512,
                 sr=22050
        ):
        self.audio_path = audio_path
        self.annotations_path = annotations_path

        self.strings_num = 6
        self.frets_num = 21

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sr = sr

    def generate(self):
        audio_filenames = _get_filenames_from_path(self.audio_path, ".wav")
        annotations_filenames = _get_filenames_from_path(self.annotations_path, ".jams")

        if len(audio_filenames) != len(annotations_filenames):
            print("Audio files and annotations files quantity mismatch, aborted")
            return

        # if we are here audio filenames and annotations filenames length could be used interchangeably
        for i in range(len(audio_filenames)):
            audio_filename = audio_filenames[i]
            annotations_filename = annotations_filenames[i]
            print(audio_filename)
            print(annotations_filename)

            y, sr = librosa.load(audio_filename, sr=self.sr)
            data = self.__preprocess_data(y)
            times = librosa.frames_to_time(
                range(len(data)),
                sr=sr,
                hop_length=self.hop_length,
            )

            labels_shape = (self.strings_num, len(times))
            labels = np.full(labels_shape, -1)

            annotations_file = jams.load(annotations_filename)
            for string_index, anno in enumerate(annotations_file.annotations['note_midi']):
                label = anno.to_samples(times)
                label_flat = np.array([l[0] if len(l) > 0 else -1 for l in label])
                labels[string_index, :] = label_flat
            print(labels)

    def __preprocess_data(self, data):
        # apply some preprocess to data
        data = librosa.util.normalize(data)
        data = librosa.feature.melspectrogram(
            y=data,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        # swap axes
        return data.T

def _get_filenames_from_path(path, extension=""):
    filenames = []
    for item in os.listdir(path):
        if item.endswith(extension):
            item_path = os.path.join(path, item)
            if os.path.isfile(item_path):
                filenames.append(item_path)

    return filenames

def main():
    generator = SpecAnnGenerator("GuitarSet/audio/audio_mic", "GuitarSet/annotation")
    generator.generate()

if __name__ == "__main__":
    main()