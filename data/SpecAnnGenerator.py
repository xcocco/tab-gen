import os

import keras.utils
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

        # standard tuning
        self.strings_pitches = [40, 45, 50, 55, 59, 64]
        self.strings_num = 6
        self.frets_num = 19

        # open string + not played + 19 frets
        self.num_classes = self.frets_num + 2

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
            labels = np.empty(labels_shape, dtype=int)

            annotations_file = jams.load(annotations_filename)
            for string_index, anno in enumerate(annotations_file.annotations['note_midi']):
                notes_per_string = anno.to_samples(times)
                notes_per_string_flat = np.empty((len(notes_per_string)), dtype=int)
                for j in range(len(notes_per_string)):
                    note = notes_per_string[j]
                    if not note:
                        notes_per_string_flat[j] = 0
                    else:
                        notes_per_string_flat[j] = int(round(note[0]) - self.strings_pitches[string_index] + 1)
                        if notes_per_string_flat[j] < 0 or notes_per_string_flat[j] > self.num_classes - 1:
                            notes_per_string_flat[j] = 0
                labels[string_index, :] = notes_per_string_flat

            # swap axes to get 6 columns (one column per string) and as many rows as frames in the audio
            labels = labels.T
            labels = keras.utils.to_categorical(labels, self.num_classes)

            output_path = 'spec_ann'
            audio_filename = os.path.basename(audio_filename)
            np.savez(
                os.path.join(output_path, str(audio_filename).replace('.wav', '.npz')),
                spectrogram=data,
                labels=labels,
            )

    def __preprocess_data(self, data):
        # apply some preprocess to data
        data = data.astype(float)
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
    filenames.sort()
    return filenames

def main():
    generator = SpecAnnGenerator("GuitarSet/audio/audio_mic", "GuitarSet/annotation")
    generator.generate()

if __name__ == "__main__":
    main()