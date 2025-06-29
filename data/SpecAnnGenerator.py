import os

import librosa
import librosa.feature
import jams

class SpecAnnGenerator:
    def __init__(self, audio_path, annotations_path):
        self.audio_path = audio_path
        self.annotations_path = annotations_path

        self.strings_num = 6
        self.frets_num = 21

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
            y, sr = librosa.load(audio_filename)
            y = _preprocess_data(y)
            spectrogram = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_fft=2048,
                hop_length=512
            )
            spectrogram = spectrogram.T
            times = librosa.frames_to_time(
                range(len(spectrogram)),
                sr=sr,
                hop_length=512
            )

            annotations_file = jams.load(annotations_filename)
            for anno in annotations_file.annotations['note_midi']:
                label = anno.to_samples(times)
                print(label)

def _get_filenames_from_path(path, extension= ""):
    filenames = []
    for item in os.listdir(path):
        if item.endswith(extension):
            item_path = os.path.join(path, item)
            if os.path.isfile(item_path):
                filenames.append(item_path)

    return filenames

def _preprocess_data(data):
    # apply some preprocess to data
    data = librosa.util.normalize(data)
    return data

def main():
    generator = SpecAnnGenerator("GuitarSet/audio/audio_mic", "GuitarSet/annotation")
    generator.generate()

if __name__ == "__main__":
    main()