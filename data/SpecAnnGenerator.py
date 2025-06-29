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
        audio_filenames = []
        for item in os.listdir(self.audio_path):
            if not item.endswith(".wav"):
                continue
            item_path = os.path.join(self.audio_path, item)
            if os.path.isfile(item_path):
                audio_filenames.append(item)

        annotations_filenames = []
        for item in os.listdir(self.annotations_path):
            if not item.endswith(".jams"):
                continue
            item_path = os.path.join(self.annotations_path, item)
            if os.path.isfile(item_path):
                annotations_filenames.append(item)

        if len(audio_filenames) != len(annotations_filenames):
            print("Audio files and annotations files quantity mismatch, aborted")
            return

        # if we are here audio filenames and annotations filenames could be used interchangeably
        for i in range(len(audio_filenames)):
            audio_filename = audio_filenames[i]
            annotations_filename = annotations_filenames[i]

            y, sr = librosa.load(audio_filename)
            spectrogram = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_fft=2048,
                hop_length=512
            )
            times = librosa.frames_to_time(
                range(len(spectrogram)),
                sr=sr,
                hop_length=512
            )


            annotations = jams.load(annotations_filename)



def main():
    generator = SpecAnnGenerator("GuitarSet/audio/audio_mic", "GuitarSet/annotation")
    generator.generate()

if __name__ == "__main__":
    main()