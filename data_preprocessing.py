import os
import numpy as np
import csv
import random

import librosa
from sklearn.utils import shuffle

from utility_functions import *


def get_audio_mfccs(wave, sample_rate):

    full_length = 45 * sample_rate
    crop_length = 36 * sample_rate

    sr_ms = sample_rate / 1000
    win_length = int(30 * sr_ms)

    diff_length = full_length - crop_length

    crop_start = np.random.randint(diff_length, size=1)[0]
    crop_end = crop_start + crop_length
    sample = wave[crop_start:crop_end]

    sample_mfcc = librosa.feature.mfcc(sample, sr=sample_rate, n_mfcc=20,
                                       n_fft=win_length, hop_length=win_length)

    return sample_mfcc


class DataPreprocessor:
    def __init__(self, args):

        self._data_dir = args.data_dir
        self._deam_dir = args.deam_dir
        self._audio_dir = os.path.join(self._deam_dir, 'Audio')
        self._annotations_path = os.path.join(self._deam_dir, 'static_annotations.csv')

        self._waves_dir = os.path.join(self._deam_dir, 'Waveforms')
        if not os.path.exists(self._waves_dir):
            os.mkdir(self._waves_dir)

        self._audio_extension = args.audio_extension
        self._sample_rate = args.sample_rate

        self.audio_names = []
        self.annotations = []
        self.quadrants = []

        self.train_audio_names = []
        self.train_annotations = []
        self.test_audio_names = []
        self.test_annotations = []

        self.train_mfccs = []
        self.test_mfccs = []

    def get_data_info(self):

        initial_range = (1, 9)

        with open(self._annotations_path, newline='') as infile:
            reader = csv.reader(infile)
            header = next(reader)

            # File structure: song_id, valence_mean, valence_std, arousal_mean, arousal_std
            for line in reader:
                self.audio_names.append(line[0])

                initial_valence, initial_arousal = float(line[1]), float(line[3])
                scaled_valence = scale_measurement(initial_valence, initial_range)
                scaled_arousal = scale_measurement(initial_arousal, initial_range)
                measurements = [scaled_valence, scaled_arousal]
                self.annotations.append(measurements)

                self.quadrants.append(get_quadrant(measurements))

        self.audio_names = np.array(self.audio_names)
        self.annotations = np.array(self.annotations)
        self.quadrants = np.array(self.quadrants)

        print('\nDEAM Dataset : {:d} samples'.format(len(self.audio_names)))
        for quadrant in range(4):
            quadrant_count = np.sum(self.quadrants == quadrant + 1)
            print('  Quadrant {:d} : {:d} samples'.format(quadrant + 1, quadrant_count))

    def get_waveforms(self):

        sr_ms = self._sample_rate / 1000
        for idx, audio_name in enumerate(self.audio_names):

            audio_path = os.path.join(self._audio_dir, '{:s}.{:s}'.format(audio_name, self._audio_extension))
            wave, _ = librosa.load(audio_path, self._sample_rate)

            duration = len(wave) / sr_ms
            if duration < 45000:
                diff = int((duration - 45000) * sr_ms)
                wave = np.concatenate([wave, wave[diff:]])
            else:
                wave = wave[:45*self._sample_rate]

            wave_path = os.path.join(self._waves_dir, '{:s}.npy'.format(audio_name))
            np.save(wave_path, wave)

    def augment_quadrants(self):

        desired_size = 500
        quadrant_names = [1, 2, 3, 4]

        for q_name in quadrant_names:

            q_idxs = np.where(self.quadrants == q_name)[0]
            q_size = len(q_idxs)
            print('\nQUADRANT {:d} : {:>4d} samples'.format(q_name, q_size))

            if q_size >= desired_size:
                q_augmented_idxs = q_idxs[np.array(random.sample(range(q_size), desired_size))]
                q_audio_names = self.audio_names[q_augmented_idxs]
                q_annotations = self.annotations[q_augmented_idxs]

                print('    Choosing {:>4d} samples'.format(desired_size))
                print('   Resulting {:>4d} samples'.format(len(q_audio_names)))

            else:
                augm_size = desired_size - q_size
                q_augmented_idxs = q_idxs[np.random.randint(q_size, size=augm_size)]
                q_audio_names = np.concatenate([self.audio_names[q_idxs], self.audio_names[q_augmented_idxs]])
                q_annotations = np.concatenate([self.annotations[q_idxs], self.annotations[q_augmented_idxs]])

                print('     Keeping {:>4d} samples'.format(q_size))
                print('    Choosing {:>4d} samples'.format(augm_size))
                print('   Resulting {:>4d} samples'.format(len(q_audio_names)))

            self.train_audio_names.extend(list(q_audio_names)[:400])
            self.train_annotations.extend(list(q_annotations)[:400])
            self.test_audio_names.extend(list(q_audio_names)[400:])
            self.test_annotations.extend(list(q_annotations)[400:])

        self.train_audio_names, self.train_annotations = shuffle(self.train_audio_names, self.train_annotations)
        self.test_audio_names, self.test_annotations = shuffle(self.test_audio_names, self.test_annotations)

        np.save(os.path.join(self._data_dir, 'train_audio_names.npy'), self.train_audio_names)
        np.save(os.path.join(self._data_dir, 'train_annotations.npy'), self.train_annotations)
        np.save(os.path.join(self._data_dir, 'test_audio_names.npy'), self.test_audio_names)
        np.save(os.path.join(self._data_dir, 'test_annotations.npy'), self.test_annotations)

    def make_train_test_sets(self):

        for idx, audio_name in enumerate(self.train_audio_names):

            wave_path = os.path.join(self._waves_dir, '{:s}.npy'.format(audio_name))
            wave = librosa.load(wave_path, self._sample_rate)

            mfcc = get_audio_mfccs(wave, self._sample_rate)
            self.train_mfccs.append(mfcc)

        for idx, audio_name in enumerate(self.test_audio_names):
            wave_path = os.path.join(self._waves_dir, '{:s}.npy'.format(audio_name))
            wave = librosa.load(wave_path, self._sample_rate)

            mfcc = get_audio_mfccs(wave, self._sample_rate)
            self.test_mfccs.append(mfcc)

        np.save(os.path.join(self._data_dir, 'train_mfccs.npy'), self.train_mfccs)
        np.save(os.path.join(self._data_dir, 'test_mfccs.npy'), self.test_mfccs)
