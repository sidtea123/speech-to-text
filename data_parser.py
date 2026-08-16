import numpy as np
import soundfile
import torch
import torch.nn as nn
import torchaudio
import csv
import scipy
from torch.utils.data import DataLoader, Dataset

CSV_PATH = 'speech_data/metadata.csv'
AUDIO_PATH = 'speech_data/wavs/'
OUTPUT_AUDIO_PATH = 'speech_data/audio_tensor.pt'
OUTPUT_TRANSCRIPTS_PATH = 'speech_data/transcript_tensor.pt'
NOISE_PATH = 'speech_data/background_noise.wav'
SAMPLE_RATE = 16000
TARGET_LENGTH = 10 * SAMPLE_RATE
NUM_FILES = 13100

class STTDataset(Dataset):
    def __init__(self, audios, labels):
        self.data, self.labels = audios, labels
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def generate_dataset_and_loader(images, labels, batch_size):
    dataset = STTDataset(images, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

def read_files():
    return torch.load(OUTPUT_AUDIO_PATH), torch.load(OUTPUT_TRANSCRIPTS_PATH)


encoding = {
    ' ': 0, '\'': 27,
    'a': 1,  'b': 2,  'c': 3,  'd': 4,  'e': 5, 
    'f': 6,  'g': 7,  'h': 8,  'i': 9,  'j': 10, 'k': 11, 
    'l': 12, 'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17, 
    'r': 18, 's': 19, 't': 20, 'u': 21, 'v': 22, 
    'w': 23, 'x': 24, 'y': 25, 'z': 26
}

decoding = { value: key for key, value in encoding.items() }

def greedy_decode(values):
    output = ''
    for val in values:
        if val in decoding:
            output += decoding[val]

    return output

def encode_string(str):
    output = []
    for char in str.lower():
        if char in encoding:
            output.append(encoding[char])

    return output


def get_and_save_files():
    data_tensors = []
    transcriptions = []

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE,
                                                           n_mels=64)

    with open(CSV_PATH, mode='r', newline='', encoding='utf-8') as file:
        num_rows = 0
        reader = csv.reader(file, delimiter='|', quotechar=None)

        for row in reader:
            if (len(row) < 3):
                print(row)
                print(num_rows + 1)
                continue
            transcription = row[2]
            file_path = AUDIO_PATH + row[0] + '.wav'
            data, sr = soundfile.read(file_path)
            info = soundfile.info(file_path)

            # resamples audio
            if info.samplerate != SAMPLE_RATE:
                data = scipy.signal.resample(data, int(len(data) * SAMPLE_RATE / sr))

            # tiles noise audio after audio finishes so all 10 seconds long precisely
            if len(data) < TARGET_LENGTH:
                noise_data, _ = soundfile.read(NOISE_PATH)
                data = np.concatenate((data, noise_data[np.arange(TARGET_LENGTH - len(data)) % len(noise_data)]))
            # chops off if too long
            elif len(data) > TARGET_LENGTH:
                data = data[:TARGET_LENGTH]

            data_tensor = torch.unsqueeze(mel_spectrogram(torch.tensor(data, dtype=torch.float32)), dim=0)
            data_tensors.append(data_tensor)
            transcriptions.append(torch.tensor(encode_string(transcription)))
            num_rows += 1

            if num_rows % 100 == 0:
                print(f'finished {num_rows} / {NUM_FILES}')

    output_audio = torch.stack(data_tensors)
    output_transcriptions = nn.utils.rnn.pad_sequence(transcriptions, batch_first=True, padding_value=0)
    print(output_audio.shape)
    print(output_transcriptions.shape)
    print(f'now saving tensors to file')
    torch.save(output_audio, OUTPUT_AUDIO_PATH)
    torch.save(output_transcriptions, OUTPUT_TRANSCRIPTS_PATH)

if __name__ == '__main__':
    get_and_save_files()