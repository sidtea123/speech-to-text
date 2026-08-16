import torch
from asr import *
from data_parser import *
import random

model = STTModel()
model.load_state_dict(torch.load(MODEL_URL, weights_only=True))
model.eval()

audios, transcripts = read_files()

length = transcripts.shape[0]
num_tries = 1

for i in range(num_tries):
    index = random.randint(0, length - 1)
    print(transcripts[index])
    print(torch.argmax(model(audios[index].unsqueeze(0)), dim=-1))
