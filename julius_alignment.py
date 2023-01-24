import argparse
import json
import math
import os
import re
import sys
import tempfile

import librosa
import matplotlib.pyplot as plt
import numpy as np
import params
import torch
import torchaudio as ta
from model import GradTTS
from model.utils import fix_len_compatibility
from scipy.io.wavfile import write
from text import ja_text_to_sequence, ja_text_to_sequence_for_editts
from text.symbols import symbols
from utils import (
    alignment_preprocess,
    create_emphases,
    intersperse,
    intersperse_emphases,
    julius,
)

sys.path.append("./hifigan/")
from env import AttrDict
from meldataset import mel_spectrogram
from models import Generator as HiFiGAN

torch.manual_seed(1234)

speech_file = "/data/editts/real_audio/BASIC5000_0006.wav"
text1 = "週に四回、フランスの授業があります"

audio, _ = librosa.load(speech_file, sr=16000)
julius_sf = 16000
with tempfile.TemporaryDirectory() as tmpdir:
    tmpfile = os.path.join(tmpdir, "temp.wav")
    # ta.save(tmpfile, audio, SAMPLE_RATE, encoding="PCM_S", bits_per_sample=16)
    alignment_preprocess(speech_file, tmpfile)
    duration, phn = julius(text1, tmpfile)
    pos = librosa.frames_to_samples(np.cumsum(duration), hop_length=200)
plt.figure(figsize=(12, 9))
plt.plot(audio)
for d, p in zip(pos, phn):
    plt.vlines(d, -1, 1, color="g")
    plt.text(d - 500, 1, p, fontsize=14)
    print(d, p)
# plt.show()
plt.savefig("julius_alignment.png")
