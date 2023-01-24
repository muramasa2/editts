import os
import glob
import numpy as np
import matplotlib.pyplot as plt

import torch
import re
import pyopenjtalk
import subprocess
from subprocess import PIPE
import librosa
import soundfile
from audiomentations import Compose, AddGaussianNoise

julius_tmpdir = "julius"
def intersperse(lst, item):
    # Adds blank symbol
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def intersperse_emphases(emphases):
    for n in range(len(emphases)):
        emphases[n][0] = 2 * emphases[n][0]
        emphases[n][1] = 2 * emphases[n][1] + 1
    return emphases


def latest_checkpoint_path(dir_path, regex="grad_*.pt"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    x = f_list[-1]
    return x


def load_checkpoint(logdir, model, num=None):
    if num is None:
        model_path = latest_checkpoint_path(logdir, regex="grad_*.pt")
    else:
        model_path = os.path.join(logdir, f"grad_{num}.pt")
    print(f'Loading checkpoint {model_path}...')
    model_dict = torch.load(model_path, map_location=lambda loc, storage: loc)
    model.load_state_dict(model_dict, strict=False)
    return model

def writefile(name, lines):
    with open(name, "w") as f:
        f.writelines(lines)


def create_emphases(text):
    i = 0
    result = []
    emphases = []
    emphasis_interval = []
    for c in text:
        if c == "|":
            emphasis_interval.append(i)
            if len(emphasis_interval) == 2:
                emphases.append(emphasis_interval)
                emphasis_interval = []
        else:
            i += 1
            result.append(c)

    final_emphases = []
    for emphasis in emphases:
        left_tokens = len(pyopenjtalk.g2p("".join(result[: emphasis[0]])).split())
        if left_tokens != 0:
            left_tokens += 1
        right_tokens = len(pyopenjtalk.g2p("".join(result[: emphasis[1]])).split())
        emphasis_interval = [left_tokens, right_tokens]
        final_emphases.append(emphasis_interval)

    return final_emphases


def alignment_preprocess(audio_path, out_path):
    julius_sampling_rate = 16000
    augfunc = Compose([AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.001, p=1)])
    audio, _ = librosa.load(audio_path, sr=julius_sampling_rate)
    audio = augfunc(samples=audio, sample_rate=julius_sampling_rate)
    soundfile.write(out_path, audio, samplerate=julius_sampling_rate)
    return audio


def julius(text, wav):
    conv = {"I": "i", "U": "u", "v": "b", "cl": "q", "pau": "sp", "ty": "ch"}
    full = " ".join(
        [conv[s] if s in conv else s for s in pyopenjtalk.g2p(text).split()]
    )
    words = ["silB", full, "silE"]
    num = len(words) - 1  # last index of words
    lines = [
        "{} {} {} 0 {}\n".format(i, num - i, i + 1, 1 if i == 0 else 0)
        for i in range(num + 1)
    ] + ["{} -1 -1 1 0\n".format(num + 1)]
    os.makedirs(julius_tmpdir, exist_ok=True)
    phnfname = "".join(full.split())[:30]
    writefile(julius_tmpdir + phnfname + ".dfa", lines)
    writefile(
        julius_tmpdir + phnfname + ".dict",
        ["{} [w_{}] {}\n".format(i, i, words[i]) for i in range(num + 1)],
    )
    proc = subprocess.Popen(
        f"julius -nostrip -norealtime -palign -input rawfile -h segmentation-kit/models/hmmdefs_monof_mix16_gid.binhmm -fshift 200"
        " -gram " + julius_tmpdir + phnfname,
        shell=True,
        stdin=PIPE,
        stdout=PIPE,
        stderr=PIPE,
        text=True,
    )
    out = proc.communicate(str(wav))[0]
    subprocess.run(["rm", julius_tmpdir + phnfname + ".dfa"])
    subprocess.run(["rm", julius_tmpdir + phnfname + ".dict"])
    sw = 0
    durations = []
    outlines = out.split("\n")

    for line in outlines:
        if line == "=== begin forced alignment ===":
            sw = 1
        if sw == 1 and line[:1] == "[":
            line = line.replace("[", "[ ").replace("]", " ]").split()
            durations.append(int(line[2]) - int(line[1]) + 1)
        if line == "=== end forced alignment ===\n":
            sw = 0
    return durations, " ".join(words).split()