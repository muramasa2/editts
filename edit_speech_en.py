import json
import math
import torch
import params
import sys, os
import librosa
import argparse
import numpy as np
import torchaudio as ta

from model import GradTTS
from text.symbols import symbols
from scipy.io.wavfile import write
from model.utils import fix_len_compatibility
from utils import intersperse, intersperse_emphases
from text import text_to_sequence_for_editts, cmudict

sys.path.append("./hifigan/")
from env import AttrDict
from meldataset import mel_spectrogram
from models import Generator as HiFiGAN

torch.manual_seed(1234)

HIFIGAN_CONFIG = "./checkpts/hifigan-config.json"
HIFIGAN_CHECKPT = "./checkpts/hifigan.pt"
VOLUME_MAX = 32768
SAMPLE_RATE = 22050

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        required=True,
        help="path to a file with texts to synthesize",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        required=True,
        help="path to a checkpoint of Grad-TTS",
    )
    parser.add_argument(
        "-t",
        "--timesteps",
        type=int,
        default=1000,
        help="number of timesteps of reverse diffusion",
    )
    parser.add_argument(
        "-s",
        "--save_dir",
        type=str,
        default="out/content/wavs",
        help="directory path to save outuputs",
    )
    parser.add_argument(
        "--speech_file",
        type=str,
        default="real_audio/ljspeech_test.wav",
        help="path to speech file",
    )
    args = parser.parse_args()

    print("Initializing Grad-TTS...")
    generator = GradTTS(
        len(symbols) + 1,
        params.n_enc_channels,
        params.filter_channels,
        params.filter_channels_dp,
        params.n_heads,
        params.n_enc_layers,
        params.enc_kernel,
        params.enc_dropout,
        params.window_size,
        params.n_feats,
        params.dec_dim,
        params.beta_min,
        params.beta_max,
        params.pe_scale,
    )
    generator.load_state_dict(
        torch.load(args.checkpoint, map_location=lambda loc, storage: loc)
    )
    _ = generator.cuda().eval()
    print(f"Number of parameters: {generator.nparams}")

    print("Initializing HiFi-GAN...")
    with open(HIFIGAN_CONFIG) as f:
        h = AttrDict(json.load(f))
    vocoder = HiFiGAN(h)
    vocoder.load_state_dict(
        torch.load(HIFIGAN_CHECKPT, map_location=lambda loc, storage: loc)["generator"]
    )
    _ = vocoder.cuda().eval()
    vocoder.remove_weight_norm()

    with open(args.file, "r", encoding="utf-8") as f:
        text = f.readline()
    text1, text2 = text.split("#")
    cmu = cmudict.CMUDict("./resources/cmu_dictionary")

    audio, sr = ta.load(args.speech_file)
    audio = ta.functional.resample(audio, orig_freq=sr, new_freq=SAMPLE_RATE)

    mel = mel_spectrogram(
        audio,
        params.n_fft,
        params.n_mels,
        SAMPLE_RATE,
        params.hop_length,
        params.win_length,
        params.f_min,
        params.f_max,
        center=False,
    )

    start = 5  # [s]
    end = 6
    time_series = np.arange(0, math.ceil(audio.shape[-1] / SAMPLE_RATE), 1)

    frame_series = librosa.time_to_frames(
        time_series,
        sr=SAMPLE_RATE,
        hop_length=params.hop_length,
        n_fft=params.window_size,
    )
    start_frame = frame_series[start]
    end_frame = frame_series[end]

    mel_len = fix_len_compatibility(mel.shape[-1])
    melspec = torch.zeros((1, mel.shape[1], mel_len), dtype=torch.float32)
    melspec[:, :, : mel.shape[-1]] = mel
    melspec = melspec.cuda()

    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():
        print("Synthesizing content-edited speech...")

        sequence, emphases = text_to_sequence_for_editts(text2, dictionary=cmu)
        x = torch.LongTensor(intersperse(sequence, len(symbols))).cuda()[None]
        emphases = intersperse_emphases(emphases)
        x_lengths = torch.LongTensor([x.shape[-1]]).cuda()

        real_emphases = [start_frame, end_frame]
        y_dec1, y_dec2, y_dec_edit, y_dec_cat = generator.edit_content_with_melspec(
            melspec,
            mel_len,
            x,
            x_lengths,
            real_emphases,
            emphases,
            n_timesteps=args.timesteps,
            temperature=1.5,
            stoc=False,
            length_scale=0.91,
        )

        audio1 = (
            vocoder(y_dec1).cpu().squeeze().clamp(-1, 1).numpy() * VOLUME_MAX
        ).astype(np.int16)
        audio2 = (
            vocoder(y_dec2).cpu().squeeze().clamp(-1, 1).numpy() * VOLUME_MAX
        ).astype(np.int16)
        audio_edit = (
            vocoder(y_dec_edit).cpu().squeeze().clamp(-1, 1).numpy() * VOLUME_MAX
        ).astype(np.int16)
        audio_cat = (
            vocoder(y_dec_cat).cpu().squeeze().clamp(-1, 1).numpy() * VOLUME_MAX
        ).astype(np.int16)

        write(
            os.path.join(args.save_dir, f"gen_gradtts-1.wav"),
            SAMPLE_RATE,
            audio1,
        )
        write(
            os.path.join(args.save_dir, f"gen_gradtts-2.wav"),
            SAMPLE_RATE,
            audio2,
        )
        write(
            os.path.join(args.save_dir, f"gen_EdiTTS.wav"),
            SAMPLE_RATE,
            audio_edit,
        )
        write(
            os.path.join(args.save_dir, f"gen_baseline.wav"),
            SAMPLE_RATE,
            audio_cat,
        )

    print(f"Check out {args.save_dir} folder for generated samples.")