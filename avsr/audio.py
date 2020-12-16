import numpy as np
import librosa


def process_audio(wav, params):
    sr = params.get('sr', 16000)
    winlen_msec = params.get('winlen_msec', 25)
    hoplen_msec = params.get('hoplen_msec', 10)
    power = params.get('power', 1)
    n_mels = params.get('n_mels', 30)

    winlen_samples = int((sr / 1000) * winlen_msec)
    hoplen_samples = int((sr / 1000) * hoplen_msec)

    spec = librosa.feature.melspectrogram(
        wav,
        sr=sr,
        # n_fft=winlen_samples,
        win_length=winlen_samples,
        hop_length=hoplen_samples,
        power=power,
        n_mels=n_mels,
        fmin=125,
        fmax=7600,
        norm=None,

    )
    logspec = np.log(spec + 1e-6)
    return logspec


def read_wav_file(file, sr=16000):
    r"""
    Loads wav files from disk and resamples to 22050 Hz
    The output is shaped as [timesteps, 1]
    Parameters
    ----------
    file:
    sr: desired sampling rate

    Returns
    -------

    """
    data, sr = librosa.load(file, sr)
    return np.expand_dims(data, axis=-1)


def read_mp4_audio_file(file, sr=16000):
    data, native_sr = librosa.core.audio.__audioread_load(file, offset=0.0, duration=None, dtype=np.float32)

    if sr != native_sr:
        data = librosa.resample(data, native_sr, sr, res_type="kaiser_best")

    return np.expand_dims(data, axis=-1)
