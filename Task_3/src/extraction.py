import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(__file__).resolve().parents[1] / ".numba_cache"))

warnings.filterwarnings("ignore", message="Empty filters detected in mel frequency basis.*")
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="invalid value encountered in divide")
warnings.filterwarnings("ignore", message="Trying to estimate tuning from empty frequency set.*")
warnings.filterwarnings("ignore", message="n_fft=.* is too large for input signal.*")

import librosa


METADATA_COLUMNS = ["path", "fn", "spec", "pos", "mID", "time", "rID", "sID", "label"]


def get_feature_names() -> list[str]:
    return (
        [f"MFCC_{i + 1}" for i in range(13)]
        + ["Spectral_Centroid", "Spectral_Bandwidth", "Spectral_Rolloff"]
        + [f"Spectral_Contrast_{i + 1}" for i in range(7)]
        + [f"Chroma_STFT_{i + 1}" for i in range(12)]
        + [f"Chroma_CENS_{i + 1}" for i in range(12)]
        + ["Zero_Crossing_Rate", "RMS_Energy"]
        + [f"Delta_MFCC_{i + 1}" for i in range(13)]
    )


def _as_mono_float(signal) -> np.ndarray:
    sig = np.asarray(signal, dtype=float)
    if sig.ndim > 1:
        sig = np.mean(sig, axis=1)
    return np.nan_to_num(sig)


def extract_features_from_signal(signal, samplerate: int) -> np.ndarray:
    """
    Extracts the same feature family that was used in Task 2.
    """
    sig = _as_mono_float(signal)
    if sig.size == 0:
        return np.zeros(len(get_feature_names()), dtype=float)

    n_fft = min(2048, sig.size)
    hop_length = max(1, n_fft // 2)
    fmax = samplerate / 2
    contrast_fmin = max(200.0, samplerate / n_fft)

    try:
        mfcc = librosa.feature.mfcc(
            y=sig,
            sr=samplerate,
            n_mfcc=13,
            n_fft=n_fft,
            hop_length=hop_length,
            fmax=fmax,
        )
        centroid = librosa.feature.spectral_centroid(
            y=sig, sr=samplerate, n_fft=n_fft, hop_length=hop_length
        )
        bandwidth = librosa.feature.spectral_bandwidth(
            y=sig, sr=samplerate, n_fft=n_fft, hop_length=hop_length
        )
        rolloff = librosa.feature.spectral_rolloff(
            y=sig, sr=samplerate, n_fft=n_fft, hop_length=hop_length
        )
        contrast = librosa.feature.spectral_contrast(
            y=sig,
            sr=samplerate,
            n_fft=n_fft,
            hop_length=hop_length,
            fmin=contrast_fmin,
        )
        chroma_stft = librosa.feature.chroma_stft(
            y=sig, sr=samplerate, n_fft=n_fft, hop_length=hop_length
        )
        chroma_cens = librosa.feature.chroma_cens(
            y=sig, sr=samplerate, hop_length=hop_length
        )
        zcr = librosa.feature.zero_crossing_rate(
            sig, frame_length=n_fft, hop_length=hop_length
        )
        energy = librosa.feature.rms(
            y=sig, frame_length=n_fft, hop_length=hop_length
        )
        delta_mfcc = librosa.feature.delta(mfcc)

        vector = np.hstack(
            [
                np.mean(mfcc, axis=1),
                np.mean(centroid),
                np.mean(bandwidth),
                np.mean(rolloff),
                np.mean(contrast, axis=1),
                np.mean(chroma_stft, axis=1),
                np.mean(chroma_cens, axis=1),
                np.mean(zcr),
                np.mean(energy),
                np.mean(delta_mfcc, axis=1),
            ]
        )
    except Exception as exc:
        raise RuntimeError(f"Feature-Extraktion fehlgeschlagen: {exc}") from exc

    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)


def extract_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    feature_names = get_feature_names()

    for _, row in raw_df.iterrows():
        feature_values = extract_features_from_signal(row["sig"], int(row["fs"]))
        feature_row = {column: row[column] for column in METADATA_COLUMNS}
        feature_row.update(dict(zip(feature_names, feature_values)))
        rows.append(feature_row)

    return pd.DataFrame(rows)
