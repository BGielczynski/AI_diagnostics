import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def calculate_stft(sig, fs, nperseg=256, noverlap=None):
    """
    Berechnet die Kurzzeit-Fourier-Transformation (STFT) für ein gegebenes Signal.
    
    Args:
        sig (np.ndarray): Das Audiosignal.
        fs (int): Die Abtastrate (Sampling Frequency) des Signals.
        nperseg (int): Länge jedes Segments für die STFT.
        noverlap (int): Anzahl der überlappenden Samples zwischen den Segmenten.
        
    Returns:
        f (np.ndarray): Array der Frequenz-Werte.
        t (np.ndarray): Array der Zeit-Werte.
        Zxx (np.ndarray): STFT der Signaldaten (komplexes Array).
    """
    f, t, Zxx = signal.stft(sig, fs, nperseg=nperseg, noverlap=noverlap)
    return f, t, Zxx

def plot_spectrogram(t, f, Zxx, title="Spektrogramm"):
    plt.figure(figsize=(10,5))
    # Betragsspektrum berechnen (oft in dB für bessere Visualisierung: 20 * np.log10(np.abs(Zxx) + 1e-10))
    magnitude = np.abs(Zxx)
    plt.pcolormesh(t, f, magnitude, shading='gouraud')
    plt.title(title)
    plt.ylabel('Frequenz [Hz]')
    plt.xlabel('Zeit [s]')
    plt.colorbar(label='Amplitude')
    plt.tight_layout()
    plt.show()
