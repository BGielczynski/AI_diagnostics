import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def calculate_stft(sig, fs, nperseg=256, noverlap=None, window='hann'):
    """
    Berechnet die Kurzzeit-Fourier-Transformation (STFT) für ein gegebenes Signal.
    
    Args:
        sig (np.ndarray): Das Audiosignal.
        fs (int): Die Abtastrate (Sampling Frequency) des Signals.
        nperseg (int): Fensterlänge in Samples.
        noverlap (int): Anzahl der überlappenden Samples (Fensterfortsetzrate).
                        Standard: nperseg // 2 (50% Überlappung).
        window (str): Fensterfunktion, z.B. 'hann', 'hamming', 'blackman', 'boxcar'.
        
    Returns:
        f (np.ndarray): Array der Frequenz-Werte [Hz].
        t (np.ndarray): Array der Zeit-Werte [s].
        Zxx (np.ndarray): STFT der Signaldaten (komplexes Array).
    """
    f, t, Zxx = signal.stft(sig, fs, window=window, nperseg=nperseg, noverlap=noverlap)
    return f, t, Zxx

def plot_spectrogram(t, f, Zxx, title="Spektrogramm", fs=None):
    plt.figure(figsize=(10,5))
    # Betragsspektrum (linear) wie zu Beginn
    magnitude = np.abs(Zxx)
    
    # Plotten mit pcolormesh
    im = plt.pcolormesh(t, f, magnitude, shading='gouraud')
    
    plt.title(title)
    plt.ylabel('Frequenz [Hz]')
    plt.xlabel('Zeit [s]')
    
    plt.colorbar(im, label='Amplitude')
    plt.tight_layout()
    plt.show()
