import numpy as np
import matplotlib.pyplot as plt
from scipy import signal    # für die stft

def calculate_stft(sig, fs, nperseg=256, noverlap=None, window='hann'):
    """
    Berechnet die Kurzzeit-Fourier-Transformation (STFT) für ein gegebenes Signal.
    
    Argumente:
        sig (np.ndarray): Das Audiosignal.
        fs (int): Die Abtastrate (Sampling Frequency) des Signals. (wurde selbst ins Dataframe hinzugefügt)
        nperseg (int): Fensterlänge in Samples.
        noverlap (int): Anzahl der überlappenden Samples (Fensterfortsetzrate).
                        Standard: nperseg // 2 (50% Überlappung).
        window (str): Fensterfunktion, z.B. 'hann', 'hamming', 'blackman', 'boxcar'.
        
    Gibt zurück:
        f (np.ndarray): Array der Frequenz-Werte [Hz].
        t (np.ndarray): Array der Zeit-Werte [s].
        Zxx (np.ndarray): STFT der Signaldaten (komplexes Array).
    """
    f, t, Zxx = signal.stft(sig, fs, window=window, nperseg=nperseg, noverlap=noverlap)
    return f, t, Zxx