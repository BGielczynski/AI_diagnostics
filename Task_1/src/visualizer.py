import matplotlib.pyplot as plt
import numpy as np
from stft_processor import calculate_stft

def plot_dual_spectrograms(df, nperseg, noverlap, window):
    """
    Erstellt für jeden Sensor (Ch1, Ch2, Wav3) eine Übersicht:
    - Zeilen: rID 00000 und 00001
    - Spalten: Z01 (gut) und Z05 (Fehler)
    """
    s_ids = ["Ch1", "Ch2", "Wav3"]
    r_ids = ["00000", "00001"]
    specs = ["Z01", "Z05"]
    
    stft_params = {
        'nperseg': nperseg,
        'noverlap': noverlap,
        'window': window
    }

    for s_id in s_ids:
        print(f"Erstelle Übersicht für Sensor: {s_id}")
        fig, axes = plt.subplots(len(r_ids), len(specs), figsize=(12, 9), sharex=True, sharey=True)
        fig.suptitle(f"Spektrogramm-Vergleich | Sensor: {s_id}\nOben: rID 00000 | Unten: rID 00001", fontsize=16)
        
        # Plotten
        for i, r_id in enumerate(r_ids):
            for j, spec in enumerate(specs):
                ax = axes[i, j]
                signal_row = df[(df['rID'] == r_id) & (df['spec'] == spec) & (df['sID'] == s_id)]
                
                if not signal_row.empty:
                    row = signal_row.iloc[0]
                    f, t, Zxx = calculate_stft(row['sig'], row['fs'], **stft_params)
                    magnitude = np.abs(Zxx)
                    im = ax.pcolormesh(t, f, magnitude, shading='gouraud')
                    
                    ax.set_title(f"{spec} | Recording {r_id}")
                    if j == 0: ax.set_ylabel('Frequenz [Hz]')
                    if i == len(r_ids) - 1: ax.set_xlabel('Zeit [s]')
                    
                    # Begrenzung auf den relevanten Bereich des Chirps (200 kHz)
                    ax.set_ylim(0, 200000)
                else:
                    ax.text(0.5, 0.5, "Keine Daten", ha='center', va='center')

        # Gemeinsame Colorbar
        fig.subplots_adjust(right=0.9, top=0.9)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, label='Amplitude')

    plt.show()
