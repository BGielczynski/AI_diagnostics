import os
from dataframe_manager import DataFrameManager
from visualizer import plot_dual_spectrograms
from stft_processor import calculate_stft

def main():
    # 1. Datenpfad festlegen und Signale laden
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
    manager = DataFrameManager(data_dir=data_path)
    manager.load_signals()
    df = manager.get_dataframe()
    
    print(f"Erfolgreich {len(df)} Signale geladen.")

    # STFT-Parameter (bereits optimiert)
    WINDOW   = 'hann'           # Fensterfunktion
    NPERSEG  = 550              # Fensterlänge in Samples
    NOVERLAP = NPERSEG * 0.5    # Fensterfortsetzrate (50% Überlappung)

    # 2. Über jedes Signal iterieren, STFT berechnen und im DataFrame speichern
    stft_results = []
    for index, row in df.iterrows():
        sig = row['sig']
        fs  = row['fs']
        
        f, t, Zxx = calculate_stft(sig, fs, nperseg=NPERSEG, noverlap=NOVERLAP, window=WINDOW)
        stft_results.append(Zxx)
        
        print(f"STFT berechnet für {row['fn']}: Zxx Shape = {Zxx.shape}")

    # STFT-Ergebnisse im DataFrame unter Key 'stft' speichern
    df['stft'] = stft_results

    # 3. Spektrogramme geordnet zu vergleich darstellen (fkt aus visualizer.py)
    plot_dual_spectrograms(df, NPERSEG, NOVERLAP, WINDOW)

if __name__ == '__main__':
    main()
