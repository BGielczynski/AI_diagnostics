import os
from dataframe_manager import DataFrameManager
from stft_processor import calculate_stft, plot_spectrogram

def main():
    # 1. Datenpfad festlegen und Signale laden
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
    manager = DataFrameManager(data_dir=data_path)
    manager.load_signals()
    df = manager.get_dataframe()
    
    print(f"Erfolgreich {len(df)} Signale geladen.")

    # STFT-Parameter (Optimierung gemäß Aufgabenstellung)
    WINDOW   = 'hann'  # Fensterfunktion
    NPERSEG  = 512     # Fensterlänge in Samples
    NOVERLAP = 256     # Fensterfortsetzrate (50% Überlappung)

    # 2. Über jedes Signal iterieren, STFT berechnen und im DataFrame speichern
    stft_results = []
    for index, row in df.iterrows():
        sig = row['sig']
        fs  = row['fs']
        
        f, t, Zxx = calculate_stft(sig, fs, nperseg=NPERSEG, noverlap=NOVERLAP, window=WINDOW)
        stft_results.append(Zxx)
        
        print(f"STFT berechnet für {row['fn']}: Zxx Shape = {Zxx.shape}")

    # STFT-Ergebnisse im DataFrame unter Key 'stft' speichern (Aufgabenanforderung)
    df['stft'] = stft_results

    # 3. Spektrogramme für alle Signale darstellen
    for index, row in df.iterrows():
        f, t, Zxx = calculate_stft(row['sig'], row['fs'], nperseg=NPERSEG, noverlap=NOVERLAP, window=WINDOW)
        plot_spectrogram(t, f, Zxx, title=f"Spektrogramm - {row['fn']}")

if __name__ == '__main__':
    main()
