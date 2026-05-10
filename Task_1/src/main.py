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

    # Neue Spalte für STFT-Ergebnisse (optional)
    # df['Zxx'] = None 
    
    # 2. Über jedes Signal iterieren und STFT berechnen
    for index, row in df.iterrows():
        sig = row['sig']
        fs = row['fs']  # Neu: Wir haben jetzt die Abtastrate im DataFrame!
        
        # STFT berechnen (nperseg gibt die Fenstergröße an)
        f, t, Zxx = calculate_stft(sig, fs, nperseg=256)
        
        # Du kannst das Ergebnis nun weiterverarbeiten oder speichern:
        # df.at[index, 'Zxx'] = Zxx 
        
        print(f"STFT berechnet für {row['fn']}: Zxx Shape = {Zxx.shape}")
        
        # Beispiel: Das Spektrogramm für das allererste Signal plotten und abbrechen
        if index == 0:
            plot_spectrogram(t, f, Zxx, title=f"Spektrogramm - {row['fn']}")
            break

if __name__ == '__main__':
    main()
