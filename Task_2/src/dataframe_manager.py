import pandas as pd     # für Dataframe
import os               # für das durchsuchen der ordner
import glob             # für das durchsuchen der ordner
import soundfile as sf  # für das Laden des Signals

class DataFrameManager:
    """
    DataFrameManager liest die Rohdaten der Signale 
    aus dem data-Ordner und ordnet sie in einem DataFrame.
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        # Das Dataframe mit keys initialisieren
        self.df = pd.DataFrame(columns=['fn', 'sig', 'fs', 'spec', 'mID', 'time', 'rID', 'sID'])

    def load_signals(self):
        """
        Durchsucht den data_dir nach allen .wav Dateien, liest die Signale ein
        und füllt das DataFrame.
        """
        # Suche alle .wav Dateien im data_dir und in allen Unterordnern
        search_pattern = os.path.join(self.data_dir, '**', '*.wav')
        wav_files = glob.glob(search_pattern, recursive=True)
        
        data_list = []
        for filepath in wav_files:
            filename = os.path.basename(filepath)
            
            # Signaldaten einlesen
            try:
                sig, samplerate = sf.read(filepath)
            except Exception as e:
                print(f"Fehler beim Lesen von {filename}: {e}")
                continue
            
            # Metadaten extrahieren
            metadata = self.extract_metadata_from_filename(filename)
            
            # Alles in ein Dictionary packen, das den Spalten entspricht
            row_data = {
                'fn': filename,
                'sig': sig,
                'fs': samplerate,   # selbst hinzugefügt, benötigt für spätere verarbeitung
                'spec': metadata.get('spec'),
                'mID': metadata.get('mID'),
                'time': metadata.get('time'),
                'rID': metadata.get('rID'),
                'sID': metadata.get('sID')
            }
            data_list.append(row_data)
            
        # DataFrame mit allen gesammelten Daten aktualisieren
        if data_list:   # überprüft ob Liste Leer ist: [] -> False
            self.df = pd.DataFrame(data_list)
        
    def extract_metadata_from_filename(self, filename: str) -> dict:
        """
        Extrahiert die benötigten Metadaten direkt aus dem Dateinamen.
        
        Rückgabe ist Dictionary mit:
        - 'fn': Dateiname
        - 'fs': Samplerate
        - 'spec': Z01 oder Z05 (Probe)
        - 'mID': Messungs-ID
        - 'time': Zeitstempel
        - 'rID': Aufnahme-ID
        - 'sID': Sensor-ID (z.B. Ch1, Ch2)
        """
        # Dateiendung entfernen
        name_without_ext = os.path.splitext(filename)[0]
        # Beispiel: Z01_Pos01_Crp1k_200k_0000_1307031436_00000_14_Ch1
        parts = name_without_ext.split('_') # filename an den _ aufteilen
        
        # Sicherstellen, dass das Format wie erwartet ist (mind. 9 Teile durch extra Unterstrich in Crp1k_200k)
        if len(parts) < 9:
            return {}
            
        metadata = {
            'spec': parts[0],      # Z01 oder Z05
            'mID': parts[4],       # 0000 (Messungs-ID)
            'time': parts[5],      # 1307031436 (Zeitstempel)
            'rID': parts[6],       # 00000 (Aufnahme-ID)
            'sID': parts[8]        # Ch1, Ch2 oder Wav3 (Sensor-ID)
        }
        
        return metadata

    def get_dataframe(self) -> pd.DataFrame:
        # Gibt das fertige DataFrame zurück
        return self.df

if __name__ == "__main__":
    # Zum Testen der Klasse:
    
    # sucht im data Ordner eine Ebene höher:
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    manager = DataFrameManager(data_dir=data_path)
    manager.load_signals()
    
    print(f"Anzahl geladener Signale: {len(manager.get_dataframe())}")
    print("Dataframe:")
    print(manager.get_dataframe())
