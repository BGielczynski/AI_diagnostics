import pandas as pd
import os
import glob
# Eventuell scipy.io.wavfile oder librosa importieren, um .wav Dateien zu lesen
# from scipy.io import wavfile

class DataFrameManager:
    """
    Diese Klasse ist dafür zuständig, die Rohdaten der Ultraschallsignale 
    aus dem data-Ordner zu lesen und in ein sauberes Pandas DataFrame zu strukturieren.
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        # Das Dataframe mit den geforderten Spalten (Keys) initialisieren
        self.df = pd.DataFrame(columns=['fn', 'sig', 'spec', 'mID', 'time', 'rID', 'sID'])

    def load_signals(self):
        """
        Durchsucht den data_dir nach allen .wav Dateien, liest die Signale ein
        und füllt das DataFrame.
        """
        # TODO: Iteriere über alle Unterordner in data/Z01 und data/Z05
        # TODO: Nutze glob oder os.walk, um alle .wav Dateien zu finden
        # TODO: Lese jede Datei ein und hänge die Daten ans Dataframe an
        pass
        
    def extract_metadata_from_filename(self, filepath: str) -> dict:
        """
        Extrahiert die benötigten Metadaten direkt aus dem Dateinamen oder Pfad.
        
        Rückgabe soll ein Dictionary sein mit:
        - 'fn': Dateiname
        - 'spec': Z01 oder Z05 (Probe)
        - 'mID': Messungs-ID
        - 'time': Zeitstempel
        - 'rID': Aufnahme-ID
        - 'sID': Sensor-ID (z.B. Ch1, Ch2)
        """
        # Beispiel Dateiname: Z01_Pos01_Crp1k_200k_0000_1307031436_00000_14_Ch1.wav
        # TODO: Den Dateinamen am Unterstrich ("_") splitten und die Werte zuordnen
        metadata = {}
        return metadata

    def get_dataframe(self) -> pd.DataFrame:
        """Gibt das fertige DataFrame zurück."""
        return self.df

if __name__ == "__main__":
    # Kurzer Test, um zu schauen ob die Klasse funktioniert
    
    # Da wir uns in src/ befinden, ist der data Ordner eine Ebene höher
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    manager = DataFrameManager(data_dir=data_path)
    manager.load_signals()
    
    print("Dataframe Head:")
    print(manager.get_dataframe().head())
