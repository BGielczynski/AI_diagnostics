import glob
import os

import pandas as pd
import soundfile as sf


LABEL_BY_SPEC = {
    "Z01": 1,
    "Z02": 1,
    "Z03": 1,
    "Z04": 1,
    "Z05": 0,
}


class DataFrameManager:
    """
    Loads the WAV files and stores signal data plus filename metadata in a DataFrame.
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.df = pd.DataFrame(
            columns=[
                "path",
                "fn",
                "sig",
                "fs",
                "spec",
                "pos",
                "mID",
                "time",
                "rID",
                "sID",
                "label",
            ]
        )

    def load_signals(self) -> None:
        search_pattern = os.path.join(self.data_dir, "**", "*.wav")
        wav_files = sorted(glob.glob(search_pattern, recursive=True))

        data_list = []
        for filepath in wav_files:
            filename = os.path.basename(filepath)

            metadata = self.extract_metadata_from_filename(filename)
            if not metadata:
                print(f"Ueberspringe Datei mit unerwartetem Namen: {filename}")
                continue

            try:
                sig, samplerate = sf.read(filepath)
            except Exception as exc:
                print(f"Fehler beim Lesen von {filename}: {exc}")
                continue

            spec = metadata["spec"]
            row_data = {
                "path": filepath,
                "fn": filename,
                "sig": sig,
                "fs": samplerate,
                "spec": spec,
                "pos": metadata["pos"],
                "mID": metadata["mID"],
                "time": metadata["time"],
                "rID": metadata["rID"],
                "sID": metadata["sID"],
                "label": LABEL_BY_SPEC.get(spec),
            }
            data_list.append(row_data)

        if data_list:
            self.df = pd.DataFrame(data_list)

    def extract_metadata_from_filename(self, filename: str) -> dict:
        """
        Expected format:
        Z01_Pos01_RC2_75k_0000_1307031436_00000_14_Ch1.wav
        """
        name_without_ext = os.path.splitext(filename)[0]
        parts = name_without_ext.split("_")

        if len(parts) < 9:
            return {}

        return {
            "spec": parts[0],
            "pos": parts[1],
            "mID": parts[4],
            "time": parts[5],
            "rID": parts[6],
            "sID": parts[8],
        }

    def get_dataframe(self) -> pd.DataFrame:
        return self.df


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), "..", "..", "Task_2", "data", "sig")
    manager = DataFrameManager(data_dir=data_path)
    manager.load_signals()

    df = manager.get_dataframe()
    print(f"Anzahl geladener Signale: {len(df)}")
    print(df[["fn", "spec", "pos", "mID", "rID", "sID", "label"]].head())
