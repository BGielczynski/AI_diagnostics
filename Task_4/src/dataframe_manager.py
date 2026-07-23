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
    """Load WAV files and their filename metadata into one DataFrame."""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.df = pd.DataFrame()

    def load_signals(self) -> None:
        wav_files = sorted(
            glob.glob(os.path.join(self.data_dir, "**", "*.wav"), recursive=True)
        )
        rows = []
        for filepath in wav_files:
            filename = os.path.basename(filepath)
            metadata = self.extract_metadata_from_filename(filename)
            if not metadata:
                print(f"Ueberspringe Datei mit unerwartetem Namen: {filename}")
                continue

            signal, samplerate = sf.read(filepath)
            rows.append(
                {
                    "path": filepath,
                    "fn": filename,
                    "sig": signal,
                    "fs": samplerate,
                    **metadata,
                    "label": LABEL_BY_SPEC[metadata["spec"]],
                }
            )

        self.df = pd.DataFrame(rows)

    @staticmethod
    def extract_metadata_from_filename(filename: str) -> dict:
        parts = os.path.splitext(filename)[0].split("_")
        if len(parts) < 9 or parts[0] not in LABEL_BY_SPEC:
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
