from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ValidationSplit:
    name: str
    train_specs: tuple[str, ...]
    held_out_normal_spec: str

    @property
    def test_specs(self) -> tuple[str, str]:
        return self.held_out_normal_spec, "Z05"


# Reihenfolge und Zusammensetzung entsprechen Seminaraufgabe 4.
VALIDATION_SPLITS = (
    ValidationSplit("fold_1", ("Z01", "Z02", "Z03"), "Z04"),
    ValidationSplit("fold_2", ("Z01", "Z02", "Z04"), "Z03"),
    ValidationSplit("fold_3", ("Z01", "Z03", "Z04"), "Z02"),
    ValidationSplit("fold_4", ("Z02", "Z03", "Z04"), "Z01"),
)


def select_train_test(
    df: pd.DataFrame, split: ValidationSplit, channel: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    channel_df = df[df["sID"] == channel].copy()
    train_df = channel_df[
        channel_df["spec"].isin(split.train_specs) & (channel_df["label"] == 0)
    ].copy()
    test_df = channel_df[channel_df["spec"].isin(split.test_specs)].copy()
    if train_df.empty or test_df.empty:
        raise ValueError(f"Leerer Split fuer {split.name}/{channel}")
    if set(train_df["spec"]) != set(split.train_specs):
        raise ValueError(f"Unvollstaendige Trainingsgruppen fuer {split.name}/{channel}")
    return train_df, test_df
