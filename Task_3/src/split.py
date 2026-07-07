import pandas as pd


TRAIN_SPECS = {"Z01", "Z02"}
DEV_SPECS = {"Z04"}
TEST_SPECS = {"Z03"}

DAMAGED_DEV_POSITIONS = {"Pos03", "Pos05"}
DAMAGED_TEST_POSITIONS = {"Pos07", "Pos09"}


def add_position_split(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["split"] = "train"

    result.loc[result["spec"].isin(TRAIN_SPECS), "split"] = "train"
    result.loc[result["spec"].isin(DEV_SPECS), "split"] = "dev"
    result.loc[result["spec"].isin(TEST_SPECS), "split"] = "test"

    damaged_mask = result["spec"] == "Z05"
    result.loc[damaged_mask, "split"] = "train"
    result.loc[
        damaged_mask & result["pos"].isin(DAMAGED_DEV_POSITIONS),
        "split",
    ] = "dev"
    result.loc[
        damaged_mask & result["pos"].isin(DAMAGED_TEST_POSITIONS),
        "split",
    ] = "test"

    return result


def summarize_split(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["split", "spec", "sID"])
        .size()
        .reset_index(name="count")
        .sort_values(["split", "spec", "sID"])
    )
