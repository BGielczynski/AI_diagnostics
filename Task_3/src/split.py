import pandas as pd


DEV_POSITIONS = {"Pos03", "Pos05"}
TEST_POSITIONS = {"Pos07", "Pos09"}


def add_position_split(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["split"] = "train"
    result.loc[result["pos"].isin(DEV_POSITIONS), "split"] = "dev"
    result.loc[result["pos"].isin(TEST_POSITIONS), "split"] = "test"
    return result


def summarize_split(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["split", "spec", "sID"])
        .size()
        .reset_index(name="count")
        .sort_values(["split", "spec", "sID"])
    )
