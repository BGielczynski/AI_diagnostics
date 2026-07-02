import pandas as pd


GROUP_COLUMNS = ["spec", "pos", "rID", "sID", "label"]
NON_FEATURE_COLUMNS = {
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
}


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in df.columns
        if column not in NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(df[column])
    ]


def aggregate_over_mid(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Averages feature values over mID while keeping position, rID and sID separate.
    """
    feature_columns = get_feature_columns(feature_df)
    if not feature_columns:
        raise ValueError("Keine numerischen Feature-Spalten gefunden.")

    aggregated = (
        feature_df.groupby(GROUP_COLUMNS, as_index=False)
        .agg(
            {
                **{column: "mean" for column in feature_columns},
                "mID": "nunique",
                "fn": "count",
            }
        )
        .rename(columns={"mID": "mID_count", "fn": "source_file_count"})
    )

    return aggregated


def add_channel_one_hot(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    channel_dummies = pd.get_dummies(result["sID"], prefix="sID", dtype=int)
    return pd.concat([result, channel_dummies], axis=1)
