import pandas as pd
from sklearn.model_selection import train_test_split


SPLIT_GROUP_COLUMNS = ["spec", "pos", "rID"]


def add_split(
    df: pd.DataFrame,
    train_size: float = 0.60,
    dev_size: float = 0.20,
    test_size: float = 0.20,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Adds a train/dev/test split.

    Ch1 and Ch2 of the same spec/pos/rID stay in the same split. Stratification is
    done by spec so every Z01-Z05 subset is represented in each split.
    """
    total = train_size + dev_size + test_size
    if abs(total - 1.0) > 1e-9:
        raise ValueError("train_size + dev_size + test_size muss 1.0 ergeben.")

    result = df.copy()
    result["split_group"] = result[SPLIT_GROUP_COLUMNS].agg("_".join, axis=1)

    groups = result[SPLIT_GROUP_COLUMNS + ["split_group"]].drop_duplicates().copy()

    temp_size = dev_size + test_size
    train_groups, temp_groups = train_test_split(
        groups,
        test_size=temp_size,
        random_state=random_state,
        stratify=groups["spec"],
    )

    relative_test_size = test_size / temp_size
    dev_groups, test_groups = train_test_split(
        temp_groups,
        test_size=relative_test_size,
        random_state=random_state,
        stratify=temp_groups["spec"],
    )

    split_map = {
        **dict.fromkeys(train_groups["split_group"], "train"),
        **dict.fromkeys(dev_groups["split_group"], "dev"),
        **dict.fromkeys(test_groups["split_group"], "test"),
    }
    result["split"] = result["split_group"].map(split_map)

    if result["split"].isna().any():
        missing = result.loc[result["split"].isna(), "split_group"].unique()
        raise RuntimeError(f"Nicht alle Gruppen wurden gesplittet: {missing}")

    return result


def summarize_split(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["split", "spec", "sID"])
        .size()
        .reset_index(name="count")
        .sort_values(["split", "spec", "sID"])
    )
