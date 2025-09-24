import pandas as pd
import plotly.express as px


def add_age_category(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add an AgeCategory column based on passenger age.
    
    Categories:
        0-11  -> Child
        12-18 -> Teen
        19-58 -> Adult
        59+   -> Senior
    """
    age_bins = [0, 12, 19, 59, float("inf")]
    age_labels = ["Child", "Teen", "Adult", "Senior"]

    df = data.copy()
    df["AgeCategory"] = pd.cut(
        df["Age"],
        bins=age_bins,
        labels=age_labels,
        right=False,
        ordered=True
    )
    return df


def group_passengers(data: pd.DataFrame) -> pd.DataFrame:
    """
    Count passengers by Pclass, Sex, and AgeCategory.
    """
    return (
        data.groupby(["Pclass", "Sex", "AgeCategory"])
        .size()
        .reset_index(name="Count")
    )


def calculate_survival_stats(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate survival counts and rates by Pclass, Sex, and AgeCategory.
    """
    stats = (
        data.groupby(["Pclass", "Sex", "AgeCategory"])
        .agg(
            n_passengers=("Survived", "count"),
            n_survivors=("Survived", "sum")
        )
        .reset_index()
    )
    stats["survival_rate"] = stats["n_survivors"] / stats["n_passengers"]
    return stats


def generate_summary_table(data: pd.DataFrame) -> pd.DataFrame:
    """
    Merge passenger counts with survival statistics.
    """
    grouped = group_passengers(data)
    survival_stats = calculate_survival_stats(data)
    return grouped.merge(survival_stats, on=["Pclass", "Sex", "AgeCategory"], how="left")


def order_summary_table(data: pd.DataFrame) -> pd.DataFrame:
    """
    Return an ordered summary table.
    """
    summary = generate_summary_table(data)

    #categorical ages with correct order
    age_order = ["Child", "Teen", "Adult", "Senior"]
    summary["AgeCategory"] = pd.Categorical(summary["AgeCategory"], categories=age_order, ordered=True)
    return summary.sort_values(by=["Pclass", "Sex", "AgeCategory"]).reset_index(drop=True)


