import pandas as pd
import plotly.express as px
import streamlit as st

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

#toggle charts
chart_type = st.sidebar.radio(
)

def visualize_demographic(summary_table: pd.DataFrame):
    """
    Bar chart highlighting specific groups:
    - Children in 3rd class
    - Adult men in 2nd class
    """
    summary_table = summary_table.copy()
    
    #combine categories into a label
    summary_table["Group"] = (
        summary_table["Pclass"].astype(str) + " Class " +
        summary_table["Sex"].str.capitalize() + " " +
        summary_table["AgeCategory"].astype(str)
    )

    #highlight groups from the question
    summary_table["Highlight"] = "Other"
    summary_table.loc[
        (summary_table["Pclass"] == 3) & 
        (summary_table["AgeCategory"] == "Child"),
        "Highlight"
    ] = "Children 3rd Class"
    summary_table.loc[
        (summary_table["Pclass"] == 2) & 
        (summary_table["AgeCategory"] == "Adult") &
        (summary_table["Sex"] == "male"),
        "Highlight"
    ] = "Adult Men 2nd Class"

    #bar chart with color based on highlighted groups
    fig = px.bar(
        summary_table,
        x="Group",
        y="survival_rate",
        color="Highlight",
        text="survival_rate",
        title="Survival Rates by Class, Sex, and Age Group",
        labels={"survival_rate": "Survival Rate", "Group": "Passenger Group"},
        color_discrete_map={
            "Children 3rd Class": "green",
            "Adult Men 2nd Class": "red",
            "Other": "lightgray"
        }
    )

    #format bars
    fig.update_traces(texttemplate="%{text:.0%}", textposition="outside")
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    fig.update_layout(xaxis_tickangle=-25, uniformtext_minsize=8, uniformtext_mode="hide")

    return fig
