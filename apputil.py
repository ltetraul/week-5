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
    return fig

def add_family_size(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add family size column to the dataset.
    """
    df = data.copy()
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    return df
    
def group_by_family_size_and_class(data: pd.DataFrame) -> pd.DataFrame:
    """
    Group passengers by family size and passenger class
    """
    grouped = (
        data.groupby(["FamilySize", "Pclass"])
        .agg(
            n_passengers=("PassengerId", "count"),
            avg_fare=("Fare", "mean"),
            min_fare=("Fare", "min"),
            max_fare=("Fare", "max")
            )
        .reset_index()
        )
    return grouped
    
def generate_family_size_table(data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a table summarizing family size statistics by class.
    """
    grouped = group_by_family_size_and_class(data)

    #sort by class and family size
    sorted_table = grouped.sort_values(by=["Pclass", "FamilySize"]).reset_index(drop=True)
    return sorted_table

def last_names(data: pd.DataFrame) -> pd.Series:
    """
    Extract the last name of each passenger and return the count for each last name.
    """
    last_name_series = data["Name"].str.split(",").str[0]
    return last_name_series.value_counts()

def visualize_families(df: pd.DataFrame, top_n: int = 10) -> px.bar:
    """
    Visualize the largest families on board the Titanic.
    """
    #extract last names
    df['LastName'] = df['Name'].str.split(",").str[0].str.strip()
    
    #count passengers and calculate average survival rate per family
    family_summary = df.groupby('LastName').agg(
        family_size=('Name', 'count'),
        survival_rate=('Survived', 'mean')
    ).reset_index()
    
    #take top largest families
    top_families = family_summary.sort_values(by='family_size', ascending=False).head(top_n)
    
    #create bar chart
    fig2 = px.bar(
        top_families,
        x='LastName',
        y='family_size',
        color='survival_rate',
        color_continuous_scale='Viridis',
        text='family_size',
        title=f"Top {top_n} Largest Families on the Titanic",
        labels={'LastName': 'Family Last Name', 'family_size': 'Number of Passengers', 'survival_rate': 'Average Survival Rate'}
    )
    fig2.update_traces(textposition='outside')
    fig2.update_layout(xaxis_tickangle=-45) 
    return fig2

def determine_age_division(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add a column 'older_passenger' indicating whether a passenger's age
    is above the median age for their passenger class.
    """
    df = data.copy()

    #calculate median age per Pclass
    class_medians = df.groupby("Pclass")["Age"].transform("median")

    #compare passenger age with class median
    df["older_passenger"] = df["Age"] > class_medians
    return df

def visualize_age_division(data: pd.DataFrame) -> px.bar:
    """
    Visualize survival rates by passenger class and if
    a passenger was older than their class median.
    """
    df = determine_age_division(data)

    #aggregate survival stats
    summary = (
        df.groupby(["Pclass", "older_passenger"])
        .agg(
            n_passengers=("PassengerId", "count"),
            survival_rate=("Survived", "mean")
        )
        .reset_index()
    )

    #create bar chart
    fig = px.bar(
        summary,
        x="Pclass",
        y="survival_rate",
        color="older_passenger",
        text="n_passengers",
        barmode="group",
        title="Survival Rates by Class and Age Division",
        labels={
            "Pclass": "Passenger Class",
            "survival_rate": "Survival Rate",
            "older_passenger": "Older than Class Median?"
        }
    )
    return fig
