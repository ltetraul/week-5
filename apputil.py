import pandas as pd
import plotly.express as px

#exercise 1
def survival_demographics(data: pd.DataFrame) -> px.bar:
    """
    Visualize survival rates by class, sex, age group, and age relative to class median.

    Also shows older vs younger passengers relative to class median as a grouped bar.
    """
    df = data.copy()
    
    #age categories
    age_bins = [0, 12, 19, 59, float("inf")]
    age_labels = ["Child", "Teen", "Adult", "Senior"]
    df["AgeCategory"] = pd.cut(df["Age"], bins=age_bins, labels=age_labels, right=False, ordered=True)
    
    #highlighted demographic groups
    summary = (
        df.groupby(["Pclass", "Sex", "AgeCategory"])
        .agg(n_passengers=("Survived", "count"), n_survivors=("Survived", "sum"))
        .reset_index()
    )
    summary["survival_rate"] = summary["n_survivors"] / summary["n_passengers"]
    summary["Group"] = summary["Pclass"].astype(str) + " Class " + summary["Sex"].str.capitalize() + " " + summary["AgeCategory"].astype(str)
    
    summary["Highlight"] = "Other"
    summary.loc[(summary["Pclass"] == 3) & (summary["AgeCategory"] == "Child"), "Highlight"] = "Children 3rd Class"
    summary.loc[(summary["Pclass"] == 2) & (summary["AgeCategory"] == "Adult") & (summary["Sex"] == "male"), "Highlight"] = "Adult Men 2nd Class"
    
    #create bar chart
    fig1 = px.bar(
        summary,
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
    
    #age relative to class median
    df["older_passenger"] = df["Age"] > df.groupby("Pclass")["Age"].transform("median")
    age_division_summary = (
        df.groupby(["Pclass", "older_passenger"])
        .agg(n_passengers=("PassengerId", "count"), survival_rate=("Survived", "mean"))
        .reset_index()
    )
    
    #create grouped bar chart
    fig2 = px.bar(
        age_division_summary,
        x="Pclass",
        y="survival_rate",
        color="older_passenger",
        text="n_passengers",
        barmode="group",
        title="Survival Rates by Class and Age Relative to Class Median",
        labels={
            "Pclass": "Passenger Class",
            "survival_rate": "Survival Rate",
            "older_passenger": "Older than Class Median?"
        }
    )
    
    #return both figures as a dictionary
    return {"demographics": fig1, "median_age": fig2}


#exercise 2 ---
def family_groups(data: pd.DataFrame, top_n: int = 10) -> px.bar:
    """
    Visualize largest families on board the Titanic and their survival rates.
    """
    df = data.copy()
    
    #extract last names
    df['LastName'] = df['Name'].str.split(",").str[0].str.strip()
    
    #aggregate family info
    family_summary = df.groupby('LastName').agg(
        family_size=('Name', 'count'),
        survival_rate=('Survived', 'mean')
    ).reset_index()
    
    #largest families
    top_families = family_summary.sort_values(by='family_size', ascending=False).head(top_n)
    
    #create bar chart
    fig = px.bar(
        top_families,
        x='LastName',
        y='family_size',
        color='survival_rate',
        color_continuous_scale='Viridis',
        text='family_size',
        title=f"Top {top_n} Largest Families on the Titanic",
        labels={'LastName': 'Family Last Name', 'family_size': 'Number of Passengers', 'survival_rate': 'Average Survival Rate'}
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(xaxis_tickangle=-45)
    
    return fig
