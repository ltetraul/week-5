import streamlit as st
import pandas as pd
from apputil import survival_demographics, family_groups, visualize_age_division, last_names

#load Titanic dataset
df = pd.read_csv('https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv')

# --- Visualization 1 ---
st.markdown("""
## Were children in third class more likely to survive than adult men in second class?
### Titanic Visualization 1
""")
fig1 = survival_demographics(df)
st.plotly_chart(fig1, use_container_width=True)

#last names summary
last_name_counts = last_names(df)
st.write(f"There are {last_name_counts.shape[0]} unique last names in the dataset.")

# --- Visualization 2 ---
st.markdown("""
## Which families had the most passengers on board and did larger families have higher survival rates?
### Titanic Visualization 2
""")
fig2 = family_groups(df, top_n=10)
st.plotly_chart(fig2, use_container_width=True)

#table for top families
top_families = df.groupby('LastName').agg(
    family_size=('Name', 'count'),
    survival_rate=('Survived', 'mean')
).reset_index().sort_values(by='family_size', ascending=False).head(10)
st.dataframe(top_families)

# --- Bonus Visualization ---
st.markdown("""
## Survival rates based on age relative to class median
### Titanic Visualization Bonus
""")
fig3 = visualize_age_division(df)
st.plotly_chart(fig3, use_container_width=True)
