import streamlit as st

from apputil import *
from apputil import visualize_demographic
from apputil import order_summary_table
from apputil import visualize_families
from apputil import visualize_age_division

# Load Titanic dataset
df = pd.read_csv('https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv')

st.write("Were children in third class more likely to survive than adult men in second class?"
'''
# Titanic Visualization 1

'''
)
# Generate and display the figure
fig1 = visualize_demographic(order_summary_table(df))
st.plotly_chart(fig1, use_container_width=True)

last_name_counts = last_names(df)
st.write("There are", last_name_counts.shape[0], "unique last names in the dataset.")
         
st.write("Which families had the most passengers on board and did larger families have higher survival rates?")
'''
# Titanic Visualization 2
'''

# Generate and display the figure
fig2 = visualize_families(df, top_n=10)
st.plotly_chart(fig2, use_container_width=True)

st.write(
'''
# Titanic Visualization Bonus
'''
)
# Generate and display the figure
fig3 = visualize_age_division(df)
st.plotly_chart(fig3, use_container_width=True)
