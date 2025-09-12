import streamlit as st
# from style import set_black_background

# set_black_background() 

# Define your pages
Classification = st.Page(page="pages/classification.py", title="Classification", icon="🅰️")
Regression = st.Page(page="pages/regression.py", title="Regression", icon="📈")

# Navigation setup
pg = st.navigation([Classification, Regression])
pg.run()
