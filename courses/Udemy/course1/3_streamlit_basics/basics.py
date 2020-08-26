import streamlit as st
import numpy as np
import pandas as pd

#########################################################################################
st.title("Sample App")

st.text("Sample text for the app")

st.dataframe(
    pd.DataFrame(
        {
            '1st column': [1, 2, 4, 5, 6],
            '2nd column': [352, 6, 32, 6, 2]
        }
    )
)

chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['1st', '2nd', '3rd']
)

st.line_chart(chart_data)

