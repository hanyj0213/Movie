from dotenv import load_dotenv
import os
import streamlit as st
import pandas as pd
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI

load_dotenv()
API_KEY = os.environ["OPENAI_API_KEY"]

llm = OpenAI(api_token=API_KEY)
pandas_ai = PandasAI(llm)

st.title("Movie Recommender")
uploaded_file = st.file_uploader("Upload a CSV file with movie data", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Movie Data:")
    st.write(df.head(3))
    st.dataframe(df)

    Prompt = st.text_area("Enter your movie title or preferences or query:")

    if st.button("get Recommendations"):
        if Prompt:
            with st.spinner("Generating recommendations..."):
                st.write(pandas_ai.run(df, Prompt=Prompt))
    else:
        st.warning("Please upload a CSV file to get started.")