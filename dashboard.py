import streamlit as st
import pandas as pd

st.set_page_config(page_title="Disaster NLP Dashboard", layout="wide")

st.title("🌍 Disaster Information Extraction & Summarization Dashboard")

# Load summarized dataset
@st.cache_data
def load_data():
    return pd.read_csv("disaster_summarized_data.csv")

df = load_data()

# Sidebar Filters
st.sidebar.header("Filter Options")

category = st.sidebar.selectbox(
    "Select Category",
    ["All"] + list(df["Category"].unique())
)

priority = st.sidebar.selectbox(
    "Select Priority",
    ["All"] + list(df["Priority"].unique())
)

filtered_df = df.copy()

if category != "All":
    filtered_df = filtered_df[filtered_df["Category"] == category]

if priority != "All":
    filtered_df = filtered_df[filtered_df["Priority"] == priority]

st.subheader("📊 Filtered Messages")
st.write("Total Messages:", len(filtered_df))

st.dataframe(
    filtered_df[[
        "date",
        "source",
        "Category",
        "Priority",
        "Summary"
    ]]
)

st.subheader("📌 Entity Visualizations")

st.image("top_locations.png")
st.image("entity_distribution.png")

st.markdown("---")
st.markdown("Built using Streamlit + SpaCy + Transformers")