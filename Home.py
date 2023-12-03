import streamlit as st

# page headings
st.set_page_config(layout="wide", page_title="INFO7374: Algorithmic Marketing")
st.title("Multimodal Retrieval Application")
st.header("INFO7374: Assignment 5 - Part 2")
st.subheader("Group 1 (Team 2): Adit Bhosle, Sowmya Chatti, Vasundhara Sharma")


with open("Home.html", mode="r",  encoding="utf8") as file:
    home_page_content = file.read()

st.markdown(home_page_content, unsafe_allow_html=True)