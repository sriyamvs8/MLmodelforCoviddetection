import streamlit as st

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

def login():

    st.title("🩺 COVID AI Assistant")

    username = st.text_input("Username")
    password = st.text_input(
        "Password",
        type="password"
    )

    if st.button("Login"):

        if username == "admin" and password == "1234":

            st.session_state.logged_in = True

            st.success("Login Successful")

            st.rerun()

        else:

            st.error("Invalid Credentials")