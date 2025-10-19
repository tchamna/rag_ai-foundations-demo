import os
import streamlit as st

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    st.run("app.py", server_port=port, server_address="0.0.0.0")
