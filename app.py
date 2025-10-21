"""app.py

Minimal entrypoint for Streamlit that calls page config first and then
imports the real app module. set_page_config must be the first Streamlit
call in the process.
"""

import streamlit as st

# Page config must be called before any other Streamlit calls or imports
# that themselves invoke Streamlit APIs.
st.set_page_config(page_title="RAG Demo", layout="wide")

from src import app_streamlit  # pragma: no cover