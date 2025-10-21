"""app.py

Minimal entrypoint for Streamlit that calls page config first and then
imports the real app module. set_page_config must be the first Streamlit
call in the process.
"""

import streamlit as st

# Page config must be called before any other Streamlit calls or imports
# that themselves invoke Streamlit APIs.
st.set_page_config(page_title="bank-rag-ai-app", page_icon="🏦", layout="wide")
import os
# Mark that we set the page config so imported modules can avoid
# calling set_page_config again (Streamlit requires it to be the first call).
os.environ.setdefault("STREAMLIT_PAGE_CONFIG_SET", "1")

from src import app_streamlit  