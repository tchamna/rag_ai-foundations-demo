import streamlit as st
import sys

if __name__ == '__main__':
    # Page configuration
    st.set_page_config(page_title="RAG Demo", layout="wide")
    
    # 🔹 Welcome Banner
    st.markdown(
        """
        ### Pen sɑ́' pəpē': Bienvenu, Welcome 
    
        This app was developed by **Shck Tchamna** to demonstrate how 
        **Retrieval-Augmented Generation (RAG)** can be used to make the most 
        of your data — by querying it **without leaking your data via the internet**.  
        """
    )
    
    st.divider()  # a horizontal line before the main UI

    sys.exit(0)

