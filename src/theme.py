# def get_theme_css(dark_mode: bool) -> str:
#     """Return HTML/CSS string for Streamlit theme.

#     Keep CSS here so app_streamlit.py stays clean. dark_mode toggles dark vs light.
#     """
#     if dark_mode:
#         return r"""
#         <style>
#         :root {
#           --bg: #0f1113;
#           --panel: #121416;
#           --muted: #9aa3ad;
#           --text: #e6eef6;
#           --accent: #0ea5a4;
#           --card: #14161a;
#           --border: #26292d;
#         }

#         .stApp, .stApp .css-1d391kg { background: var(--bg) !important; color: var(--text) !important; }
#         .stSidebar, .stSidebar .css-1d391kg { background: var(--panel) !important; color: var(--text) !important; }
#         header, .stApp header, .stDeployButton, .stStatus { background: var(--panel) !important; color: var(--text) !important; }

#         .stButton>button,
#         .stDownloadButton>button,
#         .stSidebar button,
#         .stSidebar .stButton>button,
#         .stApp .stButton>button,
#         button[role="switch"] {
#             background: var(--accent) !important;
#             color: #ffffff !important;
#             border: 1px solid rgba(255,255,255,0.06) !important;
#             box-shadow: 0 1px 0 rgba(0,0,0,0.4) inset;
#             padding: 6px 12px !important;
#             border-radius: 8px !important;
#         }

#         .stButton>button:hover,
#         .stDownloadButton>button:hover,
#         .stSidebar button:hover {
#             filter: brightness(0.95) !important;
#             transform: translateY(-1px);
#         }

#         .stButton>button:disabled,
#         .stDownloadButton>button:disabled {
#             opacity: 0.5 !important;
#             cursor: not-allowed !important;
#         }

#         input, textarea, select, .stTextInput input, .stTextArea textarea, .stSelectbox select, .stFileUploader { background: var(--card) !important; color: var(--text) !important; border: 1px solid var(--border) !important; }
#         /* Make placeholder text more visible in dark mode */
#         input::placeholder,
#         textarea::placeholder,
#         .stTextInput input::placeholder,
#         .stTextArea textarea::placeholder,
#         .stFileUploader input::placeholder {
#             color: var(--muted) !important;
#             opacity: 1 !important;
#         }
#         .stFileUploader div, .stFileUploader span, .stFileUploader p { color: var(--muted) !important; }

#         label, .stTextInput label, .stSelectbox label, .stSlider label, .stCheckbox label, .stRadio label, .stTextArea label { color: var(--muted) !important; }

#         .stMarkdown, .stCaption, .stHeader, .stSubheader, .stText, .stTitle, h1, h2, h3, h4, h5, h6, p { color: var(--text) !important; }

#         .stSuccess, .stInfo, .stWarning, .stError,
#         .stInfoBox, .stAlert, .stException, .stMessage {
#             color: var(--text) !important;
#             background: var(--card) !important;
#             border: 1px solid var(--border) !important;
#             padding: 12px !important;
#             border-radius: 10px !important;
#             box-shadow: none !important;
#         }

#         .stAlert > div, .stSuccess > div, .stInfo > div, .stWarning > div, .stError > div,
#         .stAlert > div > div, .stSuccess > div > div, .stInfo > div > div, .stWarning > div > div {
#             background: var(--card) !important;
#             color: var(--text) !important;
#             border: none !important;
#             box-shadow: none !important;
#         }

#         [data-testid="stAlert"] > div, [data-testid="stInfo"] > div, [data-testid="stSuccess"] > div {
#             background: var(--card) !important;
#             color: var(--text) !important;
#         }

#         .stAlert svg, .stSuccess svg, .stInfo svg, .stWarning svg { fill: var(--accent) !important; }

#         .css-1adrfps, .css-1v0mbdj, .css-1outpf7 { background: var(--card) !important; border: 1px solid var(--border) !important; }

#         .stCheckbox, .stRadio, .stSwitch { color: var(--text) !important; }

#         .stBadge, .stInfoBox { background: var(--panel) !important; color: var(--text) !important; }

#         pre, code, .stText, .stText pre, .stText code {
#             color: var(--text) !important;
#             background: var(--card) !important;
#             border: 1px solid var(--border) !important;
#             padding: 8px !important;
#             border-radius: 4px !important;
#         }

#         </style>
#         """
#     else:
#         return r"""
#         <style>
#         .stApp {
#             background-color: #ffffff;
#             color: #000000;
#         }
#         </style>
#         """

def get_theme_css(dark_mode: bool) -> str:
    """Return HTML/CSS string for Streamlit theme.

    Keep CSS here so app_streamlit.py stays clean. dark_mode toggles dark vs light.
    """
    if dark_mode:
        return r"""
        <style>
        :root {
          --bg: #0f1113;
          --panel: #121416;
          --muted: #9aa3ad;
          --text: #e6eef6;
          --accent: #0ea5a4;
          --card: #14161a;
          --border: #26292d;
        }

        /* ---------- GLOBAL LAYOUT (unchanged) ---------- */
        .stApp, .stApp .css-1d391kg { background: var(--bg) !important; color: var(--text) !important; }
        .stSidebar, .stSidebar .css-1d391kg { background: var(--panel) !important; color: var(--text) !important; }
        header, .stApp header, .stDeployButton, .stStatus { background: var(--panel) !important; color: var(--text) !important; }

        .stButton>button,
        .stDownloadButton>button,
        .stSidebar button,
        .stSidebar .stButton>button,
        .stApp .stButton>button,
        button[role="switch"] {
            background: var(--accent) !important;
            color: #ffffff !important;
            border: 1px solid rgba(255,255,255,0.06) !important;
            box-shadow: 0 1px 0 rgba(0,0,0,0.4) inset;
            padding: 6px 12px !important;
            border-radius: 8px !important;
        }

        .stButton>button:hover,
        .stDownloadButton>button:hover,
        .stSidebar button:hover {
            filter: brightness(0.95) !important;
            transform: translateY(-1px);
        }

        .stButton>button:disabled,
        .stDownloadButton>button:disabled {
            opacity: 0.5 !important;
            cursor: not-allowed !important;
        }

        input, textarea, select,
        .stTextInput input,
        .stTextArea textarea,
        .stSelectbox select,
        .stFileUploader {
            background: var(--card) !important;
            color: var(--text) !important;
            border: 1px solid var(--border) !important;
        }

        input::placeholder,
        textarea::placeholder,
        .stTextInput input::placeholder,
        .stTextArea textarea::placeholder,
        .stFileUploader input::placeholder {
            color: var(--muted) !important;
            opacity: 1 !important;
        }

        .stFileUploader div,
        .stFileUploader span,
        .stFileUploader p {
            color: var(--muted) !important;
        }

        label,
        .stTextInput label,
        .stSelectbox label,
        .stSlider label,
        .stCheckbox label,
        .stRadio label,
        .stTextArea label {
            color: var(--muted) !important;
        }

        .stMarkdown,
        .stCaption,
        .stHeader,
        .stSubheader,
        .stText,
        .stTitle,
        h1, h2, h3, h4, h5, h6, p {
            color: var(--text) !important;
        }

        .stSuccess, .stInfo, .stWarning, .stError,
        .stInfoBox, .stAlert, .stException, .stMessage {
            color: var(--text) !important;
            background: var(--card) !important;
            border: 1px solid var(--border) !important;
            padding: 12px !important;
            border-radius: 10px !important;
            box-shadow: none !important;
        }

        [data-testid="stAlert"] > div,
        [data-testid="stInfo"] > div,
        [data-testid="stSuccess"] > div {
            background: var(--card) !important;
            color: var(--text) !important;
        }

        pre, code,
        .stCodeBlock,
        .stCodeBlock pre,
        .stCodeBlock code {
            color: var(--text) !important;
            background: var(--card) !important;
            border: 1px solid var(--border) !important;
            padding: 8px !important;
            border-radius: 4px !important;
        }

        /* ======================================================
           FIX RETRIEVED CHUNKS / VECTOR DB BROWSER VISIBILITY
           RIGHT COLUMN (expanders)
           ------------------------------------------------------
           Streamlit renders each "st.expander(...)" as:
             <details> ... <summary>header</summary> <div>body...</div> </details>

           We'll style:
           - <details> container background/border
           - <summary> header text color
           - <div> inside details (the body text)
           ====================================================== */

        /* outer expander box */
        details {
            background-color: #1b1f24 !important;      /* readable dark card */
            border: 1px solid #2c3035 !important;
            border-radius: 8px !important;
            color: #e6eef6 !important;
        }

        /* header row "🔎 Chunk 1 — ..." */
        details > summary {
            background-color: #1f242a !important;      /* slightly lighter than body for contrast */
            color: #ffffff !important;                 /* bright header text */
            border-bottom: 1px solid #2c3035 !important;
            font-weight: 500 !important;
        }

        /* Prevent browser default gray triangle color */
        details > summary::-webkit-details-marker {
            color: #ffffff !important;
        }

        /* For Firefox marker */
        details > summary::marker {
            color: #ffffff !important;
        }

        /* body container inside the expander (the retrieved text itself) */
        details > div,
        details > *:not(summary) {
            background-color: #1b1f24 !important;
            color: #ffffff !important;
        }

        /* paragraphs, spans, etc. inside the body */
        details p,
        details span,
        details code,
        details pre,
        details div {
            color: #ffffff !important;
            background-color: transparent !important;
        }

        /* This ensures long paragraphs of retrieved text do NOT render dim gray */
        details p {
            line-height: 1.45rem !important;
        }

        /* ======================================================
           OPTIONAL: tighten vertical spacing inside expanders
        ====================================================== */
        details > div {
            padding: 0.6rem 0.9rem !important;
        }

        </style>
        """
    else:
        # Light mode can stay minimal; override just enough to look sane
        return r"""
        <style>
        .stApp {
            background-color: #ffffff;
            color: #000000;
        }
        pre, code {
            color: #000000 !important;
            background: #f6f6f6 !important;
        }

        /* Light mode expander readability */
        details {
            background-color: #ffffff !important;
            border: 1px solid #d0d0d0 !important;
            border-radius: 8px !important;
            color: #000000 !important;
        }
        details > summary {
            background-color: #fafafa !important;
            color: #000000 !important;
            border-bottom: 1px solid #d0d0d0 !important;
            font-weight: 500 !important;
        }
        details > summary::-webkit-details-marker,
        details > summary::marker {
            color: #000000 !important;
        }
        details > div,
        details > *:not(summary) {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        details p,
        details span,
        details code,
        details pre,
        details div {
            color: #000000 !important;
            background-color: transparent !important;
        }
        details > div {
            padding: 0.6rem 0.9rem !important;
        }
        </style>
        """
