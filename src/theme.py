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

        input, textarea, select, .stTextInput input, .stTextArea textarea, .stSelectbox select, .stFileUploader { background: var(--card) !important; color: var(--text) !important; border: 1px solid var(--border) !important; }
        .stFileUploader div, .stFileUploader span, .stFileUploader p { color: var(--muted) !important; }

        label, .stTextInput label, .stSelectbox label, .stSlider label, .stCheckbox label, .stRadio label, .stTextArea label { color: var(--muted) !important; }

        .stMarkdown, .stCaption, .stHeader, .stSubheader, .stText, .stTitle, h1, h2, h3, h4, h5, h6, p { color: var(--text) !important; }

        .stSuccess, .stInfo, .stWarning, .stError,
        .stInfoBox, .stAlert, .stException, .stMessage {
            color: var(--text) !important;
            background: var(--card) !important;
            border: 1px solid var(--border) !important;
            padding: 12px !important;
            border-radius: 10px !important;
            box-shadow: none !important;
        }

        .stAlert > div, .stSuccess > div, .stInfo > div, .stWarning > div, .stError > div,
        .stAlert > div > div, .stSuccess > div > div, .stInfo > div > div, .stWarning > div > div {
            background: var(--card) !important;
            color: var(--text) !important;
            border: none !important;
            box-shadow: none !important;
        }

        [data-testid="stAlert"] > div, [data-testid="stInfo"] > div, [data-testid="stSuccess"] > div {
            background: var(--card) !important;
            color: var(--text) !important;
        }

        .stAlert svg, .stSuccess svg, .stInfo svg, .stWarning svg { fill: var(--accent) !important; }

        .css-1adrfps, .css-1v0mbdj, .css-1outpf7 { background: var(--card) !important; border: 1px solid var(--border) !important; }

        .stCheckbox, .stRadio, .stSwitch { color: var(--text) !important; }

        .stBadge, .stInfoBox { background: var(--panel) !important; color: var(--text) !important; }

        </style>
        """
    else:
        return r"""
        <style>
        .stApp {
            background-color: #ffffff;
            color: #000000;
        }
        </style>
        """
