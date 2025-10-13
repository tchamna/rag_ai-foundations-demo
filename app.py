import os

# Top-level entrypoint for Hugging Face Spaces (Gradio)
# This imports the Gradio `demo` object from `src.app_gradio` and launches it.
from src.app_gradio import demo

if __name__ == '__main__':
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
