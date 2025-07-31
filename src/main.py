from cli import run_cli
from streamlit_ui import run_streamlit_ui
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == "cli":
            run_cli()
        elif mode == "streamlit":
            run_streamlit_ui()
        else:
            print("Unknown mode. Use 'cli' or 'streamlit'.")
    else:
        run_cli()
