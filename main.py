from src.movie_agent import run_cli, run_streamlit_ui
import argparse


def main():
    parser = argparse.ArgumentParser(description="Universal Movie Assistant")
    parser.add_argument(
        "--mode", choices=["ui", "cli"], default="cli", help="Choose the interface mode"
    )
    args = parser.parse_args()

    if args.mode == "ui":
        run_streamlit_ui()
    else:
        run_cli()


if __name__ == "__main__":
    main()
