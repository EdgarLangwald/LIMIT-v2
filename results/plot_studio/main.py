"""Entrypoint for the LIMIT-v2 plot studio.

    python results/plot_studio/main.py            # open the interactive panel
    python results/plot_studio/main.py --export   # write summary.pdf headless (default config)
    python results/plot_studio/main.py --export out.pdf
"""
import sys

from core import PDF_PATH
from state import default_config, export_pdf


def main():
    args = sys.argv[1:]
    if args and args[0] == "--export":
        out = args[1] if len(args) > 1 else PDF_PATH
        export_pdf(default_config(), out)
        print("wrote", out)
        return
    try:
        from gui import launch_gui
        launch_gui()
    except Exception as e:                              # no display / Tk unavailable -> headless
        print(f"GUI unavailable ({e}); writing default PDF instead.")
        export_pdf(default_config())
        print("wrote", PDF_PATH)


if __name__ == "__main__":
    main()
