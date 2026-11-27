"""Unified `mumblebee` command-line interface.

Usage:
    mumblebee MyBook.epub [options]            # Full pipeline (prep + Kokoro TTS)
    mumblebee prep MyBook.pdf [options]        # Clean & audit text only
    mumblebee kokoro --input ... [options]     # Kokoro-82M TTS only
    mumblebee tinyvoca --text-file ... [opts]  # Ultra-fast tiny-voca TTS only
    mumblebee cover --title ... [options]      # Fetch cover.png only
"""

import sys

from . import __version__

_HELP_TEXT = """usage: mumblebee <book | command> [options]

Turn messy PDFs, EPUBs, and OCR'd books into clean, chaptered local audiobooks.

default (end-to-end):
  mumblebee <book> [options]    Clean a book (.epub/.pdf/.txt/.md) into *_tts_ready.txt
                                and render per-chapter .opus + cover.png via Kokoro.
                                Pass --tts tiny-voca or --tts none to override TTS.

commands:
  prep <book> [options]         Clean and audit book text only (writes *_tts_ready.txt
                                and *_prep_report.txt without running TTS).
  kokoro [options]              Convert cleaned/plain text to chaptered .opus + cover.png
                                with parallel ONNX / Hybrid Kokoro-82M TTS.
  tinyvoca [options]            Ultra-fast 4.5M-parameter parallel ONNX TTS (tiny-voca).
  cover [options]               Fetch cover.png (EPUB -> Open Library -> Google Books -> PDF).

options:
  -h, --help                    Show this help message and exit (or run: mumblebee <cmd> --help)
  -V, --version                 Show program version and exit
"""

_SUBCOMMANDS = {"prep", "kokoro", "tinyvoca", "tiny-voca", "cover"}


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    if not args or args[0] in ("-h", "--help"):
        print(_HELP_TEXT, end="")
        return 0
    if args[0] in ("-V", "--version"):
        print(f"mumblebee {__version__}")
        return 0

    cmd = args[0]
    rest = args[1:]

    if cmd == "prep":
        from . import prep

        return prep.main(argv=rest, prog="mumblebee prep", default_tts="none")
    if cmd == "kokoro":
        from . import tts_kokoro

        return tts_kokoro.main(argv=rest, prog="mumblebee kokoro")
    if cmd in ("tinyvoca", "tiny-voca"):
        from . import tts_tinyvoca

        return tts_tinyvoca.main(argv=rest, prog="mumblebee tinyvoca")
    if cmd == "cover":
        from . import cover

        return cover.main(argv=rest, prog="mumblebee cover")

    # Default shorthand: `mumblebee MyBook.epub [options]` runs full pipeline (prep + kokoro)
    from . import prep

    return prep.main(argv=args, prog="mumblebee", default_tts="kokoro")


if __name__ == "__main__":
    sys.exit(main())
