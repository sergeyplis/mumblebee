from contextlib import ExitStack, redirect_stdout
import io
import sys
import types
import unittest
from unittest import mock

import mumblebee
from mumblebee import __version__, cli


def _patch_submod(name: str, return_value=0):
    """Patch a lazily-imported mumblebee submodule whether or not it is already imported."""
    fake = types.SimpleNamespace(main=mock.MagicMock(return_value=return_value))
    stack = ExitStack()
    stack.enter_context(mock.patch.dict(sys.modules, {f"mumblebee.{name}": fake}))
    stack.enter_context(mock.patch.object(mumblebee, name, fake, create=True))
    return stack, fake


class TestUnifiedCli(unittest.TestCase):
    def test_help_and_version(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli.main(["--help"])
        self.assertEqual(rc, 0)
        self.assertIn("mumblebee <book>", buf.getvalue())
        self.assertIn("prep <book>", buf.getvalue())
        self.assertIn("kokoro", buf.getvalue())
        self.assertIn("tinyvoca", buf.getvalue())
        self.assertIn("cover", buf.getvalue())

        buf2 = io.StringIO()
        with redirect_stdout(buf2):
            rc2 = cli.main(["--version"])
        self.assertEqual(rc2, 0)
        self.assertEqual(buf2.getvalue().strip(), f"mumblebee {__version__}")

    def test_direct_book_defaults_to_kokoro(self):
        stack, fake_prep = _patch_submod("prep", 0)
        with stack:
            rc = cli.main(["MyBook.epub", "--output-dir", "./books", "--resume"])
        self.assertEqual(rc, 0)
        fake_prep.main.assert_called_once_with(
            argv=["MyBook.epub", "--output-dir", "./books", "--resume"],
            prog="mumblebee",
            default_tts="kokoro",
        )

    def test_prep_subcommand_defaults_to_no_tts(self):
        stack, fake_prep = _patch_submod("prep", 0)
        with stack:
            rc = cli.main(["prep", "MyBook.pdf", "--resume"])
        self.assertEqual(rc, 0)
        fake_prep.main.assert_called_once_with(
            argv=["MyBook.pdf", "--resume"],
            prog="mumblebee prep",
            default_tts="none",
        )

    def test_kokoro_tinyvoca_cover_subcommands(self):
        stack, fake_k = _patch_submod("tts_kokoro", 0)
        with stack:
            self.assertEqual(cli.main(["kokoro", "--input", "a.txt"]), 0)
            fake_k.main.assert_called_once_with(argv=["--input", "a.txt"], prog="mumblebee kokoro")

        stack, fake_tv = _patch_submod("tts_tinyvoca", 0)
        with stack:
            self.assertEqual(cli.main(["tinyvoca", "--text", "hi"]), 0)
            fake_tv.main.assert_called_once_with(argv=["--text", "hi"], prog="mumblebee tinyvoca")

        stack, fake_tv2 = _patch_submod("tts_tinyvoca", 0)
        with stack:
            self.assertEqual(cli.main(["tiny-voca", "--text", "hi"]), 0)
            fake_tv2.main.assert_called_once_with(argv=["--text", "hi"], prog="mumblebee tinyvoca")

        stack, fake_c = _patch_submod("cover", 0)
        with stack:
            self.assertEqual(cli.main(["cover", "--title", "Book"]), 0)
            fake_c.main.assert_called_once_with(argv=["--title", "Book"], prog="mumblebee cover")


if __name__ == "__main__":
    unittest.main()
