import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from book2opus.cover import fetch_book_cover
from book2opus.epub import extract_epub


class TestEpubExtraction(unittest.TestCase):
    def test_epub3_text_metadata_and_cover(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            epub = td_path / "sample.epub"
            img_path = td_path / "cover_src.jpg"
            Image.new("RGB", (320, 480), (40, 90, 160)).save(img_path)

            with zipfile.ZipFile(epub, "w") as zf:
                zf.writestr(
                    "META-INF/container.xml",
                    """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles>
</container>""",
                )
                zf.writestr(
                    "OEBPS/content.opf",
                    """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/opf" version="3.0">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Test EPUB Book</dc:title>
    <dc:creator>Ada Lovelace</dc:creator>
  </metadata>
  <manifest>
    <item id="cover-img" href="images/cover.jpg" media-type="image/jpeg" properties="cover-image"/>
    <item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/>
    <item id="ch2" href="ch2.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine><itemref idref="ch1"/><itemref idref="ch2"/></spine>
</package>""",
                )
                zf.write(img_path, "OEBPS/images/cover.jpg")
                zf.writestr(
                    "OEBPS/ch1.xhtml",
                    """<html xmlns:epub="http://www.idpf.org/ops"><body>
<h1>Chapter 1</h1>
<p>Clean prose here<sup>14</sup> with 2<sup>nd</sup> ordinal kept.</p>
<figure><img src="x.png"/><figcaption>Figure 1: Should be stripped.</figcaption></figure>
<aside epub:type="footnote"><p>Footnote body to strip.</p></aside>
</body></html>""",
                )
                zf.writestr("OEBPS/ch2.xhtml", "<html><body><h1>Chapter 2</h1><p>Second chapter body.</p></body></html>")

            text, meta = extract_epub(epub)
            self.assertEqual(meta, {"title": "Test EPUB Book", "author": "Ada Lovelace"})
            self.assertIn("2nd ordinal kept", text)
            self.assertNotIn("14", text)
            self.assertNotIn("Figure 1", text)
            self.assertNotIn("Footnote body", text)
            self.assertIn("\f", text)

            cover_out = td_path / "cover.png"
            ok, info = fetch_book_cover("Test EPUB Book", cover_out, "Ada Lovelace", epub_source=epub)
            self.assertTrue(ok)
            self.assertIn("EPUB embedded cover", info)
            self.assertTrue(cover_out.is_file() and cover_out.stat().st_size > 0)


if __name__ == "__main__":
    unittest.main()
