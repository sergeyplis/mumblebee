"""Fetch a book cover image as cover.png for audiobook players (e.g. Voice on F-Droid).

Zero-API-key lookup cascade:
  1. Embedded EPUB publisher cover (EPUB 3 cover-image / EPUB 2 meta cover)
  2. Open Library Search + Covers API (high-res -L.jpg with ?default=false)
  3. Google Books Volumes API (zoom=2/1 thumbnail fallback)
  4. Optional local PDF Page 1 render via pdftoppm
"""

import argparse
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request

from PIL import Image

from .epub import extract_epub_cover_bytes

USER_AGENT = "book2opus/0.1 (personal audiobook cover fetcher)"
MIN_DIM = 140


def _http_get(url: str, timeout: int = 15) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _validate_and_load_image(data: bytes) -> Image.Image | None:
    try:
        img = Image.open(BytesIO(data))
        img.load()
        w, h = img.size
        if w < MIN_DIM or h < MIN_DIM:
            return None
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        return img
    except Exception:
        return None


def _try_epub_cover(epub_path: Path | None) -> tuple[Image.Image, str] | None:
    if not epub_path:
        return None
    res = extract_epub_cover_bytes(epub_path)
    if not res:
        return None
    data, name = res
    img = _validate_and_load_image(data)
    if img is None:
        return None
    return img, f"EPUB embedded cover ({name}, {img.size[0]}x{img.size[1]})"


def _try_open_library(title: str, author: str = "") -> tuple[Image.Image, str] | None:
    params = {"title": title, "limit": "8"}
    if author:
        params["author"] = author
    url = "https://openlibrary.org/search.json?" + urllib.parse.urlencode(params)
    try:
        data = json.loads(_http_get(url).decode("utf-8"))
    except Exception:
        return None

    for doc in data.get("docs", []):
        cover_i = doc.get("cover_i")
        if not cover_i:
            continue
        img_url = f"https://covers.openlibrary.org/b/id/{cover_i}-L.jpg?default=false"
        try:
            img = _validate_and_load_image(_http_get(img_url))
            if img is not None:
                return img, f"Open Library (cover_i={cover_i}, {img.size[0]}x{img.size[1]})"
        except Exception:
            continue
    return None


def _try_google_books(title: str, author: str = "") -> tuple[Image.Image, str] | None:
    q = f'intitle:"{title}"'
    if author:
        q += f' inauthor:"{author}"'
    url = "https://www.googleapis.com/books/v1/volumes?" + urllib.parse.urlencode({"q": q, "maxResults": "6"})
    try:
        data = json.loads(_http_get(url).decode("utf-8"))
    except Exception:
        return None

    for item in data.get("items", []):
        links = item.get("volumeInfo", {}).get("imageLinks") or {}
        base = links.get("extraLarge") or links.get("large") or links.get("medium") or links.get("thumbnail")
        if not base:
            continue
        base = base.replace("http://", "https://").replace("&edge=curl", "")
        candidates = [base.replace("zoom=1", "zoom=2"), base] if "zoom=1" in base else [base]
        for img_url in candidates:
            try:
                img = _validate_and_load_image(_http_get(img_url))
                if img is not None:
                    return img, f"Google Books ({img.size[0]}x{img.size[1]})"
            except Exception:
                continue
    return None


def _try_pdf_page1(pdf_path: Path | None) -> tuple[Image.Image, str] | None:
    if not pdf_path or not pdf_path.is_file():
        return None
    try:
        with tempfile.TemporaryDirectory(prefix="cover_pdf_") as tmp:
            prefix = Path(tmp) / "page"
            subprocess.run(
                ["pdftoppm", "-png", "-f", "1", "-l", "1", "-r", "150", str(pdf_path), str(prefix)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            pngs = sorted(Path(tmp).glob("page*.png"))
            if not pngs:
                return None
            img = _validate_and_load_image(pngs[0].read_bytes())
            if img is not None:
                return img, f"PDF page 1 ({img.size[0]}x{img.size[1]})"
    except Exception:
        return None
    return None


def fetch_book_cover(
    title: str,
    output_path: Path,
    author: str = "",
    epub_source: Path | None = None,
    pdf_fallback: Path | None = None,
    overwrite: bool = False,
) -> tuple[bool, str]:
    """Fetch cover and save to output_path (PNG/JPG based on extension)."""
    if output_path.is_file() and output_path.stat().st_size > 0 and not overwrite:
        return True, f"already exists ({output_path})"

    clean_title = title.replace("_", " ").strip()
    clean_author = author.strip()

    result = (
        _try_epub_cover(epub_source)
        or _try_open_library(clean_title, clean_author)
        or (clean_author and _try_open_library(clean_title, ""))
        or _try_google_books(clean_title, clean_author)
        or (clean_author and _try_google_books(clean_title, ""))
        or _try_pdf_page1(pdf_fallback)
    )
    if not result:
        return False, "no cover found"

    img, source = result
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, format="PNG" if output_path.suffix.lower() == ".png" else None, optimize=True)
    return True, source


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch a book cover image (EPUB -> Open Library -> Google Books -> PDF page 1)."
    )
    parser.add_argument("--title", required=True, help="Book title")
    parser.add_argument("--author", default="", help="Author name (optional, improves matching)")
    parser.add_argument("--epub", type=Path, help="Optional source EPUB (extracts embedded publisher cover first)")
    parser.add_argument("--pdf", type=Path, help="Optional fallback PDF file (renders page 1 if online lookup fails)")
    parser.add_argument("-o", "--output", type=Path, default=Path("cover.png"), help="Output image path (default: cover.png)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing cover file")
    args = parser.parse_args()

    ok, info = fetch_book_cover(
        title=args.title,
        output_path=args.output,
        author=args.author,
        epub_source=args.epub,
        pdf_fallback=args.pdf,
        overwrite=args.overwrite,
    )
    if ok:
        size_kb = args.output.stat().st_size / 1024
        print(f"Saved {args.output} ({size_kb:.1f} KB) via {info}")
        return 0
    print(f"Could not fetch cover for '{args.title}': {info}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
