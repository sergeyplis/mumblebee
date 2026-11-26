"""Pure-stdlib EPUB text, metadata, and embedded cover extractor."""

from html.parser import HTMLParser
from pathlib import Path
import posixpath
import re
import urllib.parse
import xml.etree.ElementTree as ET
import zipfile


class _EpubHTMLTextExtractor(HTMLParser):
    """Extract clean readable paragraphs from an EPUB XHTML/HTML spine document."""

    _SKIP_TAGS = frozenset({"script", "style", "head", "nav", "figcaption", "svg", "math"})
    _BLOCK_TAGS = frozenset({
        "p", "div", "h1", "h2", "h3", "h4", "h5", "h6",
        "li", "blockquote", "section", "article", "header", "hr", "br", "tr",
    })

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._in_sup = 0
        self._sup_buf: list[str] = []
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]):
        t = tag.lower()
        attr_map = {k.lower(): (v or "") for k, v in attrs}
        epub_type = attr_map.get("epub:type", "") or attr_map.get("type", "")
        role = attr_map.get("role", "")
        cls = attr_map.get("class", "")

        if self._skip_depth > 0:
            self._skip_depth += 1
            return

        if (
            t in self._SKIP_TAGS
            or (t == "aside" and re.search(r"footnote|endnote|note|rearnote", epub_type + " " + role, re.I))
            or (t == "a" and re.search(r"noteref|doc-noteref", epub_type + " " + role + " " + cls, re.I))
        ):
            self._skip_depth = 1
            return

        if t == "sup":
            self._in_sup += 1
            self._sup_buf = []
            return

        if t == "br":
            self._parts.append("\n")
        elif t in self._BLOCK_TAGS:
            self._parts.append("\n\n")

    def handle_endtag(self, tag: str):
        t = tag.lower()
        if self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if t == "sup" and self._in_sup > 0:
            self._in_sup -= 1
            if self._in_sup == 0:
                sup_txt = "".join(self._sup_buf).strip()
                is_ordinal = sup_txt.lower() in ("st", "nd", "rd", "th")
                is_note_ref = bool(re.fullmatch(r"[\[\(]?(?:\d{1,4}|[*†‡§¶]+|[a-z])[\]\)]?", sup_txt))
                if sup_txt and (is_ordinal or not is_note_ref):
                    self._parts.append(sup_txt)
                self._sup_buf = []
            return
        if t in self._BLOCK_TAGS and t != "br":
            self._parts.append("\n\n")

    def handle_data(self, data: str):
        if self._skip_depth > 0:
            return
        if self._in_sup > 0:
            self._sup_buf.append(data)
            return
        self._parts.append(data)

    def get_text(self) -> str:
        raw = "".join(self._parts)
        lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in raw.split("\n")]
        text = "\n".join(lines)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def _read_opf(zf: zipfile.ZipFile) -> tuple[str, ET.Element]:
    container_xml = ET.fromstring(zf.read("META-INF/container.xml"))
    rootfile = container_xml.find(".//{*}rootfile")
    if rootfile is None or not rootfile.attrib.get("full-path"):
        raise ValueError("Invalid EPUB: missing rootfile in META-INF/container.xml")
    opf_path = rootfile.attrib["full-path"]
    return posixpath.dirname(opf_path), ET.fromstring(zf.read(opf_path))


def _resolve_href(opf_dir: str, href: str) -> str:
    decoded = urllib.parse.unquote(href)
    return posixpath.normpath(posixpath.join(opf_dir, decoded)) if opf_dir else decoded


def extract_epub(input_path: Path) -> tuple[str, dict[str, str]]:
    """Extract spine-ordered text (separated by \\f) and Dublin Core metadata from an .epub."""
    meta: dict[str, str] = {}
    with zipfile.ZipFile(input_path, "r") as zf:
        opf_dir, opf = _read_opf(zf)

        title_el = opf.find(".//{*}metadata/{*}title")
        if title_el is not None and title_el.text:
            meta["title"] = re.sub(r"\s+", " ", title_el.text).strip()
        creator_el = opf.find(".//{*}metadata/{*}creator")
        if creator_el is not None and creator_el.text:
            meta["author"] = re.sub(r"\s+", " ", creator_el.text).strip()

        manifest: dict[str, tuple[str, str]] = {}
        for item in opf.findall(".//{*}manifest/{*}item"):
            iid = item.attrib.get("id", "")
            href = item.attrib.get("href", "")
            mtype = item.attrib.get("media-type", "")
            if iid and href:
                manifest[iid] = (_resolve_href(opf_dir, href), mtype)

        docs: list[str] = []
        for itemref in opf.findall(".//{*}spine/{*}itemref"):
            idref = itemref.attrib.get("idref", "")
            if idref not in manifest:
                continue
            href, mtype = manifest[idref]
            if "html" not in mtype and not href.lower().endswith((".xhtml", ".html", ".htm")):
                continue
            try:
                raw_html = zf.read(href).decode("utf-8", errors="replace")
            except KeyError:
                continue
            parser = _EpubHTMLTextExtractor()
            parser.feed(raw_html)
            parser.close()
            txt = parser.get_text()
            if txt:
                docs.append(txt)

    return "\n\f\n".join(docs), meta


def extract_epub_cover_bytes(epub_path: Path) -> tuple[bytes, str] | None:
    """Extract embedded cover image bytes and filename from an EPUB 2/3 archive."""
    if not epub_path or not epub_path.is_file():
        return None
    try:
        with zipfile.ZipFile(epub_path, "r") as zf:
            opf_dir, opf = _read_opf(zf)

            cover_id = None
            for meta in opf.findall(".//{*}metadata/{*}meta"):
                if meta.attrib.get("name", "").lower() == "cover" and meta.attrib.get("content"):
                    cover_id = meta.attrib["content"]
                    break

            candidates: list[str] = []
            for item in opf.findall(".//{*}manifest/{*}item"):
                iid = item.attrib.get("id", "")
                href = item.attrib.get("href", "")
                mtype = item.attrib.get("media-type", "")
                props = item.attrib.get("properties", "")
                if not href or not mtype.startswith("image/"):
                    continue
                full = _resolve_href(opf_dir, href)
                if "cover-image" in props.split():
                    candidates.insert(0, full)
                elif cover_id and iid == cover_id:
                    candidates.insert(0, full)
                elif "cover" in iid.lower() or "cover" in posixpath.basename(full).lower():
                    candidates.append(full)

            for path in candidates:
                try:
                    return zf.read(path), posixpath.basename(path)
                except KeyError:
                    continue
    except Exception:
        return None
    return None
