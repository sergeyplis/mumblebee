"""LLM-powered book preprocessor for audiobook TTS (Kokoro / tiny-voca).

Takes a raw .epub, .pdf, .txt, or .md book and uses any OpenAI-compatible LLM
server (llama-server, Ollama, vLLM, LM Studio, etc.) to:
  1. Detect book metadata (title, author), running headers, readable body
     boundaries (keeping front matter like Preface/Intro, stripping publishing
     boilerplate, References/Notes, Bibliography, and Index), and chapter lines.
  2. Clean each chapter in sentence-bounded chunks with PDF paragraph restoration,
     automated defect validation (headers, page numbers, Fig./TABLE captions,
     ALL-CAPS diagram labels, merged walls of text, letter-spaced text, broken
     stitches), and automatic Tier-2 escalation to a stronger fallback model.
  3. Write <stem>_tts_ready.txt with standardized <<<CHAPTER: ...>>> markers
     plus an audit report, with optional automatic TTS handoff.
"""

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

from tqdm import tqdm

from .epub import extract_epub

DEFAULT_API_URL = os.environ.get("MUMBLEBEE_API_URL", "http://127.0.0.1:8081/v1/chat/completions")
DEFAULT_MODEL = os.environ.get("MUMBLEBEE_MODEL", "qwen3-4b-instruct")
DEFAULT_FALLBACK_API_URL = os.environ.get("MUMBLEBEE_FALLBACK_API_URL", "http://127.0.0.1:8080/v1/chat/completions")
DEFAULT_FALLBACK_MODEL = os.environ.get("MUMBLEBEE_FALLBACK_MODEL", "qwen3.8-flash-next")

LIGATURES = str.maketrans({
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "ft",
    "\ufb06": "st",
})

PREAMBLE_RE = re.compile(
    r"^(?:here is the cleaned(?: text)?|cleaned text|output)\s*:\s*\n+",
    re.IGNORECASE,
)
_ENDS_SENTENCE_RE = re.compile(r'[.!?…]["\'\u201d\u2019)\]]*$')
_VISUAL_PAREN_RE = re.compile(
    r"\s*\((?:see\s+)?(?:Figs?\.?|Figures?|Tables?|Plates?|Charts?|Diagrams?|Maps?|Illustrations?)\s+[0-9IVX]+(?:[\s,–-]+[0-9IVXa-z]+)*\)",
    re.IGNORECASE,
)
_FIG_CAPTION_RE = re.compile(
    r"^\s*(?:Fig(?:ure)?s?\.?\s+\d+[\.:]|TABLE\s+[0-9IVX]+\b|Plate\s+[0-9IVX]+[\.:])",
    re.IGNORECASE | re.MULTILINE,
)
_FALLBACK_LOCK = threading.Lock()


def extract_text(input_path: Path) -> tuple[str, dict[str, str]]:
    ext = input_path.suffix.lower()
    meta: dict[str, str] = {}
    if ext == ".pdf":
        if shutil.which("pdftotext") is None:
            raise RuntimeError("pdftotext (poppler-utils) is required for .pdf inputs")
        res = subprocess.run(
            ["pdftotext", str(input_path), "-"],
            capture_output=True,
            text=True,
            check=True,
        )
        raw = res.stdout
    elif ext == ".epub":
        raw, meta = extract_epub(input_path)
    else:
        raw = input_path.read_text(encoding="utf-8", errors="replace")

    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    raw = raw.translate(LIGATURES)
    raw = re.sub(r"\u00ad[ \t]*\n[ \t]*", "", raw)
    raw = raw.replace("\u00ad", "")
    return raw, meta


def llm_chat(
    api_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 4096,
    temperature: float = 0.05,
    enable_thinking: bool = False,
    api_key: str | None = None,
    timeout: int = 300,
) -> tuple[str, dict]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "top_p": 0.95,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    headers = {"Content-Type": "application/json"}
    token = api_key or os.environ.get("OPENAI_API_KEY")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(api_url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    choice = body["choices"][0]
    content = choice["message"].get("content") or ""
    return content.strip(), {
        "usage": body.get("usage") or {},
        "timings": body.get("timings") or {},
        "finish_reason": choice.get("finish_reason"),
    }


def extract_json_object(text: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Could not locate JSON object in LLM response:\n{text[:500]}")
    return json.loads(cleaned[start : end + 1])


def _norm_alpha(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _find_title_candidates(lines: list[str], titles: list[str], start_after: int = 0) -> list[tuple[int, str]]:
    """Find compact candidate line locations for TOC / chapter / back-matter titles (OCR-space tolerant)."""
    keys: list[str] = []
    for t in titles:
        t_clean = re.sub(
            r"^(?:chapter|part|section|appendix)\s+[0-9ivxab]+[:.\-\s]*", "", t, flags=re.IGNORECASE
        ).strip()
        for variant in (t, t_clean):
            k = _norm_alpha(variant)
            if len(k) >= 5:
                keys.append(k)
    keys = sorted(set(keys), key=len, reverse=True)

    hits: dict[int, str] = {}
    n = len(lines)
    i = start_after
    while i < n:
        raw_i = lines[i].replace("\f", "").strip()
        if not raw_i or len(raw_i) > 95:
            i += 1
            continue
        win_lines = []
        j = i
        while j < n and len(win_lines) < 4:
            s = lines[j].replace("\f", "").strip()
            if s:
                if len(s) > 95:
                    break
                win_lines.append((j, s))
            j += 1
        if not win_lines:
            i += 1
            continue
        joined_norm = _norm_alpha(" ".join(s for _, s in win_lines))
        matched = False
        for k in keys:
            if k in joined_norm:
                anchor = win_lines[0][0]
                if anchor > 0:
                    prev_s = lines[anchor - 1].replace("\f", "").strip()
                    if prev_s and re.fullmatch(
                        r"[-—\s]*(?:chapter\s+)?(?:\d{1,2}|[ivx]+)[-—\s]*", prev_s, re.IGNORECASE
                    ):
                        anchor -= 1
                if anchor not in hits:
                    snippet = " / ".join(s for _, s in win_lines[:4])
                    pb = "[PB] " if "\f" in lines[anchor] or (anchor > 0 and "\f" in lines[anchor - 1]) else ""
                    hits[anchor] = f"L{anchor+1}: {pb}{snippet[:140]}"
                matched = True
                break
        i = (win_lines[-1][0] + 1) if matched else (i + 1)
    return sorted(hits.items())


def detect_structure(
    lines: list[str],
    api_url: str,
    model: str,
    keep_appendices: bool = False,
    api_key: str | None = None,
) -> dict:
    """Two-step structure detection: (1a) read front/back matter & TOC, (1b) locate exact body start lines."""
    n = len(lines)
    front_n = min(220, n)
    back_start = max(front_n, n - 180)

    front_txt = "\n".join(
        f"L{i+1}: {lines[i].replace(chr(12), '<PB> ')[:140]}"
        for i in range(front_n)
        if lines[i].replace("\f", "").strip()
    )
    back_txt = "\n".join(
        f"L{i+1}: {lines[i].replace(chr(12), '<PB> ')[:140]}"
        for i in range(back_start, n)
        if lines[i].replace("\f", "").strip()
    )

    sys_1a = (
        "You are an expert book editor preparing a book for audiobook narration. "
        "Return ONLY valid JSON."
    )
    user_1a = f"""Inspect the front matter and back matter of this {n}-line book.

Extract:
1. `title`: clean book title (fix OCR split spaces).
2. `author`: clean author name.
3. `running_headers`: likely page running headers (e.g. author name, book title).
4. `toc_end_line`: 1-based line number where the Table of Contents ends (or 0 if no TOC).
5. `toc_sections`: ordered list of ALL readable section/chapter titles listed in the TOC (keep Preface/Prologue/Intro/numbered chapters; fix OCR split spaces).
6. `back_matter_titles`: titles of back-matter sections to EXCLUDE from narration (e.g. Appendices, Glossary, References and Notes, Bibliography, Index).{" Note: KEEP narrative Appendices in toc_sections rather than back_matter_titles." if keep_appendices else ""}

Return ONLY JSON:
{{
  "title": "...",
  "author": "...",
  "running_headers": ["..."],
  "toc_end_line": 165,
  "toc_sections": ["..."],
  "back_matter_titles": ["..."]
}}

=== FRONT MATTER (L1..L{front_n}) ===
{front_txt}

=== BACK MATTER (L{back_start+1}..L{n}) ===
{back_txt}
"""
    raw_1a, _ = llm_chat(
        api_url=api_url,
        model=model,
        system_prompt=sys_1a,
        user_prompt=user_1a,
        max_tokens=2048,
        temperature=0.0,
        enable_thinking=False,
        api_key=api_key,
    )
    meta_1a = extract_json_object(raw_1a)
    toc_end = max(0, min(n - 1, int(meta_1a.get("toc_end_line") or 0)))
    toc_sections = [str(s).strip() for s in meta_1a.get("toc_sections", []) if str(s).strip()]
    back_titles = [str(s).strip() for s in meta_1a.get("back_matter_titles", []) if str(s).strip()]

    all_search_titles = toc_sections + back_titles + [
        "Preface", "Prologue", "Foreword", "Introduction",
        "Appendix", "Glossary", "References", "Notes", "Bibliography", "Index",
    ]
    candidates = _find_title_candidates(lines, all_search_titles, start_after=toc_end)

    page_starts: list[tuple[int, str]] = []
    for i in range(toc_end, n):
        if "\f" in lines[i]:
            snip = []
            for j in range(i, min(i + 8, n)):
                s = lines[j].replace("\f", "").strip()
                if s:
                    snip.append(s[:65])
                    if len(snip) >= 3:
                        break
            if snip:
                page_starts.append((i, f"L{i+1}: [PB] {' / '.join(snip)}"))

    cand_map = dict(candidates)
    if not toc_sections:
        for idx, txt in page_starts:
            cand_map.setdefault(idx, txt)
    else:
        for idx, txt in page_starts:
            if re.search(
                r"(?:chapter|part|preface|intro|appendix|glossary|references|notes|index|\b\d{1,2}\s*/|---)",
                txt,
                re.IGNORECASE,
            ):
                cand_map.setdefault(idx, txt)

    ordered_cands = [cand_map[k] for k in sorted(cand_map.keys())][:350]
    cand_block = "\n".join(ordered_cands)

    appendix_rule = (
        "INCLUDE narrative Appendices in sections, STOP before Glossary / References / Notes / Bibliography / Index."
        if keep_appendices
        else "STOP before Appendices, Glossary, References and Notes, Bibliography, and Index."
    )
    user_1b = f"""Match each readable section of '{meta_1a.get('title', '')}' by {meta_1a.get('author', '')} to its EXACT 1-based `start_line` in the body (after TOC line L{toc_end}).

Expected readable sections (from TOC): {json.dumps(toc_sections, ensure_ascii=False)}
Back-matter sections to EXCLUDE: {json.dumps(back_titles, ensure_ascii=False)}

Rules:
- Every `start_line` MUST be the actual chapter/section heading line in the body (NOT inside References/Notes at the back, and NOT an incidental mention in a paragraph or table).
- `body_start_line` = `start_line` of the first readable section (keep Preface/Prologue/Intro).
- `body_end_line` = line immediately BEFORE the first excluded back-matter section ({appendix_rule}).
- `sections` MUST be strictly ordered by ascending `start_line`.

Return ONLY JSON:
{{
  "body_start_line": 168,
  "body_end_line": 9104,
  "sections": [
    {{"title": "Preface", "start_line": 168}}
  ]
}}

CANDIDATE LINES IN BODY (total book lines = {n}):
{cand_block}
"""
    raw_1b, _ = llm_chat(
        api_url=api_url,
        model=model,
        system_prompt=sys_1a,
        user_prompt=user_1b,
        max_tokens=2048,
        temperature=0.0,
        enable_thinking=False,
        api_key=api_key,
    )
    loc = extract_json_object(raw_1b)

    b_start = max(1, min(n, int(loc.get("body_start_line", 1))))
    b_end = max(b_start, min(n, int(loc.get("body_end_line", n))))
    sections = []
    for sec in loc.get("sections", []):
        st = max(b_start, min(b_end, int(sec["start_line"])))
        title = re.sub(r"\s+", " ", str(sec["title"])).strip()
        if not title:
            continue
        if sections and st <= sections[-1]["start_line"]:
            continue
        sections.append({"title": title, "start_line": st})
    if not sections:
        sections = [{"title": meta_1a.get("title") or "Full Book", "start_line": b_start}]
    else:
        sections[0]["start_line"] = min(sections[0]["start_line"], b_start)

    return {
        "title": meta_1a.get("title") or "",
        "author": meta_1a.get("author") or "",
        "running_headers": meta_1a.get("running_headers") or [],
        "body_start_line": b_start,
        "body_end_line": b_end,
        "sections": sections,
    }


def _restore_pdf_paragraphs(section_lines: list[str]) -> list[str]:
    """Insert blank lines at likely paragraph ends in pdftotext output."""
    body_lens = [len(l.strip()) for l in section_lines if "\f" not in l and len(l.strip()) >= 35]
    if not body_lens:
        return section_lines
    body_lens.sort()
    typical_len = body_lens[int(len(body_lens) * 0.65)]
    short_thresh = max(30, int(typical_len * 0.84))

    out: list[str] = []
    n = len(section_lines)
    for i, line in enumerate(section_lines):
        out.append(line)
        if "\f" in line:
            continue
        s = line.strip()
        if not s or i + 1 >= n:
            continue
        nxt = section_lines[i + 1]
        if "\f" in nxt or not nxt.strip():
            continue
        nxt_s = nxt.strip()
        first_ch = nxt_s[0]
        if (
            len(s) <= short_thresh
            and _ENDS_SENTENCE_RE.search(s)
            and (first_ch.isupper() or first_ch in '"\u201c\u2018\'')
        ):
            out.append("")
    return out


def _is_header_or_pagenum(blk: str, running_headers_norm: set[str]) -> bool:
    s = blk.strip()
    if not s:
        return True
    if re.fullmatch(r"(?:\d{1,4}|[ivxlcdm]{1,6})", s, re.IGNORECASE):
        return True
    norm = _norm_alpha(s)
    if norm and norm in running_headers_norm:
        return True
    if len(s) <= 45 and any(norm == h or (len(norm) >= 5 and norm in h) for h in running_headers_norm):
        return True
    return False


def _block_looks_like_complete_prose(blk: str) -> bool:
    lines = [l.strip() for l in blk.split("\n") if l.strip()]
    if not lines:
        return False
    last = lines[-1]
    if len(blk.split()) < 8:
        return False
    if re.search(r"\b(?:Mr|Mrs|Ms|Dr|Prof|St|Fig|Figs|No|Vol|pp?|ed|al|vs|etc|mm|cm|ml)\.$", last, re.IGNORECASE):
        return False
    return bool(_ENDS_SENTENCE_RE.search(last))


def _next_prose_starts_cleanly(blocks: list[str], start_idx: int, running_headers_norm: set[str]) -> bool:
    for j in range(start_idx, len(blocks)):
        b = blocks[j]
        if b == "\f" or _is_header_or_pagenum(b, running_headers_norm):
            continue
        s = b.strip()
        if not s:
            continue
        if re.match(r"^(?:TABLE\s+[0-9IVX]+|FIG(?:URE)?\.?\s+\d+)", s, re.IGNORECASE):
            return False
        first_alpha = next((ch for ch in s if ch.isalpha()), "")
        if first_alpha and first_alpha.islower():
            return False
        return True
    return True


def split_section_into_chunks(
    section_lines: list[str],
    target_words: int = 1100,
    max_words: int = 1540,
    running_headers: list[str] | None = None,
    is_pdf: bool = False,
) -> list[str]:
    """Split a section into chunks ONLY at clean sentence/paragraph boundaries (never on mid-sentence page breaks)."""
    if is_pdf:
        section_lines = _restore_pdf_paragraphs(section_lines)
    running_headers_norm = {_norm_alpha(h) for h in (running_headers or []) if _norm_alpha(h)}

    blocks: list[str] = []
    cur: list[str] = []
    for line in section_lines:
        if "\f" in line:
            before, after = line.split("\f", 1)
            if before.strip():
                cur.append(before)
            if cur:
                blocks.append("\n".join(cur))
                cur = []
            blocks.append("\f")
            if after.strip():
                cur.append(after)
        elif not line.strip():
            if cur:
                blocks.append("\n".join(cur))
                cur = []
        else:
            cur.append(line)
    if cur:
        blocks.append("\n".join(cur))

    chunks: list[str] = []
    cur_blocks: list[str] = []
    cur_words = 0

    for idx, blk in enumerate(blocks):
        if blk == "\f":
            cur_blocks.append("<PAGEBREAK>")
            continue

        cur_blocks.append(blk)
        cur_words += len(blk.split())

        can_split_cleanly = (
            cur_words >= target_words
            and _block_looks_like_complete_prose(blk)
            and _next_prose_starts_cleanly(blocks, idx + 1, running_headers_norm)
        )
        force_split = cur_words >= max_words and _block_looks_like_complete_prose(blk)
        extreme_split = cur_words >= int(max_words * 1.3)

        if can_split_cleanly or force_split or extreme_split:
            chunks.append("\n\n".join(cur_blocks))
            cur_blocks = []
            cur_words = 0

    if cur_blocks:
        if chunks and cur_words < 220:
            chunks[-1] = chunks[-1] + "\n\n" + "\n\n".join(cur_blocks)
        else:
            chunks.append("\n\n".join(cur_blocks))
    return [c for c in chunks if c.replace("<PAGEBREAK>", "").strip()]


def build_clean_prompts(
    chunk_text: str,
    book_title: str,
    author: str,
    chapter_title: str,
    running_headers: list[str],
    prev_tail: str,
    next_head: str,
    retry_issues: list[str] | None = None,
) -> tuple[str, str]:
    headers_str = ", ".join(repr(h) for h in running_headers) if running_headers else "author name / book title"
    system_prompt = (
        "You are a meticulous audiobook copy-editor cleaning OCR/PDF/EPUB book text for text-to-speech narration.\n"
        "ABSOLUTE RULES:\n"
        "1. PRESERVE 100% OF THE AUTHOR'S PROSE VERBATIM. Never summarize, paraphrase, abridge, rewrite, or invent text.\n"
        "2. PRESERVE PARAGRAPH BREAKS:\n"
        "   - Keep every paragraph separated by a blank line. NEVER merge separate paragraphs into one huge wall of text!\n"
        "   - Only join lines that belong to the SAME sentence/paragraph (wrapped lines, hyphenated words, or a sentence split across a <PAGEBREAK>, running header, or floating figure/table).\n"
        "3. FIX OCR AND TYPOGRAPHICAL SCARS:\n"
        "   - Fix broken spaces inside words and letter-spaced typography (e.g. 'o f' -> 'of', 'w ork' -> 'work', 'ad u lth oo d' -> 'adulthood', 'A n n a H e m p s t e a d B r a n c h' -> 'Anna Hempstead Branch').\n"
        "   - Rejoin hyphenated words split across line breaks and normalize broken quotes/dashes into clean standard punctuation.\n"
        "4. REMOVE ALL NON-NARRATIVE INTERRUPTIONS:\n"
        "   - Remove <PAGEBREAK> markers, standalone page numbers (Arabic or Roman like ix, X, 12, 23), and running headers (" + headers_str + ").\n"
        "   - Remove inline footnote/endnote reference numbers glued to words or punctuation (e.g. 'century14' -> 'century', 'humanity.2' -> 'humanity.'), footnote bodies at page bottoms, parenthetical academic citations like '(Smith, 1974, p. 23)', and end-of-chapter reference lists.\n"
        "   - Remove ALL figure/photo/illustration/chart/map captions (e.g. 'Fig. 1. Showing...', 'Figure 3...'), photo credits ('(After Schultz)', 'Courtesy of...', 'Redrawn from...'), diagram/anatomical labels OCR'd out of illustrations, and ALL raw data tables / multi-column tabular lists (delete BOTH the table header AND its descriptive title AND EVERY SINGLE ROW/ITEM of the table!). Also strip parenthetical visual pointers like '(see Table II)', '(Figure 9)', '(Fig. 3)'.\n"
        "   - Keep natural prose, epigraphs, block quotes, poetry/verse, and section subheadings, ALWAYS placing each subheading on its own standalone paragraph separated by blank lines.\n"
        "5. BOUNDARY HANDLING (CRITICAL):\n"
        "   - Only output cleaned text for <TARGET_CHUNK>. Do NOT repeat text from <PREV_CONTEXT> or <NEXT_CONTEXT>.\n"
        "   - NEVER invent, guess, or add words to finish an incomplete sentence at the end of <TARGET_CHUNK>! If <TARGET_CHUNK> ends mid-sentence, stop at the exact last word of <TARGET_CHUNK> with NO invented ending and NO added period.\n"
        "6. OUTPUT FORMAT:\n"
        "   - Output ONLY the cleaned prose paragraphs separated by blank lines. No commentary, no markdown fences, no 'Here is the cleaned text:'.\n"
        "   - If <TARGET_CHUNK> contains ONLY figures, diagram labels, tables, references, or page numbers with zero narrative prose, output exactly: [[EMPTY]]"
    )
    if retry_issues:
        system_prompt += (
            "\n\nCRITICAL CORRECTION REQUIRED — A previous draft failed automated quality checks for:\n- "
            + "\n- ".join(retry_issues)
            + "\nFix EVERY ONE of those issues completely while keeping all genuine author prose verbatim and properly paragraphed."
        )

    user_parts = [f"Book: {book_title} by {author} | Section: {chapter_title}"]
    if prev_tail:
        user_parts.append(f"<PREV_CONTEXT (read-only, DO NOT OUTPUT)>\n{prev_tail}\n</PREV_CONTEXT>")
    user_parts.append(f"<TARGET_CHUNK>\n{chunk_text}\n</TARGET_CHUNK>")
    if next_head:
        user_parts.append(f"<NEXT_CONTEXT (read-only, DO NOT OUTPUT)>\n{next_head}\n</NEXT_CONTEXT>")
    return system_prompt, "\n\n".join(user_parts)


def postprocess_llm_chunk(raw: str) -> str:
    text = raw.strip()
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = PREAMBLE_RE.sub("", text).strip()
    if text == "[[EMPTY]]":
        return ""
    text = text.replace("<PAGEBREAK>", "")
    text = _VISUAL_PAREN_RE.sub("", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


_VALID_SHORT_WORDS = {
    "a", "i",
    "am", "an", "as", "at", "be", "by", "do", "go", "he", "hi", "if", "in", "is", "it",
    "me", "my", "no", "of", "oh", "ok", "on", "or", "ox", "so", "to", "up", "us", "we",
    "ye", "ah", "ha", "ho", "lo", "um", "uh",
    # Common foreign particles, Latin, Roman numerals, units, and standard abbreviations
    "la", "le", "de", "du", "di", "da", "el", "en", "un", "et", "ex", "ad", "id", "al", "y",
    "re", "vs", "dr", "mr", "ms", "st", "jr", "sr", "pp", "ch", "bc", "ce",
    "kg", "mm", "cm", "ml", "km", "mg", "lb", "oz", "ft", "yr", "hr",
    "ii", "iv", "vi", "ix", "xi", "xv", "xx",
}
_OCR_SPLIT_TAIL_RE = re.compile(r"(?<![A-Za-z'\u2019\-])([A-Za-z]{2,})\s+([a-z]{1,2})(?![A-Za-z\-\.'\u2019])")
_OCR_SPLIT_LEAD_RE = re.compile(r"(?<![A-Za-z'\u2019\.\-])([A-Za-z])\s+([a-z]{1,2})(?![A-Za-z\-\.'\u2019])")


def find_ocr_word_splits(text: str) -> list[str]:
    """Detect likely single-space mid-word OCR splits like 'Sometim es', 'o f', or 'L a'."""
    matches: list[tuple[int, int, str]] = []
    occupied: list[tuple[int, int]] = []

    for m in _OCR_SPLIT_LEAD_RE.finditer(text):
        w1, w2 = m.group(1), m.group(2)
        hit = False
        if w1.islower():
            if w1 not in ("a", "i") or (len(w2) == 1 and w2 not in ("a", "i")):
                hit = True
        else:
            if len(w2) == 1 and not (w1 in ("A", "I") and w2 in ("a", "i")):
                hit = True
        if hit:
            matches.append((m.start(), m.end(), m.group(0)))
            occupied.append((m.start(), m.end()))

    for m in _OCR_SPLIT_TAIL_RE.finditer(text):
        w2 = m.group(2)
        if w2 not in _VALID_SHORT_WORDS:
            s, e = m.start(), m.end()
            if any(not (e <= os_ or s >= oe_) for os_, oe_ in occupied):
                continue
            matches.append((s, e, m.group(0)))

    matches.sort(key=lambda item: item[0])
    hits: list[str] = []
    seen: set[str] = set()
    for _, _, pair in matches:
        if pair not in seen:
            seen.add(pair)
            hits.append(pair)
    return hits


def strip_duplicate_chapter_heading(ch_title: str, ch_body: str) -> str:
    """Remove a leading chapter heading line/paragraph if it duplicates <<<CHAPTER: ...>>>."""
    body = ch_body.strip()
    if not body or not ch_title.strip():
        return body

    variants = {_norm_alpha(ch_title)}
    m = re.match(r"^(?:chapter|part|section)\s+[0-9ivxlcdm]+[\s:.\-\u2013\u2014]+(.+)$", ch_title.strip(), re.I)
    if m and _norm_alpha(m.group(1)):
        variants.add(_norm_alpha(m.group(1)))
    variants.discard("")
    if not variants:
        return body

    parts = re.split(r"\n\s*\n", body, maxsplit=1)
    if _norm_alpha(parts[0]) in variants:
        return parts[1].strip() if len(parts) > 1 else ""

    lines = body.splitlines()
    if lines and _norm_alpha(lines[0]) in variants:
        return "\n".join(lines[1:]).lstrip("\n").strip()

    return body


def validate_cleaned_chunk(
    cleaned: str,
    chunk_text: str,
    running_headers: list[str],
    finish_reason: str | None,
) -> list[str]:
    """Return concrete quality defects found in a cleaned chunk (empty list = passed)."""
    issues: list[str] = []
    in_words = len(chunk_text.replace("<PAGEBREAK>", "").split())
    out_words = len(cleaned.split())
    ratio = out_words / max(1, in_words)

    if finish_reason == "length":
        issues.append("output truncated at max_tokens")
    if in_words >= 80 and ratio > 1.18:
        issues.append(f"output bloated ({in_words}w -> {out_words}w, {ratio:.2f}x)")
    if in_words >= 250 and out_words == 0 and not re.search(r"TABLE|FIG|REFERENCES|NOTES|INDEX", chunk_text, re.I):
        issues.append("unexpectedly empty output on prose chunk")

    if not cleaned:
        return issues

    hdr_norms = {_norm_alpha(h) for h in running_headers if len(_norm_alpha(h)) >= 5}
    hdr_exact = {h.strip() for h in running_headers if h.strip()}
    leaked_hdrs = []
    for ln in cleaned.split("\n"):
        s = ln.strip()
        if not s:
            continue
        if re.fullmatch(r"\d{1,3}", s):
            leaked_hdrs.append(s)
        elif len(s) <= 45 and _norm_alpha(s) in hdr_norms and (s.isupper() or s in hdr_exact):
            leaked_hdrs.append(s)
    if leaked_hdrs:
        issues.append(f"leaked running headers / page numbers: {leaked_hdrs[:4]}")

    fig_hits = _FIG_CAPTION_RE.findall(cleaned)
    if fig_hits:
        issues.append(f"leaked figure/table captions: {[m.strip() for m in fig_hits[:4]]}")

    for para in cleaned.split("\n\n"):
        plines = [l.strip() for l in para.split("\n") if l.strip()]
        caps_short = [l for l in plines if len(l.split()) <= 4 and l.isupper() and any(c.isalpha() for c in l)]
        if len(caps_short) >= 3:
            issues.append(f"leaked ALL-CAPS diagram/table labels: {caps_short[:4]}")
            break

    in_paras = [p for p in chunk_text.replace("<PAGEBREAK>", "").split("\n\n") if len(p.split()) >= 15]
    out_paras = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
    max_para_w = max((len(p.split()) for p in out_paras), default=0)
    if len(in_paras) >= 4 and max_para_w > 650:
        issues.append(f"paragraphs merged into a {max_para_w}-word wall of text (preserve paragraph breaks!)")

    bad_low = [p[:40] for p in out_paras[1:] if p and p[0].islower()]
    if bad_low:
        issues.append(f"broken lowercase-start paragraph(s): {bad_low[:2]}")

    spaced_hits = re.findall(r"(?:\b[A-Za-z]\s+){4,}[A-Za-z]\b", cleaned)
    if spaced_hits:
        issues.append(f"letter-spaced text not collapsed: {spaced_hits[:2]}")

    ocr_splits = find_ocr_word_splits(cleaned)
    if ocr_splits:
        issues.append(f"possible mid-word OCR split(s) — rejoin broken words: {ocr_splits[:5]}")

    return issues


def _endpoint_alive(api_url: str) -> bool:
    if not api_url:
        return False
    try:
        base = api_url.split("/v1/")[0]
        with urllib.request.urlopen(base + "/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def clean_chunk_with_guardrails(
    chunk_text: str,
    book_title: str,
    author: str,
    chapter_title: str,
    running_headers: list[str],
    prev_tail: str,
    next_head: str,
    api_url: str,
    model: str,
    fallback_api_url: str | None = None,
    fallback_model: str | None = None,
    api_key: str | None = None,
) -> tuple[str, dict]:
    in_words = len(chunk_text.replace("<PAGEBREAK>", "").split())
    max_tok = max(1024, min(6144, int(in_words * 2.1) + 256))

    attempts_plan = [(api_url, model, 0.05, False)]
    if fallback_api_url and fallback_model and (fallback_api_url != api_url or fallback_model != model):
        attempts_plan.append((fallback_api_url, fallback_model, 0.0, True))
    else:
        attempts_plan.append((api_url, model, 0.0, False))

    last_out = ""
    last_info: dict = {}
    retry_issues: list[str] | None = None
    escalated = False

    for attempt_idx, (ep_url, ep_model, temp, is_fallback) in enumerate(attempts_plan):
        if is_fallback and not _endpoint_alive(ep_url):
            ep_url, ep_model, is_fallback = api_url, model, False

        sys_p, usr_p = build_clean_prompts(
            chunk_text=chunk_text,
            book_title=book_title,
            author=author,
            chapter_title=chapter_title,
            running_headers=running_headers,
            prev_tail=prev_tail,
            next_head=next_head,
            retry_issues=retry_issues,
        )
        if is_fallback:
            escalated = True
            with _FALLBACK_LOCK:
                raw, meta = llm_chat(
                    api_url=ep_url,
                    model=ep_model,
                    system_prompt=sys_p,
                    user_prompt=usr_p,
                    max_tokens=max_tok,
                    temperature=temp,
                    enable_thinking=False,
                    api_key=api_key,
                )
        else:
            raw, meta = llm_chat(
                api_url=ep_url,
                model=ep_model,
                system_prompt=sys_p,
                user_prompt=usr_p,
                max_tokens=max_tok,
                temperature=temp,
                enable_thinking=False,
                api_key=api_key,
            )

        cleaned = postprocess_llm_chunk(raw)
        out_words = len(cleaned.split())
        ratio = out_words / max(1, in_words)
        finish = meta.get("finish_reason")
        timings = meta.get("timings") or {}
        issues = validate_cleaned_chunk(cleaned, chunk_text, running_headers, finish)

        last_out = cleaned
        last_info = {
            "in_words": in_words,
            "out_words": out_words,
            "ratio": ratio,
            "finish_reason": finish,
            "predicted_per_second": timings.get("predicted_per_second", 0.0),
            "predicted_n": timings.get("predicted_n", 0),
            "attempts": attempt_idx + 1,
            "model_used": ep_model,
            "escalated": escalated,
            "issues": issues,
        }
        if not issues:
            break
        retry_issues = issues

    flag = None
    if last_info.get("issues"):
        flag = f"remaining_issues ({last_info['model_used']}): " + "; ".join(last_info["issues"])
    elif escalated:
        flag = f"escalated_and_fixed ({last_info['model_used']})"
    elif in_words >= 150 and last_info["ratio"] < 0.55:
        flag = f"high_reduction (in={in_words}w -> out={last_info['out_words']}w, {last_info['ratio']:.2f}x)"

    last_info["flag"] = flag
    return last_out, last_info


def stitch_chunks(cleaned_chunks: list[str]) -> str:
    """Join adjacent cleaned chunks and repair mid-sentence blank lines left after removing tables/headers."""
    non_empty = [c.strip() for c in cleaned_chunks if c.strip()]
    if not non_empty:
        return ""
    out = non_empty[0]
    for nxt in non_empty[1:]:
        if out and nxt and not re.search(r'[.!?…:"\')\]]$', out.rstrip()) and nxt[0].islower():
            out = out.rstrip() + " " + nxt.lstrip()
        else:
            out = out.rstrip() + "\n\n" + nxt.lstrip()

    paras = [p.strip() for p in out.split("\n\n") if p.strip()]
    merged: list[str] = []
    for p in paras:
        if (
            merged
            and p
            and p[0].islower()
            and not re.search(r'[.!?…:"\u201d\u2019)\]]$', merged[-1].rstrip())
            and not merged[-1].startswith("<<<")
        ):
            merged[-1] = merged[-1].rstrip() + " " + p.lstrip()
        else:
            merged.append(p)
    return "\n\n".join(merged)


def _find_fast_server_cmd() -> list[str] | None:
    env_cmd = os.environ.get("MUMBLEBEE_FAST_SERVER_CMD")
    if env_cmd:
        return shlex.split(env_cmd)
    found = shutil.which("book-prep-server")
    return [found] if found else None


def maybe_start_fast_server(api_url: str) -> list[str] | None:
    if _endpoint_alive(api_url):
        return None
    cmd = _find_fast_server_cmd()
    if not cmd:
        return None
    print(f"Starting helper LLM server via {' '.join(cmd)} start...", flush=True)
    subprocess.run(cmd + ["start"], check=True)
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clean and chapter-split an EPUB/PDF/TXT book for TTS using an OpenAI-compatible LLM server."
    )
    parser.add_argument("input", type=Path, help="Input .epub, .pdf, .txt, or .md book file")
    parser.add_argument("-o", "--output", type=Path, help="Output cleaned text path (default: <stem>_tts_ready.txt)")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help=f"Primary fast LLM chat completions URL (default: {DEFAULT_API_URL})")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Primary fast model alias (default: {DEFAULT_MODEL})")
    parser.add_argument("--fallback-api-url", default=DEFAULT_FALLBACK_API_URL, help=f"Stronger fallback URL for failed chunks (default: {DEFAULT_FALLBACK_API_URL})")
    parser.add_argument("--fallback-model", default=DEFAULT_FALLBACK_MODEL, help=f"Stronger fallback model alias (default: {DEFAULT_FALLBACK_MODEL})")
    parser.add_argument("--api-key", help="Optional Bearer API key (or set OPENAI_API_KEY)")
    parser.add_argument("--no-fallback", action="store_true", help="Disable escalation to fallback model")
    parser.add_argument("--concurrency", type=int, default=4, help="Parallel chunk requests to primary LLM server (default: 4)")
    parser.add_argument("--chunk-words", type=int, default=1100, help="Target words per LLM chunk (default: 1100)")
    parser.add_argument("--keep-appendices", action="store_true", help="Include Appendices instead of dropping them as back matter")
    parser.add_argument("--max-chapters", type=int, default=0, help="Optional: only process first N chapters (for fast testing)")
    parser.add_argument("--resume", action="store_true", help="Resume from cached structure and cleaned chunks")
    parser.add_argument("--no-cache", action="store_true", help="Ignore existing cache and re-run from scratch")
    parser.add_argument("--stop-server", action="store_true", help="Stop helper server (if configured via MUMBLEBEE_FAST_SERVER_CMD or book-prep-server) when done")
    parser.add_argument(
        "--tts",
        choices=("none", "kokoro", "tiny-voca"),
        default="none",
        help="Optionally launch TTS conversion right after cleaning (default: none)",
    )
    parser.add_argument("--tts-output-dir", type=Path, default=Path("."), help="Output dir when using --tts")
    parser.add_argument("--voice", help="Voice override when using --tts")
    parser.add_argument("--speed", type=float, help="Speed override when using --tts")
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        return 2

    server_cmd = maybe_start_fast_server(args.api_url) or _find_fast_server_cmd()

    out_path = args.output or args.input.with_name(f"{args.input.stem}_tts_ready.txt")
    report_path = out_path.with_name(f"{args.input.stem}_prep_report.txt")
    cache_dir = out_path.parent / f".{args.input.stem}_prep_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting text from {args.input}...", flush=True)
    raw_text, source_meta = extract_text(args.input)
    lines = raw_text.split("\n")
    raw_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()[:12]
    meta_note = f", opf='{source_meta.get('title','')}' by '{source_meta.get('author','')}'" if source_meta else ""
    print(f"Extracted {len(lines):,} lines, {len(raw_text.split()):,} raw words (hash={raw_hash}{meta_note})", flush=True)

    struct_file = cache_dir / f"structure_{raw_hash}.json"
    if args.resume and not args.no_cache and struct_file.is_file():
        struct = json.loads(struct_file.read_text(encoding="utf-8"))
        print(f"Loaded cached structure: {struct['title']} by {struct['author']} ({len(struct['sections'])} sections)", flush=True)
    else:
        print(f"Stage 1: Analyzing book structure, front/back matter, and chapters via {args.model}...", flush=True)
        t_s1 = time.perf_counter()
        struct = detect_structure(lines, args.api_url, args.model, keep_appendices=args.keep_appendices, api_key=args.api_key)
        if not struct.get("title") and source_meta.get("title"):
            struct["title"] = source_meta["title"]
        if not struct.get("author") and source_meta.get("author"):
            struct["author"] = source_meta["author"]
        struct_file.write_text(json.dumps(struct, indent=2, ensure_ascii=False), encoding="utf-8")
        print(
            f"Structure detected in {time.perf_counter() - t_s1:.1f}s: "
            f"'{struct['title']}' by {struct['author']} | "
            f"body lines {struct['body_start_line']}..{struct['body_end_line']} (of {len(lines)}) | "
            f"{len(struct['sections'])} sections",
            flush=True,
        )

    sections = struct["sections"][: args.max_chapters] if args.max_chapters > 0 else struct["sections"]

    plan = []
    b_end = struct["body_end_line"]
    all_sections = struct["sections"]
    is_pdf = args.input.suffix.lower() == ".pdf"
    for idx, sec in enumerate(sections):
        start_l = sec["start_line"]
        end_l = (all_sections[idx + 1]["start_line"] - 1) if (idx + 1 < len(all_sections)) else b_end
        sec_lines = lines[start_l - 1 : end_l]
        chunks = split_section_into_chunks(
            sec_lines,
            target_words=args.chunk_words,
            max_words=int(args.chunk_words * 1.4),
            running_headers=struct.get("running_headers") or [],
            is_pdf=is_pdf,
        )
        plan.append((idx + 1, sec["title"], start_l, end_l, chunks))

    flat_tasks = []
    for p_idx, (sec_num, sec_title, s_l, e_l, chunks) in enumerate(plan):
        for c_idx, chunk in enumerate(chunks):
            prev_tail = ""
            if c_idx > 0:
                prev_lines = [l for l in chunks[c_idx - 1].split("\n") if l.strip() and l.strip() != "<PAGEBREAK>"]
                prev_tail = "\n".join(prev_lines[-3:])
            next_head = ""
            if c_idx + 1 < len(chunks):
                next_lines = [l for l in chunks[c_idx + 1].split("\n") if l.strip() and l.strip() != "<PAGEBREAK>"]
                next_head = "\n".join(next_lines[:3])
            flat_tasks.append((p_idx, sec_num, sec_title, c_idx, len(chunks), chunk, prev_tail, next_head))

    total_chunks = len(flat_tasks)
    workers = max(1, min(args.concurrency, total_chunks))
    print(f"Stage 2: Cleaning {len(plan)} section(s) across {total_chunks} chunk(s) [model={args.model}, concurrency={workers}]...", flush=True)
    for sec_num, sec_title, s_l, e_l, chs in plan:
        print(f"  {sec_num:02d}. {sec_title} (L{s_l}-L{e_l}, {len(chs)} chunks)", flush=True)

    t_s2 = time.perf_counter()
    sec_results: list[list[tuple[str, dict] | None]] = [[None] * len(chs) for *_, chs in plan]
    prog_lock = threading.Lock()
    total_pred_tokens = 0

    pbar = tqdm(total=total_chunks, desc=f"Cleaning ({args.model})", unit="chunk", dynamic_ncols=True)

    def run_task(task):
        nonlocal total_pred_tokens
        p_idx, sec_num, sec_title, c_idx, n_ch, chunk, prev_tail, next_head = task
        chunk_key = hashlib.sha256(f"{raw_hash}:{sec_num}:{c_idx}:{chunk}".encode("utf-8")).hexdigest()[:16]
        c_file = cache_dir / f"chunk_{sec_num:02d}_{c_idx:03d}_{chunk_key}.json"

        from_cache = False
        if args.resume and not args.no_cache and c_file.is_file():
            cached = json.loads(c_file.read_text(encoding="utf-8"))
            cleaned = cached["text"]
            info = cached["info"]
            from_cache = True
        else:
            cleaned, info = clean_chunk_with_guardrails(
                chunk_text=chunk,
                book_title=struct.get("title") or args.input.stem,
                author=struct.get("author") or "",
                chapter_title=sec_title,
                running_headers=struct.get("running_headers") or [],
                prev_tail=prev_tail,
                next_head=next_head,
                api_url=args.api_url,
                model=args.model,
                fallback_api_url=None if args.no_fallback else args.fallback_api_url,
                fallback_model=None if args.no_fallback else args.fallback_model,
                api_key=args.api_key,
            )
            c_file.write_text(json.dumps({"text": cleaned, "info": info}, ensure_ascii=False), encoding="utf-8")

        sec_results[p_idx][c_idx] = (cleaned, info)
        with prog_lock:
            if not from_cache:
                total_pred_tokens += info.get("predicted_n", 0)
            elapsed = max(time.perf_counter() - t_s2, 1e-6)
            agg_tps = total_pred_tokens / elapsed
            status = f"sec {sec_num}/{len(plan)}, {agg_tps:.1f} agg tok/s" if total_pred_tokens else f"sec {sec_num}/{len(plan)} (cached)"
            pbar.set_postfix_str(status, refresh=False)
            pbar.update(1)

    try:
        if workers == 1:
            for t in flat_tasks:
                run_task(t)
        else:
            with cf.ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(run_task, t) for t in flat_tasks]
                for f in futs:
                    f.result()
    finally:
        pbar.close()

    cleaned_sections: list[tuple[str, str]] = []
    report_lines = [
        f"Source: {args.input}",
        f"Model: {args.model} ({args.api_url})",
        f"Title: {struct.get('title', '')}",
        f"Author: {struct.get('author', '')}",
        f"Running headers: {struct.get('running_headers', [])}",
        f"Kept body lines: {struct['body_start_line']}..{struct['body_end_line']} / {len(lines)}",
        "",
    ]
    total_in_words = 0
    total_out_words = 0
    escalated_chunks = 0
    unresolved_chunks = 0

    for p_idx, (sec_num, sec_title, s_l, e_l, chunks) in enumerate(plan):
        sec_cleaned = []
        for c_idx, (cleaned, info) in enumerate(sec_results[p_idx]):
            sec_cleaned.append(cleaned)
            total_in_words += info.get("in_words", 0)
            total_out_words += info.get("out_words", 0)
            if info.get("escalated"):
                escalated_chunks += 1
            if info.get("issues"):
                unresolved_chunks += 1
            if info.get("flag"):
                report_lines.append(f"[FLAG] {sec_num:02d}. {sec_title} chunk {c_idx+1}/{len(chunks)}: {info['flag']}")
        stitched = strip_duplicate_chapter_heading(sec_title, stitch_chunks(sec_cleaned))
        if stitched:
            cleaned_sections.append((sec_title, stitched))
            sec_paras = [p.strip() for p in stitched.split("\n\n") if p.strip()]
            broken = [p[:60] for p in sec_paras if p and p[0].islower()]
            long_p = [len(p.split()) for p in sec_paras if len(p.split()) > 650]
            if broken:
                report_lines.append(
                    f"[AUDIT] {sec_num:02d}. {sec_title}: {len(broken)} lowercase-start paragraph(s): "
                    + "; ".join(repr(b) for b in broken[:3])
                )
            if long_p:
                report_lines.append(f"[AUDIT] {sec_num:02d}. {sec_title}: {len(long_p)} long >650w para(s): {long_p}")
            report_lines.append(f"Section {sec_num:02d}: {sec_title} -> {len(stitched.split()):,} words ({len(sec_paras)} paras)")
    report_lines.append(f"\nEscalated to fallback: {escalated_chunks}/{total_chunks} | Unresolved chunk flags: {unresolved_chunks}")

    out_parts = [f"<<<BOOK_TITLE: {struct.get('title') or args.input.stem}>>>"]
    if struct.get("author"):
        out_parts.append(f"<<<BOOK_AUTHOR: {struct['author']}>>>")
    out_parts.append("")
    for sec_title, sec_body in cleaned_sections:
        out_parts.append(f"<<<CHAPTER: {sec_title}>>>")
        out_parts.append("")
        out_parts.append(sec_body)
        out_parts.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_parts).strip() + "\n", encoding="utf-8")

    dt_s2 = time.perf_counter() - t_s2
    report_lines += [
        "",
        f"Total input words (body): {total_in_words:,}",
        f"Total cleaned words     : {total_out_words:,} ({100*total_out_words/max(1,total_in_words):.1f}% kept)",
        f"Cleaning time           : {dt_s2:.1f}s ({total_pred_tokens/max(dt_s2,1e-6):.1f} agg tok/s)",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(
        f"\nSaved {out_path} ({len(cleaned_sections)} chapters, {total_out_words:,} words, "
        f"{100*total_out_words/max(1,total_in_words):.1f}% of raw body) in {dt_s2:.1f}s\n"
        f"Audit report: {report_path}",
        flush=True,
    )

    if args.stop_server and server_cmd:
        subprocess.run(server_cmd + ["stop"], check=False)

    if args.tts == "kokoro":
        cmd = [
            sys.executable, "-m", "mumblebee.tts_kokoro",
            "--input", str(out_path),
            "--by-chapter",
            "--backend", "auto",
            "--output-dir", str(args.tts_output_dir),
        ]
        if args.voice:
            cmd += ["--voice", args.voice]
        if args.speed is not None:
            cmd += ["--speed", str(args.speed)]
        print("Running Kokoro per-chapter conversion:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    elif args.tts == "tiny-voca":
        tv_bin = shutil.which("mumblebee-tinyvoca") or shutil.which("tiny-voca-fast")
        cmd = ([tv_bin] if tv_bin else [sys.executable, "-m", "mumblebee.tts_tinyvoca"]) + [
            "--text-file", str(out_path),
            "--output", str(args.tts_output_dir / f"{args.input.stem}.opus"),
        ]
        if struct.get("title"):
            cmd += ["--title", struct["title"], "--album", struct["title"]]
        if struct.get("author"):
            cmd += ["--artist", struct["author"]]
        if args.voice:
            cmd += ["--voice", args.voice]
        if args.speed is not None:
            cmd += ["--speed", str(args.speed)]
        print("Running tiny-voca conversion:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
