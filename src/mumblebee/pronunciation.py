"""Stage 3: Automatic Kokoro-82M pronunciation audit using misaki G2P + local LLM.

Deterministically scans cleaned book text with Kokoro's actual `misaki` phonemizer
to find OOV / foreign / multi-word / CamelCase terms, translates misaki's shorthand
IPA into standard IPA so an LLM can read it reliably, asks the local LLM to flag
only genuine mispronunciations (with a strict wall-clock timeout so it never stalls
a build), and validates every phoneme against Kokoro's exact vocabulary.
"""

from collections import Counter
import json
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.request

_STOP_CAPS = {
    "The", "In", "For", "This", "That", "These", "Those", "It", "Its", "We", "Our", "They",
    "Their", "He", "His", "She", "Her", "As", "But", "And", "Or", "If", "When", "While",
    "Where", "What", "Which", "Who", "Why", "How", "Here", "There", "Now", "Then", "Thus",
    "Yet", "So", "No", "Not", "All", "Some", "Many", "Most", "Much", "More", "Each", "Every",
    "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "First",
    "Second", "Third", "Such", "Both", "Even", "Only", "Also", "Still", "Since", "Because",
    "Although", "Though", "However", "Moreover", "Furthermore", "Indeed", "Perhaps", "Chapter",
    "Section", "Part", "Preface", "Introduction", "Appendix", "Acknowledgments", "Summary",
    "From", "With", "Without", "Within", "Between", "Among", "After", "Before", "During",
    "Through", "Under", "Over", "Into", "Upon", "About", "Against", "Toward", "Towards",
    "Another", "Other", "Others", "Any", "May", "Can", "Will", "Would", "Should", "Could",
    "Must", "Let", "See", "Note", "Figure", "Table", "Professor", "Dr", "Mr", "Mrs", "Ms",
}

_FOREIGN_PARTICLES = {
    "y", "de", "del", "la", "le", "da", "di", "du", "van", "von",
    "el", "al", "los", "las", "san", "santa",
}

_KNOWN_CAMEL_OK = {
    "McGraw", "MacArthur", "McDonald", "McGill", "DeVore", "LeGros",
    "ChatGPT", "AlphaFold", "ImageNet", "DeepMind", "OpenAI", "YouTube", "GitHub",
    "JavaScript", "TypeScript", "PyTorch", "TensorFlow", "PowerPoint", "LinkedIn",
}

_SKIP_REASON_RE = re.compile(
    r"\b(?:skip(?:ping)?|is fine|is correct|is acceptable|no change|already correct|remove this)\b",
    re.IGNORECASE,
)


def kokoro_ipa_to_standard(k_ipa: str) -> str:
    """Convert Kokoro/misaki shorthand IPA into standard IPA so LLMs read it accurately."""
    s = re.sub(r"([bcdfghjklmnpqrstvwxyzʧʤʃʒθðŋɹɡT]+)([ˈˌ])", r"\2\1", k_ipa)
    for old, new in (
        ("A", "eɪ"),
        ("I", "aɪ"),
        ("O", "oʊ"),
        ("Q", "əʊ"),
        ("W", "aʊ"),
        ("Y", "ɔɪ"),
        ("ʧ", "tʃ"),
        ("ʤ", "dʒ"),
        ("ᵊ", "ə"),
        ("ᵻ", "ɪ"),
        ("T", "t"),
    ):
        s = s.replace(old, new)
    return s


def standard_ipa_to_kokoro(ipa: str, vocab: set[str] | None = None) -> tuple[str, list[str]]:
    """Convert standard IPA into Kokoro/misaki's exact phoneme alphabet and return (norm, invalid_chars)."""
    s = ipa.strip().strip("/")
    for old, new in (
        ("tʃ", "ʧ"),
        ("dʒ", "ʤ"),
        ("eɪ", "A"),
        ("aɪ", "I"),
        ("oʊ", "O"),
        ("əʊ", "Q"),
        ("aʊ", "W"),
        ("ɔɪ", "Y"),
        ("ər", "ɚ"),
        ("ɜr", "ɜɹ"),
        ("ɒ", "ɑ"),
        ("g", "ɡ"),
        ("r", "ɹ"),
        ("'", "ˈ"),
        (",", "ˌ"),
    ):
        s = s.replace(old, new)
    missing = sorted({ch for ch in s if ch not in vocab}) if vocab is not None else []
    return s, missing


def phonetic_skeleton(k_ipa: str) -> str:
    """Normalize away stress placement and minor allophonic/length notation to catch no-op edits."""
    s = re.sub(r"[ˈˌː\s]+", "", k_ipa)
    for old, new in (
        ("ɚ", "əɹ"),
        ("T", "t"),
        ("ᵊ", "ə"),
        ("ᵻ", "ɪ"),
        ("ɒ", "ɑ"),
        ("r", "ɹ"),
        ("g", "ɡ"),
    ):
        s = s.replace(old, new)
    return s


def _in_golds_with_inflection(word: str, golds: set[str]) -> bool:
    if word in golds or word.lower() in golds:
        return True
    wl = word.lower()
    if wl.endswith("ies") and (wl[:-3] + "y") in golds:
        return True
    if wl.endswith("es") and (wl[:-2] in golds or wl[:-1] in golds):
        return True
    if wl.endswith("s") and wl[:-1] in golds:
        return True
    if wl.endswith("ed") and (wl[:-2] in golds or wl[:-1] in golds or (wl.endswith("ied") and (wl[:-3] + "y") in golds)):
        return True
    if wl.endswith("ing") and (wl[:-3] in golds or (wl[:-3] + "e") in golds):
        return True
    if "-" in word and all((p in golds or p.lower() in golds) for p in word.split("-") if p):
        return True
    return False


def extract_candidates(text: str, pipe, max_candidates: int = 120) -> list[dict]:
    """Deterministically find OOV words, foreign/multi-word proper nouns, CamelCase, and key domain terms."""
    body = "\n".join(ln for ln in text.splitlines() if not ln.strip().startswith("<<<"))
    golds = set(pipe.g2p.lexicon.golds.keys())
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n\n+", body) if s.strip()]

    def find_snippet(term: str) -> str:
        pat = re.compile(rf"\b{re.escape(term)}\b")
        for s in sentences:
            m = pat.search(s)
            if m:
                start = max(0, m.start() - 45)
                end = min(len(s), m.end() + 45)
                return s[start:end].replace("\n", " ")
        return ""

    cand_scores: dict[str, tuple[str, int]] = {}

    # 1. CamelCase / internal-cap compounds (e.g. ForAllTokens; skip Mc/Mac surnames like McCarthy)
    for w, cnt in Counter(re.findall(r"\b[A-Za-z]*[a-z][A-Z][A-Za-z]*\b", body)).items():
        if w not in _KNOWN_CAMEL_OK and not re.match(r"^(?:Mc|Mac|De|La|Le|Di|Du|O')[A-Z]", w):
            cand_scores[w] = ("camel", cnt)

    # 2. Multi-word capitalized phrases with foreign particles or non-gold words
    mw_re = re.compile(
        r"\b([A-Z][a-zA-Z'’-]+(?:\s+(?:(?:de|del|la|le|da|di|du|van|von|y|el|al)\s+)?[A-Z][a-zA-Z'’-]+){1,2})\b"
    )
    for phrase, cnt in Counter(m.group(1) for m in mw_re.finditer(body)).items():
        words = phrase.split()
        if words[0] in _STOP_CAPS or words[-1] in _STOP_CAPS:
            continue
        has_foreign = any(w.lower() in _FOREIGN_PARTICLES for w in words)
        has_non_gold = any(not _in_golds_with_inflection(w, golds) for w in words if len(w) >= 2)
        if has_foreign or has_non_gold:
            cand_scores[phrase] = ("multiword", cnt)

    # 3. Single words missing from misaki's gold lexicon (OOV / silver / espeak fallback)
    word_counts = Counter(re.findall(r"\b[A-Za-z][a-zA-Z-]{2,}\b", body))
    for w, cnt in word_counts.items():
        if w in cand_scores or _in_golds_with_inflection(w, golds):
            continue
        if cnt >= 2 or (w[0].isupper() and cnt >= 1 and len(w) >= 4):
            cand_scores[w] = ("non_gold", cnt)

    # 4. High-frequency mid-sentence proper surnames & technical domain words even if in golds
    for w, cnt in Counter(re.findall(r"(?<=[a-z,;:\s])\b([A-Z][a-z]{3,})\b", body)).items():
        if w not in cand_scores and w not in _STOP_CAPS and cnt >= 4 and w.lower() not in golds:
            cand_scores[w] = ("freq_name", cnt)

    for w, cnt in word_counts.items():
        if w in cand_scores or w[0].isupper():
            continue
        if len(w) >= 8 and cnt >= 6 and w.endswith(("ous", "osis", "ism", "geny", "tomy", "logy", "morph", "pathy", "phile", "tene")):
            cand_scores[w] = ("domain_term", cnt)

    prio = {"camel": 4, "multiword": 3, "freq_name": 2, "domain_term": 2, "non_gold": 1}
    ordered = sorted(
        cand_scores.items(),
        key=lambda kv: (kv[1][1] * prio[kv[1][0]], prio[kv[1][0]], kv[1][1]),
        reverse=True,
    )[:max_candidates]

    results = []
    for term, (cat, cnt) in ordered:
        ipa = " ".join(ph for _, ph, _ in pipe(term, voice="af_heart")).strip()
        if not ipa:
            continue
        results.append(
            {
                "term": term,
                "category": cat,
                "count": cnt,
                "kokoro_ipa": ipa,
                "std_ipa": kokoro_ipa_to_standard(ipa),
                "snippet": find_snippet(term),
            }
        )
    return results


def filter_and_validate_fixes(
    raw_fixes: list[dict],
    cand_map: dict[str, dict],
    vocab: set[str] | None,
) -> dict[str, dict]:
    """Validate LLM-proposed fixes, rejecting invalid IPA, no-ops, and skip-comments."""
    overrides: dict[str, dict] = {}
    for item in raw_fixes:
        if not isinstance(item, dict):
            continue
        term = str(item.get("term", "")).strip()
        ftype = str(item.get("type", "ipa")).strip().lower()
        fix = str(item.get("fix", "")).strip()
        reason = str(item.get("reason", "")).strip()
        if not term or not fix or term not in cand_map or term in overrides:
            continue
        if _SKIP_REASON_RE.search(reason):
            continue
        old_ipa = cand_map[term]["kokoro_ipa"]
        if ftype == "ipa":
            norm_ipa, missing = standard_ipa_to_kokoro(fix, vocab)
            if missing or not norm_ipa:
                continue
            if phonetic_skeleton(norm_ipa) == phonetic_skeleton(old_ipa):
                continue
            overrides[term] = {"ipa": norm_ipa, "reason": reason}
        elif ftype == "respell":
            if fix.strip() == term:
                continue
            overrides[term] = {"respell": fix, "reason": reason}
    return overrides


def run_pronunciation_audit(
    text: str,
    book_title: str,
    api_url: str,
    model: str,
    *,
    api_key: str | None = None,
    timeout_s: float = 120.0,
    max_candidates: int = 120,
    batch_size: int = 45,
) -> tuple[dict[str, dict], dict]:
    """Run Stage 3 pronunciation audit with strict wall-clock timeout.

    Returns (overrides_dict, stats_dict). Never raises on timeout or missing kokoro.
    """
    t0 = time.monotonic()
    deadline = t0 + max(1.0, float(timeout_s))
    stats: dict = {
        "candidates": 0,
        "fixes": 0,
        "timed_out": False,
        "skipped_reason": None,
        "elapsed_s": 0.0,
    }

    try:
        from huggingface_hub import hf_hub_download
        from kokoro import KPipeline
    except ImportError:
        stats["skipped_reason"] = "kokoro/misaki not installed"
        stats["elapsed_s"] = round(time.monotonic() - t0, 2)
        return {}, stats

    try:
        cfg_path = hf_hub_download("hexgrad/Kokoro-82M", "config.json")
        vocab = set(json.loads(Path(cfg_path).read_text(encoding="utf-8"))["vocab"].keys())
        pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", model=False)
    except Exception as exc:
        stats["skipped_reason"] = f"kokoro init failed ({exc})"
        stats["elapsed_s"] = round(time.monotonic() - t0, 2)
        return {}, stats

    candidates = extract_candidates(text, pipe, max_candidates=max_candidates)
    stats["candidates"] = len(candidates)
    if not candidates:
        stats["elapsed_s"] = round(time.monotonic() - t0, 2)
        return {}, stats

    sys_prompt = """You are a meticulous American-English audiobook pronunciation editor auditing a TTS engine.
You will be given a list of book terms with the EXACT standard IPA pronunciation (`current_ipa`) that the TTS engine currently produces.

TASK:
Inspect EVERY term and its `current_ipa`. Return ONLY terms where `current_ipa` is CLEARLY WRONG, MANGLED, or AWKWARD for an educated American English audiobook listener.
Examples of REAL errors to catch:
- "Los Angeles" | current_ipa: `/ˈloʊz ˈeɪndʒəlz/` -> type "ipa", fix: `lˌɔs ˈændʒələs`
- "Ramon y Cajal" | current_ipa: `/ɹəˈmoʊn ˈwaɪ ˈkædʒəl/` -> type "ipa", fix: `ɹɑˈmoʊn i kɑˈhɑl`
- "Gehlen" | current_ipa: `/ˈdʒeɪlən/` -> type "ipa", fix: `ˈɡeɪlən`
- "tabu" | current_ipa: `/ˈtæbu/` -> type "ipa", fix: `təˈbuː`
- "ForAllTokens" | unspaced CamelCase -> type "respell", fix: `For All Tokens`

STRICT RULES:
- DO NOT include terms whose `current_ipa` is already natural and accurate in American English!
- Ignore minor transcription nuances (e.g., `/ɪ/` vs `/ə/`, `/ɔ/` vs `/ɑ/`, `/ɜɹ/` vs `/ɚ/`, missing length mark `ː`, or stress mark placement on 1-syllable words).
- If a term is already acceptable, OMIT IT from the JSON array completely (never include entries saying "skipping" or "acceptable").
- For CamelCase / squashed identifiers, use `"type": "respell"` with plain spaced English words.
- For pronunciation fixes, use `"type": "ipa"` with clean standard American English IPA.
- Return ONLY a JSON array (empty `[]` if everything in the batch sounds fine):
[{"term": "...", "type": "ipa"|"respell", "fix": "...", "reason": "short 5-10 word explanation"}]"""

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    raw_fixes: list[dict] = []
    for i in range(0, len(candidates), batch_size):
        remaining = deadline - time.monotonic()
        if remaining < 5.0:
            stats["timed_out"] = True
            break
        batch = candidates[i : i + batch_size]
        lines = [
            f"- {c['term']} (x{c['count']}, {c['category']}) | current_ipa: `/{c['std_ipa']}/` | ctx: {c['snippet']}"
            for c in batch
        ]
        user_prompt = f"Book: {book_title}\n\nTerms to audit:\n" + "\n".join(lines)
        payload = json.dumps(
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.0,
                "max_tokens": 2048,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        ).encode("utf-8")
        req = urllib.request.Request(api_url, data=payload, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=remaining) as resp:
                raw = json.loads(resp.read())["choices"][0]["message"]["content"].strip()
        except (TimeoutError, urllib.error.URLError, OSError) as exc:
            stats["timed_out"] = True
            print(f"Warning: pronunciation audit timed out / interrupted ({exc}); keeping partial results.", file=sys.stderr)
            break
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            try:
                items = json.loads(m.group(0))
                if isinstance(items, list):
                    raw_fixes.extend(items)
            except json.JSONDecodeError:
                pass

    cand_map = {c["term"]: c for c in candidates}
    overrides = filter_and_validate_fixes(raw_fixes, cand_map, vocab)
    stats["fixes"] = len(overrides)
    stats["elapsed_s"] = round(time.monotonic() - t0, 2)
    return overrides, stats
