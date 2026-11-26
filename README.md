# book2opus

Turn messy books (`.epub`, `.pdf`, OCR scans, `.txt`, `.md`) into clean, chapter-split, metadata-tagged `.opus` audiobooks with `cover.png` using **local OpenAI-compatible LLMs** and **parallel ONNX TTS** ([Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) and [tiny-voca](https://github.com/n4ze3m/tiny-voca)).

Designed for pleasant, uninterrupted audiobook listening in players like [Voice (F-Droid)](https://f-droid.org/packages/de.ph1b.audiobook/) and Audiobookshelf.

---

## Features & Tools

| CLI | Purpose | Typical Speed (16-core Zen 5 / Ryzen AI MAX+ 395) |
|---|---|---|
| **`book-prep`** (`book2opus-prep`) | Cleans `.epub`/`.pdf`/`.txt`/`.md` into a verbatim `*_tts_ready.txt` with `<<<CHAPTER: ...>>>` markers and an audit report. Removes running headers, page numbers, inline footnote markers/bodies, citations, figure/table captions, diagram labels, and back-matter notes/index while keeping author prose intact. | ~100–130 agg tok/s (~15 min for an 86k-word OCR PDF using 4-slot `Qwen3-4B` + Tier-2 escalation) |
| **`kokoro-fast`** (`book2opus-kokoro`) | Parallel Kokoro-82M TTS outputting one tagged `.opus` file per chapter + `cover.png`. Supports 16-core ONNX CPU (`0 MB` GPU), Hybrid CPU+GPU, and ROCm/CUDA. | **~31–35× RT** (ONNX CPU FP32) / **~40.5× RT** (Hybrid CPU+ROCm) |
| **`tiny-voca-fast`** (`book2opus-tinyvoca`) | Ultra-fast 4.5M-parameter parallel ONNX TTS (`M2`/`F2` voices) with pacing controls (`--speed`, `--sentence-pause`, `--paragraph-pause`), live progress bars, and tagged `.opus` output. | **~1,150–1,200× RT** synthesis (~850× RT end-to-end WAV) |
| **`fetch-book-cover`** (`book2opus-cover`) | Zero-API-key cover fetcher saving `cover.png`: embedded EPUB cover → Open Library → Google Books → PDF page 1 fallback. Also called automatically by `kokoro-fast`. | ~1–2 s |

---

## Install

Requires Python 3.10+, `ffmpeg` (with `libopus`), and `poppler-utils` (`pdftotext`, `pdftoppm`) for `.pdf` inputs.

```bash
git clone https://github.com/sergeyplis/book2opus.git
cd book2opus

# With Kokoro TTS:
pip install -e ".[kokoro]"

# Or with tiny-voca TTS:
pip install -e ".[tiny-voca]"

# Or both:
pip install -e ".[all]"
```

---

## Quickstart

### 1. Full Pipeline (EPUB / PDF → Cleaned Book → Per-Chapter Opus + Cover)

Point `book-prep` at any running OpenAI-compatible LLM server (`llama-server`, Ollama, vLLM, LM Studio, etc.):

```bash
# Single command (cleans text, writes MyBook_tts_ready.txt + audit report, then runs kokoro-fast):
book-prep MyBook.epub \
  --api-url http://127.0.0.1:8081/v1/chat/completions --model qwen3-4b-instruct \
  --fallback-api-url http://127.0.0.1:8080/v1/chat/completions --fallback-model qwen3.8-flash-next \
  --resume --tts kokoro --tts-output-dir ./books
```

### 2. Separate Execution

#### Step A — Preprocess & Audit Text Only (`book-prep`)
```bash
# Clean an EPUB or PDF (keeps Preface/Intro, drops endnotes/bibliography/index):
book-prep MyBook.pdf --resume

# Keep narrative Appendices too:
book-prep MyBook.pdf --keep-appendices --resume

# Fast sanity test on only the first 3 sections:
book-prep MyBook.pdf --max-chapters 3 --resume
```
Inspect `MyBook_prep_report.txt` and `MyBook_tts_ready.txt` before generating audio.

#### Step B1 — Convert to Per-Chapter Opus with Kokoro (`kokoro-fast`)
```bash
# Auto-detects <<<CHAPTER: ...>>>, title/author tags, cover.png, and safe backend:
kokoro-fast --input MyBook_tts_ready.txt --output-dir ./books

# Force 0-GPU ONNX CPU mode (~31-35x RT, leaves GPU free for LLMs):
kokoro-fast --input MyBook_tts_ready.txt --backend onnx --output-dir ./books

# Force Hybrid CPU + GPU mode (~40.5x RT when GPU is free):
kokoro-fast --input MyBook_tts_ready.txt --backend hybrid --output-dir ./books

# Customize voice, playback speed, bitrate:
kokoro-fast --input MyBook_tts_ready.txt --voice af_heart --speed 1.25 --bitrate 32k --output-dir ./books
```

#### Step B2 — Convert with `tiny-voca` (`tiny-voca-fast`)
```bash
tiny-voca-fast \
  --text-file MyBook_tts_ready.txt \
  --voice M2 --speed 0.88 --sentence-pause 0.15 --paragraph-pause 0.45 \
  --bitrate 32k --artist "Author Name" --album "Book Title" --title "Book Title" \
  -o ./books/MyBook_M2.opus
```

#### Step C — Standalone Cover Fetch (`fetch-book-cover`)
```bash
fetch-book-cover --title "Growing Young" --author "Ashley Montagu" --pdf GrowingYoung.pdf -o ./books/Growing_Young/cover.png
```

---

## Design Reasoning (The "Why" Behind the Architecture)

These empirical lessons drove every major architectural choice and are easy to lose track of when reading code years later:

### 1. Why LLM preprocessing instead of regex/grep
Real books—especially academic non-fiction and OCR'd PDFs—entangle noise semantically with prose. A regex cannot tell whether `century14` is a footnote number or a real quantity, cannot repair arbitrary OCR word splits (`ad u lth oo d`, `o f`), cannot stitch a sentence interrupted mid-clause by a 20-line floating table or 4 pages of skull diagrams, and cannot distinguish a keep-worthy Preface from 80 pages of endnotes. Decoupling **`book-prep` → `*_tts_ready.txt` → TTS** lets you audit the cleaned text once and re-synthesize cheaply with any voice, speed, or TTS engine.

### 2. Why two-tier LLM cleaning (fast 4B + fallback escalation) beats a single giant model
- On 256 GB/s unified memory, an ~85 GB frontier model is bandwidth-bound (~24–32 tok/s, ~70 min/book). Copy-editing is mostly verbatim transcription and does not need frontier reasoning on 90% of chunks.
- A 4B instruction model (`Qwen3-4B-Instruct-2507` Q4_K_M, 2.4 GB) running **4 continuous-batching slots** hits **~100–130 aggregate tok/s** (~15 min/book) and handles normal prose cleanly.
- **Why `ngram-simple` speculative decoding was rejected**: Despite ~95% verbatim copying, OCR line wraps every ~10 words shattered n-gram matches (9% draft acceptance), wasting verify compute and stalling 4-slot continuous batching (~35 tok/s vs ~72–130 tok/s with pure `--cont-batching`).
- **Why automated chunk validation + Tier-2 escalation**: On the ~10% of chunks dominated by multi-page diagrams (`Fig. 1–5`, `PROBOSCIS MONKEY`, running headers), a 4B model can fall into verbatim-copy mode. Checking every chunk for leaked headers/page numbers, `Fig.`/`TABLE` captions, ALL-CAPS diagram clusters, merged >650-word walls, letter-spaced text (`A n n a ...`), and broken lowercase starts lets `book-prep` automatically escalate *only* failing chunks to a stronger fallback model (`:8080`).

### 3. Critical PDF & Chunking Traps We Hit and Fixed
- **Never use `str.splitlines()` on PDFs**: Python's `splitlines()` silently splits on and strips `\f` form-feed page breaks, blinding chapter detection. Always use `split("\n")`.
- **Never split chunks on `\f` page breaks**: Printed page breaks almost always cut mid-sentence (or right across a multi-page table). Splitting at `\f` causes small LLMs to hallucinate an ending for the dangling sentence while the next chunk opens mid-air (`maximus (the large muscle...)`). Keep `<PAGEBREAK>` *inside* chunks and split only after blocks ending in terminal sentence punctuation whose next real body line (skipping headers and page numbers) starts cleanly.
- **PDF paragraph restoration**: `pdftotext` (non-layout, required to de-column 2-page spreads) omits blank lines between paragraphs, causing LLMs to merge whole pages into 1,500-word walls. Re-inserting blank lines after short sentence-ending lines restores paragraphs before chunking.
- **Prefer `.epub` over `.pdf` whenever available**: Pure-stdlib EPUB extraction gives exact reading-order spine chapters, Dublin Core `title`/`author`, clean structural pre-stripping of `<figcaption>`, `<aside epub:type="footnote">`, and `<sup>` note markers (preserving ordinals like `2nd`), plus the embedded publisher cover.

### 4. Why Parallel ONNX Runtime CPU Beats (or Ties) GPU for TTS
- **`tiny-voca` (4.5M params)**: Kernel-launch bound on GPU (~98–343× RT on Radeon 8060S). Running 1 single-threaded ONNX session per physical core (`16w × 1T`, pre-optimized `--optimized` + `ORT_ENABLE_ALL`, LPT scheduling, `dizione` frontend) hits **~1,150–1,200× RT** in lossless FP32 with 0 MB GPU.
- **`Kokoro-82M` on AMD ROCm / Strix Halo (`gfx1151`)**: Default PyTorch ROCm (`cudnn=True`) triggers a severe MIOpen workspace fallback (`705 MB required, provided 0`) on Kokoro's 1D transposed convs, stalling at **< 1× RT** even in `MIOPEN_FIND_MODE=FAST`. Setting `torch.backends.cudnn.enabled = False` + FP16 + 4–6 streams reaches ~31× RT (8 streams thrashes GTT). Meanwhile, **16-core ONNX Runtime CPU FP32** hits the same **~31–35× RT** with **0 MB GPU** (safe alongside a resident LLM), and **Hybrid (CPU ONNX + 4 GPU streams)** reaches **~40.5× RT** when the GPU is free.
- **Why INT8 ONNX was rejected for both TTS models**: On `tiny-voca`, INT8 MatMul gave only +15% speed while altering durations on 115/119 sentences; on `Kokoro-82M`, INT8 was *slower* (21.4× vs 31.1× RT) and shifted 27/28 chunk lengths. Lossless FP32 won on both quality and reliability.

### 5. Why Opus (`32k` VBR, `-application voip`, 24 kHz mono)
Best speech intelligibility per megabyte (~13–14 MB/hour — ~3–4× smaller than MP3, ~12× smaller than WAV) with native Vorbis tags (`title`, `artist`, `album`, `track`, `genre`, `comment`) and streaming PCM-to-`ffmpeg` piping that avoids multi-GB temp WAV buffers.

---

## Running Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```
