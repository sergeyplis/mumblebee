"""Parallel Kokoro-82M audiobook CLI (ONNX CPU, Hybrid CPU+GPU, ROCm/CUDA, PyTorch).

Reads plain text or book2opus <<<CHAPTER: ...>>> scripts, synthesizes chunks in
parallel (longest-first LPT scheduling), streams PCM directly into background
ffmpeg libopus encoders with Vorbis metadata tags, and auto-fetches cover.png.
"""

import argparse
import concurrent.futures as cf
import json
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import warnings

os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from .cover import fetch_book_cover
from .hw import cache_dir, physical_cpu_count, process_running

MAX_CHUNK_CHARS = 500
DEFAULT_MAX_CHARS_PER_FILE = 20_000
SAMPLE_RATE = 24_000
ONNX_REPO = "onnx-community/Kokoro-82M-v1.0-ONNX"
ONNX_FILENAME = "onnx/model.onnx"

_BOOK_TITLE_RE = re.compile(r"^<<<BOOK_TITLE:\s*(.+?)\s*>>>$")
_BOOK_AUTHOR_RE = re.compile(r"^<<<BOOK_AUTHOR:\s*(.+?)\s*>>>$")
_CHAPTER_RE = re.compile(r"^<<<CHAPTER:\s*(.+?)\s*>>>$")


def parse_chaptered_book(text: str) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """Parse <<<BOOK_TITLE>>>, <<<BOOK_AUTHOR>>>, and <<<CHAPTER: ...>>> markers."""
    meta: dict[str, str] = {}
    chapters: list[tuple[str, list[str]]] = []
    cur_title: str | None = None
    cur_lines: list[str] = []

    for line in text.splitlines():
        s = line.strip()
        m = _BOOK_TITLE_RE.match(s)
        if m:
            meta["title"] = m.group(1).strip()
            continue
        m = _BOOK_AUTHOR_RE.match(s)
        if m:
            meta["author"] = m.group(1).strip()
            continue
        m = _CHAPTER_RE.match(s)
        if m:
            if cur_title is not None:
                chapters.append((cur_title, cur_lines))
            cur_title = m.group(1).strip()
            cur_lines = []
            continue
        if cur_title is not None:
            cur_lines.append(line)

    if cur_title is not None:
        chapters.append((cur_title, cur_lines))

    return meta, [(t, "\n".join(ls).strip()) for t, ls in chapters if "\n".join(ls).strip()]


def slugify_chapter(title: str, max_len: int = 64) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")
    return (slug[:max_len].rstrip("_")) or "Chapter"


def clean_markdown(text: str) -> str:
    text = re.sub(r"\*+(.*?)\*+", r"\1", text)
    text = re.sub(r"`+(.*?)`+", r"\1", text)
    return re.sub(r"\n+", " ", text).strip()


def split_long_paragraph(paragraph: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    if len(paragraph) <= max_chars:
        return [paragraph]

    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    chunks = []
    current = ""

    for sentence in sentences:
        pieces = [sentence] if len(sentence) <= max_chars else re.findall(r"\S+(?:\s+|$)", sentence)
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            candidate = f"{current} {piece}".strip()
            if current and len(candidate) > max_chars:
                chunks.append(current)
                current = piece
            else:
                current = candidate
            while len(current) > max_chars:
                chunks.append(current[:max_chars])
                current = current[max_chars:]
    if current:
        chunks.append(current)
    return chunks


def split_text(text: str) -> list[str]:
    raw_paragraphs = [clean_markdown(p) for p in text.split("\n\n") if p.strip()]
    merged_paragraphs = []
    buffer = ""
    for paragraph in raw_paragraphs:
        combined = f"{buffer}\n\n{paragraph}".strip() if buffer else paragraph
        words = combined.split()
        if len(words) > 5 or combined.endswith((".", "?", "!")):
            merged_paragraphs.append(combined)
            buffer = ""
        else:
            buffer = combined
    if buffer:
        merged_paragraphs.append(buffer)
    return [chunk for paragraph in merged_paragraphs for chunk in split_long_paragraph(paragraph)]


def make_batches(paragraphs: list[str], chunks_per_file: int, max_chars_per_file: int) -> list[list[str]]:
    batches = []
    batch = []
    batch_chars = 0
    for paragraph in paragraphs:
        sep = 2 if batch else 0
        if batch and (len(batch) >= chunks_per_file or batch_chars + sep + len(paragraph) > max_chars_per_file):
            batches.append(batch)
            batch = []
            batch_chars = 0
            sep = 0
        batch.append(paragraph)
        batch_chars += sep + len(paragraph)
    if batch:
        batches.append(batch)
    return batches


def output_prefix(input_path: Path, prefix: str | None, output_dir: Path, multiple_inputs: bool) -> Path:
    book_dir = Path(output_dir) / input_path.stem
    if prefix and multiple_inputs:
        name = f"{prefix}_{input_path.stem}"
    elif prefix:
        name = prefix
    else:
        name = input_path.stem
    return book_dir / name


def build_atempo_chain(speed: float) -> str | None:
    if abs(speed - 1.0) < 1e-4:
        return None
    stages = []
    rem = speed
    while rem > 2.0:
        stages.append("atempo=2.0")
        rem /= 2.0
    while rem < 0.5:
        stages.append("atempo=0.5")
        rem /= 0.5
    stages.append(f"atempo={rem:.6g}")
    return ",".join(stages)


def ensure_ort_model() -> Path:
    ort_model = cache_dir() / "kokoro_v1_opt_ort.onnx"
    if ort_model.exists():
        return ort_model
    onnx_src = hf_hub_download(ONNX_REPO, ONNX_FILENAME)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.optimized_model_filepath = str(ort_model)
    so.intra_op_num_threads = min(4, physical_cpu_count())
    so.inter_op_num_threads = 1
    ort.InferenceSession(onnx_src, sess_options=so, providers=["CPUExecutionProvider"])
    return ort_model


def make_ort_session(model_path: Path, intra_threads: int = 1) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.intra_op_num_threads = intra_threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.enable_mem_pattern = True
    so.enable_cpu_mem_arena = True
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    so.add_session_config_entry("session.set_denormal_as_zero", "1")
    return ort.InferenceSession(str(model_path), sess_options=so, providers=["CPUExecutionProvider"])


class FastKokoroEngine:
    """Unified high-speed Kokoro engine supporting onnx, hybrid (CPU+GPU), rocm/gpu, and pytorch."""

    def __init__(self, backend: str, voice: str, workers: int, gpu_streams: int = 4, lang_code: str = "a"):
        import torch
        from kokoro import KPipeline

        self.torch = torch
        self.backend = backend
        self.workers = workers
        self.gpu_streams = gpu_streams
        self.voice_name = voice

        warnings.filterwarnings("ignore", message=".*weight_norm.*")
        warnings.filterwarnings("ignore", message=".*dropout option adds dropout.*")
        warnings.filterwarnings("ignore", message=".*ComplexHalf support is experimental.*")

        self.g2p_pipe = KPipeline(lang_code=lang_code, model=False, repo_id="hexgrad/Kokoro-82M")
        self.pack_cpu = self.g2p_pipe.load_voice(voice)
        self.pack_np = self.pack_cpu.numpy()

        cfg_path = hf_hub_download("hexgrad/Kokoro-82M", "config.json")
        with open(cfg_path, encoding="utf-8") as f:
            self.vocab = json.load(f)["vocab"]

        self.ort_sessions: list[ort.InferenceSession] = []
        self.gpu_pipe = None
        self.pack_gpu = None
        self.streams: list = []

        use_ort = backend in ("onnx", "hybrid")
        use_gpu = backend in ("rocm", "gpu", "hybrid", "pytorch") and torch.cuda.is_available()

        if use_gpu and getattr(torch.version, "hip", None):
            # Critical on AMD ROCm / Strix Halo: MIOpen workspace fallback stalls Kokoro's 1D convs
            torch.backends.cudnn.enabled = False

        if use_ort:
            ort_path = ensure_ort_model()
            with cf.ThreadPoolExecutor(max_workers=workers) as ex:
                self.ort_sessions = list(ex.map(lambda _: make_ort_session(ort_path, 1), range(workers)))

        if use_gpu:
            try:
                self.gpu_pipe = KPipeline(lang_code=lang_code, device="cuda", repo_id="hexgrad/Kokoro-82M")
                self.pack_gpu = self.pack_cpu.to("cuda")
                if backend in ("rocm", "gpu", "hybrid"):
                    n_streams = gpu_streams if backend == "hybrid" else max(gpu_streams, 6)
                    self.streams = [torch.cuda.Stream() for _ in range(n_streams)]
                    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
                        _ = self.gpu_pipe.model("həloʊ", self.pack_gpu[4], speed=1.0)
                    torch.cuda.synchronize()
            except RuntimeError as e:
                if backend == "hybrid" and self.ort_sessions:
                    print(f"Warning: GPU init failed ({e}); continuing with ONNX CPU workers.", file=sys.stderr)
                    self.backend = "onnx"
                    self.gpu_pipe = None
                else:
                    raise
        elif backend == "pytorch":
            self.gpu_pipe = KPipeline(lang_code=lang_code, device="cpu", repo_id="hexgrad/Kokoro-82M")
            self.pack_gpu = self.pack_cpu

    def phonemize_batch(self, batch_text: str) -> list[str]:
        return [
            ps
            for _, ps, _ in self.g2p_pipe(
                batch_text, voice=self.voice_name, speed=1.0, split_pattern=r"\n+"
            )
            if ps
        ]

    def _make_ort_input(self, ps: str, model_speed: float) -> dict:
        ids = [i for i in (self.vocab.get(c) for c in ps) if i is not None]
        tokens = np.array([[0, *ids, 0]], dtype=np.int64)
        style = self.pack_np[min(len(ps) - 1, len(self.pack_np) - 1)].astype(np.float32)
        if style.ndim == 1:
            style = style[None, :]
        return {
            "input_ids": tokens,
            "style": style,
            "speed": np.array([model_speed], dtype=np.float32),
        }

    def synthesize_phonemes(
        self,
        phonemes_list: list[str],
        model_speed: float = 1.0,
        desc: str = "Synthesizing",
        show_progress: bool = True,
    ) -> list[np.ndarray]:
        torch = self.torch
        n = len(phonemes_list)
        if n == 0:
            return []

        if self.backend == "pytorch":
            results = []
            pbar = tqdm(
                total=sum(len(p) for p in phonemes_list),
                desc=desc,
                unit="ph",
                unit_scale=True,
                dynamic_ncols=True,
                disable=not show_progress,
            )
            t0 = time.perf_counter()
            done_samples = 0
            with torch.inference_mode():
                for idx, ps in enumerate(phonemes_list, 1):
                    ref_s = self.pack_gpu[min(len(ps) - 1, len(self.pack_gpu) - 1)]
                    audio = self.gpu_pipe.model(ps, ref_s, speed=model_speed)
                    arr = audio.detach().float().cpu().numpy() if isinstance(audio, torch.Tensor) else np.asarray(audio, dtype=np.float32)
                    results.append(arr)
                    done_samples += len(arr)
                    pbar.update(len(ps))
                    elapsed = max(time.perf_counter() - t0, 1e-6)
                    audio_s = done_samples / SAMPLE_RATE
                    pbar.set_postfix_str(f"{idx}/{n} ch, {audio_s/60:.1f}m, {audio_s/elapsed:.1f}x RT", refresh=False)
            pbar.close()
            return results

        lengths = [len(p) for p in phonemes_list]
        order = sorted(range(n), key=lambda i: -lengths[i])
        work_q: queue.Queue[int] = queue.Queue()
        for idx in order:
            work_q.put(idx)

        ort_inputs = [self._make_ort_input(ps, model_speed) for ps in phonemes_list] if self.ort_sessions else None
        results: list[np.ndarray | None] = [None] * n
        prog_lock = threading.Lock()
        done_chunks = 0
        done_samples = 0
        t0 = time.perf_counter()

        pbar = tqdm(
            total=sum(lengths),
            desc=desc,
            unit="ph",
            unit_scale=True,
            dynamic_ncols=True,
            disable=not show_progress,
        )

        def report(idx: int, arr: np.ndarray):
            nonlocal done_chunks, done_samples
            results[idx] = arr
            if show_progress:
                with prog_lock:
                    done_chunks += 1
                    done_samples += len(arr)
                    pbar.update(lengths[idx])
                    elapsed = max(time.perf_counter() - t0, 1e-6)
                    audio_s = done_samples / SAMPLE_RATE
                    pbar.set_postfix_str(
                        f"{done_chunks}/{n} ch, {audio_s/60:.1f}m, {audio_s/elapsed:.1f}x RT",
                        refresh=False,
                    )

        def cpu_worker(sess: ort.InferenceSession):
            while True:
                try:
                    idx = work_q.get_nowait()
                except queue.Empty:
                    break
                arr = sess.run(None, ort_inputs[idx])[0].squeeze().astype(np.float32, copy=False)
                report(idx, arr)

        def gpu_worker(stream):
            with torch.inference_mode(), torch.cuda.stream(stream):
                while True:
                    try:
                        idx = work_q.get_nowait()
                    except queue.Empty:
                        break
                    ps = phonemes_list[idx]
                    ref_s = self.pack_gpu[min(len(ps) - 1, len(self.pack_gpu) - 1)]
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        audio = self.gpu_pipe.model(ps, ref_s, speed=model_speed)
                    arr = audio.detach().float().cpu().numpy()
                    report(idx, arr)

        try:
            active_cpu = min(len(self.ort_sessions), n) if self.backend in ("onnx", "hybrid") else 0
            active_gpu = min(len(self.streams), n) if self.backend in ("rocm", "gpu", "hybrid") and self.gpu_pipe else 0
            with cf.ThreadPoolExecutor(max_workers=max(1, active_cpu + active_gpu)) as ex:
                futs = [ex.submit(cpu_worker, s) for s in self.ort_sessions[:active_cpu]]
                futs += [ex.submit(gpu_worker, st) for st in self.streams[:active_gpu]]
                for f in futs:
                    f.result()
            if active_gpu > 0:
                torch.cuda.synchronize()
        finally:
            pbar.close()

        return results


def encode_opus_streaming(
    waves: list[np.ndarray],
    destination: Path,
    speed: float,
    bitrate: str,
    metadata: dict[str, str],
):
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".opus.tmp",
            delete=False,
        ) as tf:
            tmp_path = Path(tf.name)

        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", "pipe:0",
        ]
        atempo = build_atempo_chain(speed)
        if atempo:
            cmd += ["-filter:a", atempo]
        cmd += [
            "-c:a", "libopus",
            "-b:a", bitrate,
            "-vbr", "on",
            "-compression_level", "10",
            "-application", "voip",
            "-ar", str(SAMPLE_RATE),
        ]
        for key, val in metadata.items():
            if val:
                cmd += ["-metadata", f"{key}={val}"]
        cmd += ["-f", "opus", str(tmp_path)]

        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        assert proc.stdin is not None
        try:
            for w in waves:
                if w.size == 0:
                    continue
                pcm16 = np.clip(w * 32767.0, -32768, 32767).astype(np.int16)
                proc.stdin.write(pcm16.tobytes())
        finally:
            proc.stdin.close()
        stderr_data = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg failed (exit {ret}): {stderr_data.strip()}")
        if not tmp_path.is_file() or tmp_path.stat().st_size == 0:
            raise RuntimeError("ffmpeg did not create a valid Opus file")

        os.replace(tmp_path, destination)
        tmp_path = None
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def convert_book(input_path: Path, args, multiple_inputs: bool, engine: FastKokoroEngine) -> bool:
    try:
        raw_lines = input_path.read_text(encoding="utf-8").splitlines(keepends=True)
        if args.lines:
            start_s, end_s = args.lines.split(":")
            start = max(1, int(start_s)) - 1 if start_s else 0
            end = int(end_s) if end_s else len(raw_lines)
            raw_lines = raw_lines[start:end]
        text = "".join(raw_lines)
    except (OSError, UnicodeError) as error:
        print(f"Error reading {input_path}: {error}", file=sys.stderr)
        return False

    book_meta, chapter_entries = parse_chaptered_book(text)
    use_chapters = bool(chapter_entries) and (args.by_chapter or not args.no_chapters)

    book_title = args.title or book_meta.get("title") or input_path.stem
    album_name = args.album or book_meta.get("title") or input_path.stem
    artist_name = args.artist or book_meta.get("author") or ""

    if use_chapters:
        book_dir = Path(args.output_dir) / slugify_chapter(args.prefix or album_name, max_len=80)
        book_dir.mkdir(parents=True, exist_ok=True)
        jobs: list[tuple[Path, str, str, int, int]] = []
        total_parts = len(chapter_entries)
        for idx, (ch_title, ch_body) in enumerate(chapter_entries, start=1):
            first_line = ch_body.splitlines()[0].strip() if ch_body.splitlines() else ""
            norm_first = re.sub(r"[^a-z0-9]+", "", first_line.lower())
            norm_title = re.sub(r"[^a-z0-9]+", "", ch_title.lower())
            if norm_title and not norm_first.startswith(norm_title[: min(16, len(norm_title))]):
                spoken_body = f"{ch_title}.\n\n{ch_body}"
            else:
                spoken_body = ch_body
            out_file = book_dir / f"{idx:02d}_{slugify_chapter(ch_title)}.opus"
            jobs.append((out_file, ch_title, spoken_body, idx, total_parts))
        print(
            f"\n{input_path}: {total_parts} chapter(s) ('{album_name}'"
            + (f" by {artist_name}" if artist_name else "")
            + f") -> {book_dir} [backend={engine.backend}, voice={args.voice}]"
        )
    else:
        paragraphs = split_text(text)
        if not paragraphs:
            print(f"Error: {input_path} contains no readable text.", file=sys.stderr)
            return False
        prefix = output_prefix(input_path, args.prefix, args.output_dir, multiple_inputs)
        book_dir = prefix.parent
        book_dir.mkdir(parents=True, exist_ok=True)
        batches = make_batches(paragraphs, args.chunks_per_file, args.max_chars_per_file)
        total_parts = len(batches)
        jobs = [
            (
                Path(f"{prefix}_Part_{idx:03d}.opus"),
                f"{book_title} - Part {idx:03d}",
                "\n\n".join(batch),
                idx,
                total_parts,
            )
            for idx, batch in enumerate(batches, start=1)
        ]
        print(
            f"\n{input_path}: {len(paragraphs)} logical paragraphs across {total_parts} output part(s) "
            f"[backend={engine.backend}, voice={args.voice}]"
        )

    if not args.no_cover:
        cover_path = book_dir / "cover.png"
        if not cover_path.is_file():
            try:
                stem_base = re.sub(r"_tts_ready$", "", input_path.stem)
                epub_cand = input_path.with_name(f"{stem_base}.epub")
                pdf_cand = input_path.with_name(f"{stem_base}.pdf")
                ok, info = fetch_book_cover(
                    title=album_name,
                    output_path=cover_path,
                    author=artist_name,
                    epub_source=epub_cand if epub_cand.is_file() else None,
                    pdf_fallback=pdf_cand if pdf_cand.is_file() else None,
                )
                print(f"Cover saved: {cover_path} ({info})" if ok else f"Cover lookup skipped ({info})", flush=True)
            except Exception as exc:
                print(f"Cover lookup skipped ({exc})", flush=True)

    t_book0 = time.perf_counter()
    total_audio_s = 0.0

    encode_futures: list[tuple[Path, cf.Future]] = []
    with cf.ThreadPoolExecutor(max_workers=2) as enc_pool:
        for out_file, part_title, part_text, part_idx, part_total in jobs:
            if args.resume and out_file.is_file() and out_file.stat().st_size > 0:
                print(f"Skipping existing {out_file}")
                continue
            try:
                part_chunks = split_text(part_text) if use_chapters else [part_text]
                phonemes_list = engine.phonemize_batch("\n\n".join(part_chunks))
                if not phonemes_list:
                    print(f"Warning: skipping empty section '{part_title}'", file=sys.stderr)
                    continue

                short_label = part_title if len(part_title) <= 28 else part_title[:25] + "..."
                waves = engine.synthesize_phonemes(
                    phonemes_list,
                    model_speed=args.model_speed,
                    desc=f"[{part_idx}/{part_total}] {short_label}",
                    show_progress=not args.no_progress,
                )
                total_audio_s += sum(len(w) for w in waves) / SAMPLE_RATE

                metadata = {
                    "title": part_title,
                    "artist": artist_name,
                    "album": album_name,
                    "track": f"{part_idx}/{part_total}",
                    "genre": args.genre or "Audiobook",
                    "comment": f"Generated by Kokoro-82M (voice={args.voice}, speed={args.speed:g})",
                }
                fut = enc_pool.submit(encode_opus_streaming, waves, out_file, args.speed, args.bitrate, metadata)
                encode_futures.append((out_file, fut))

                while len(encode_futures) >= 2:
                    done_path, done_fut = encode_futures.pop(0)
                    done_fut.result()
                    print(f"Saved {done_path} ({done_path.stat().st_size / (1024 * 1024):.2f} MB)")
            except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
                print(f"Error converting {input_path}, part {part_idx} ({part_title}): {error}", file=sys.stderr)
                return False

        for done_path, done_fut in encode_futures:
            try:
                done_fut.result()
                print(f"Saved {done_path} ({done_path.stat().st_size / (1024 * 1024):.2f} MB)")
            except Exception as error:
                print(f"Error encoding {done_path}: {error}", file=sys.stderr)
                return False

    wall = time.perf_counter() - t_book0
    if total_audio_s > 0:
        print(
            f"Finished {input_path.name}: {total_audio_s/60:.2f}m raw audio "
            f"({total_audio_s/args.speed/60:.2f}m at {args.speed:g}x) in {wall:.2f}s "
            f"({total_audio_s/wall:.1f}x real-time end-to-end)"
        )
    return True


def parse_args():
    default_workers = physical_cpu_count()
    parser = argparse.ArgumentParser(
        description="Convert text books (plain or <<<CHAPTER>>> marked) to tagged Opus audiobooks with Kokoro-82M."
    )
    parser.add_argument("--input", required=True, nargs="+", type=Path, help="one or more input TXT files")
    parser.add_argument("--lines", help="optional 1-based line range START:END for excerpts")
    parser.add_argument("--voice", default="af_heart", help="Kokoro voice name (default: af_heart)")
    parser.add_argument("--lang", default="a", help="Kokoro language code (default: a = American English)")
    parser.add_argument(
        "--backend",
        choices=("auto", "hybrid", "onnx", "rocm", "gpu", "pytorch"),
        default="auto",
        help="inference backend: auto (onnx CPU when llama-server is running, hybrid CPU+GPU otherwise), hybrid, onnx, rocm/gpu, pytorch",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto", help="device override")
    parser.add_argument("--by-chapter", action="store_true", help="output one Opus file per <<<CHAPTER: ...>>> section (auto-enabled when markers exist)")
    parser.add_argument("--no-chapters", action="store_true", help="ignore <<<CHAPTER: ...>>> markers and split by size instead")
    parser.add_argument("--workers", type=int, default=default_workers, help=f"ONNX CPU worker threads (default: {default_workers} physical cores)")
    parser.add_argument("--gpu-streams", type=int, default=4, help="concurrent GPU streams in hybrid/gpu mode (default: 4)")
    parser.add_argument("--chunks-per-file", type=int, default=100, help="max logical paragraphs per part (default: 100)")
    parser.add_argument("--max-chars-per-file", type=int, default=DEFAULT_MAX_CHARS_PER_FILE, help="max chars per part (default: 20000)")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed via ffmpeg atempo (default: 1.0)")
    parser.add_argument("--model-speed", type=float, default=1.0, help="native Kokoro model speed (default: 1.0)")
    parser.add_argument("--bitrate", default="32k", help="Opus bitrate (default: 32k; e.g. 24k, 32k, 48k)")
    parser.add_argument("--prefix", help="output filename/directory prefix")
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="output directory (default: current directory)")
    parser.add_argument("--resume", action="store_true", help="skip non-empty output parts that already exist")
    parser.add_argument("--title", help="base title tag (default: <<<BOOK_TITLE>>> or input filename stem)")
    parser.add_argument("--artist", help="artist / author tag (default: <<<BOOK_AUTHOR>>>)")
    parser.add_argument("--album", help="album / book title tag (default: <<<BOOK_TITLE>>> or input filename stem)")
    parser.add_argument("--genre", default="Audiobook", help="genre tag (default: Audiobook)")
    parser.add_argument("--no-cover", action="store_true", help="do not auto-fetch cover.png")
    parser.add_argument("--no-progress", action="store_true", help="disable tqdm progress bars")
    args = parser.parse_args()

    if args.chunks_per_file < 1 or args.max_chars_per_file < 1:
        parser.error("--chunks-per-file and --max-chars-per-file must be >= 1")
    if not 0.5 <= args.speed <= 100:
        parser.error("--speed must be between 0.5 and 100 for ffmpeg atempo")
    missing = [path for path in args.input if not path.is_file()]
    if missing:
        parser.error("input file not found: " + ", ".join(map(os.fspath, missing)))
    if shutil.which("ffmpeg") is None:
        parser.error("ffmpeg was not found on PATH")
    return args


def main() -> int:
    import torch

    args = parse_args()
    backend = args.backend
    if backend == "auto":
        if args.device == "cpu" or not torch.cuda.is_available() or process_running("llama-server"):
            backend = "onnx"
        else:
            backend = "hybrid"
    elif backend in ("rocm", "gpu", "hybrid") and not torch.cuda.is_available():
        print("Warning: GPU not detected; falling back to onnx CPU backend.", file=sys.stderr)
        backend = "onnx"

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    print(f"Initializing Kokoro [backend={backend}, workers={args.workers}, gpu={gpu_name}]...")
    t_init0 = time.perf_counter()
    try:
        engine = FastKokoroEngine(
            backend=backend,
            voice=args.voice,
            workers=args.workers,
            gpu_streams=args.gpu_streams,
            lang_code=args.lang,
        )
    except Exception as error:
        print(f"Error initializing Kokoro: {error}", file=sys.stderr)
        return 2
    print(f"Engine ready in {time.perf_counter() - t_init0:.2f}s (active backend: {engine.backend})")

    multiple_inputs = len(args.input) > 1
    failed = [p for p in args.input if not convert_book(p, args, multiple_inputs, engine)]
    if failed:
        print("\nFailed books: " + ", ".join(map(os.fspath, failed)), file=sys.stderr)
        return 1
    print(f"\nConverted {len(args.input)} book(s) successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
