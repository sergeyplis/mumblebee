"""High-speed parallel ONNX CLI for tiny-voca (4.5M TTS, Opus default + tags).

Uses the optimized ONNX graph pre-compiled with ONNX Runtime (ORT_ENABLE_ALL),
native dizione phonemizer (TINY_VOCA_FRONTEND=own), overlapped model/text init,
LPT-scheduled parallel sentence synthesis across physical CPU cores (FP32
lossless, 0 MB GPU), live progress bars, pacing controls, and streaming Opus export.
"""

import argparse
import concurrent.futures as cf
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import time

import numpy as np
import onnxruntime as ort
import soundfile as sf
from huggingface_hub import snapshot_download
from tqdm import tqdm

from .hw import cache_dir, physical_cpu_count

PEAK = 0.99
ENGLISH_ONLY = "tiny-voca only reads English for now"
_SPACE = re.compile(r"[\s\ufeff]+")
_SENTENCE_END = re.compile(r"""(?:(?<=[.!?…])|(?<=[.!?…]["')\]]))\s+(?=["'(\[]?[A-Z0-9])""")
_NOT_AN_END = re.compile(
    r"(?:\b(?:mr|mrs|ms|dr|prof|sr|jr|st|vs|etc|no|inc|ltd|co|e\.g|i\.e|a\.m|p\.m)|\b[A-Z])\.$",
    re.IGNORECASE | re.ASCII,
)
_CLAUSE_END = re.compile(r"(?<=[,;:—])\s+")
_TEXT_END = re.compile(r"[\s\ufeff]+$")
_CLOSING = "\"'”’)]»"

SAMPLE_RATE = 24_000
MAX_TOKENS = 500
VOICES = {"F2": 0, "M2": 1}
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n+")
_MARKER_LINE_RE = re.compile(r"^<<<(?:BOOK_TITLE|BOOK_AUTHOR|CHAPTER):.*>>>$")


def resolve_weights_dir(explicit: Path | None = None) -> Path:
    if explicit:
        return explicit
    env_dir = os.environ.get("TINY_VOCA_WEIGHTS_DIR")
    if env_dir:
        return Path(env_dir).expanduser()
    # Check common local install locations, otherwise auto-download into XDG cache
    for cand in (
        cache_dir() / "tiny-voca-weights",
        Path.home() / "soft" / "tiny-voca" / "weights",
    ):
        if (cand / "phonemizer").is_dir() and ((cand / "onnx" / "model.onnx").is_file() or (cand / "tiny-voca.pt").is_file()):
            return cand
    dst = cache_dir() / "tiny-voca-weights"
    print(f"Downloading n4ze3m/tiny-voca weights to {dst}...", file=sys.stderr, flush=True)
    snapshot_download("n4ze3m/tiny-voca", local_dir=str(dst))
    return dst


def ensure_ort_model(weights_dir: Path) -> Path:
    ort_model = cache_dir() / "tiny_voca_opt_ort.onnx"
    if ort_model.exists():
        return ort_model

    # Prefer --optimized export if present or exportable from tiny-voca.pt
    opt_model = weights_dir / "onnx" / "model_opt.onnx"
    if not opt_model.is_file():
        pt_ckpt = weights_dir / "tiny-voca.pt"
        if pt_ckpt.is_file():
            try:
                opt_model = cache_dir() / "tiny_voca_opt.onnx"
                if not opt_model.is_file():
                    env = os.environ.copy()
                    env["TINY_VOCA_FRONTEND"] = "own"
                    env["DIZIONE_DATA"] = str(weights_dir / "phonemizer")
                    subprocess.run(
                        [
                            sys.executable, "-m", "tiny_voca.export",
                            "--checkpoint", str(pt_ckpt),
                            "--output", str(opt_model),
                            "--optimized",
                        ],
                        env=env,
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
            except Exception:
                opt_model = weights_dir / "onnx" / "model.onnx"
        else:
            opt_model = weights_dir / "onnx" / "model.onnx"

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.optimized_model_filepath = str(ort_model)
    so.intra_op_num_threads = min(4, physical_cpu_count())
    so.inter_op_num_threads = 1
    ort.InferenceSession(str(opt_model), sess_options=so, providers=["CPUExecutionProvider"])
    return ort_model


def sentences(text: str) -> list[str]:
    parts, current = [], ""
    for piece in _SENTENCE_END.split(_SPACE.sub(" ", text).strip()):
        current = f"{current} {piece}" if current else piece
        if not _NOT_AN_END.search(current):
            parts.append(current)
            current = ""
    if current:
        parts.append(current)
    return [part for part in parts if any(char.isalnum() for char in part)]


def finished(text: str) -> str:
    body = _TEXT_END.sub("", text)
    if not body:
        return text
    last = _SPACE.split(body)[-1]
    if any(char.isalnum() for char in last):
        marked = last.rstrip(_CLOSING)[-1:] in tuple(".!?…,;:")
    else:
        marked = any(char in ".!?…" for char in last) or last in (",", ";", ":")
    return text if marked else body + "."


def shorter(text: str) -> list[str]:
    pieces = _CLAUSE_END.split(text)
    if len(pieces) > 1:
        return pieces
    words = text.split(" ")
    if len(words) < 2:
        return []
    half = (len(words) + 1) // 2
    return [" ".join(words[:half]), " ".join(words[half:])]


def make_session(model_path: Path, intra_threads: int = 1) -> ort.InferenceSession:
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


def _strip_markers_for_speech(text: str) -> str:
    """Convert <<<CHAPTER: Title>>> to spoken heading and drop BOOK_TITLE/BOOK_AUTHOR headers."""
    out = []
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("<<<CHAPTER:") and s.endswith(">>>"):
            out.append(s[len("<<<CHAPTER:") : -3].strip() + ".")
        elif _MARKER_LINE_RE.match(s):
            continue
        else:
            out.append(ln)
    return "\n".join(out)


def prepare_utterances(text: str, paragraph_pause: float) -> list[tuple[object, bool]]:
    from tiny_voca.text import NothingToPronounce, phonemize

    text = _strip_markers_for_speech(text)
    items: list[tuple[object, bool]] = []

    def collect(piece: str, ends_paragraph: bool):
        try:
            phones = phonemize(piece)
        except NothingToPronounce:
            return
        if len(phones) <= MAX_TOKENS:
            items.append((phones, ends_paragraph))
            return
        pieces = shorter(piece)
        if not pieces:
            raise ValueError(f"one word is too long to speak: {piece[:40]!r}")
        for idx, part in enumerate(pieces):
            collect(part, ends_paragraph and idx == len(pieces) - 1)

    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT.split(text) if p.strip()] if paragraph_pause > 0 else [text]
    for p_idx, para in enumerate(paragraphs):
        sents = sentences(finished(para))
        for s_idx, sent in enumerate(sents):
            is_para_end = (paragraph_pause > 0) and (s_idx == len(sents) - 1) and (p_idx < len(paragraphs) - 1)
            collect(sent, is_para_end)
    return items


def synthesize_parallel(
    text: str,
    weights_dir: Path,
    voice: str = "M2",
    speed: float = 1.0,
    pitch_shift: float = 0.0,
    pitch_range: float = 1.0,
    workers: int | None = None,
    sentence_pause: float = 0.0,
    paragraph_pause: float = 0.3,
    show_progress: bool = True,
):
    os.environ["TINY_VOCA_FRONTEND"] = "own"
    os.environ["DIZIONE_DATA"] = str(weights_dir / "phonemizer")
    from tiny_voca.text import NothingToPronounce

    workers = workers or physical_cpu_count()
    t_total0 = time.perf_counter()
    model_path = ensure_ort_model(weights_dir)

    if show_progress:
        print("Phonemizing text and loading ONNX workers...", file=sys.stderr, flush=True)

    t_prep0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=workers + 1) as ex:
        ph_future = ex.submit(prepare_utterances, text, paragraph_pause)
        sess_futures = [ex.submit(make_session, model_path, 1) for _ in range(workers)]
        items = ph_future.result()
        if not items:
            raise NothingToPronounce(f"nothing in the text could be read: {ENGLISH_ONLY}")
        active_workers = min(workers, len(items))
        sessions = [f.result() for f in sess_futures[:active_workers]]
    t_prep = time.perf_counter() - t_prep0

    v_arr = np.array([VOICES[voice]], dtype=np.int64)
    sp_arr = np.array([speed], dtype=np.float32)
    ps_arr = np.array([pitch_shift], dtype=np.float32)
    pr_arr = np.array([pitch_range], dtype=np.float32)

    inputs = [
        {
            "tokens": np.array([phones.ids], dtype=np.int64),
            "stress": np.array([phones.stress], dtype=np.int64),
            "voice": v_arr,
            "speed": sp_arr,
            "pitch_shift": ps_arr,
            "pitch_range": pr_arr,
        }
        for phones, _ in items
    ]
    token_lens = [inp["tokens"].shape[1] for inp in inputs]
    order = sorted(range(len(inputs)), key=lambda i: -token_lens[i])
    work_q: queue.Queue[int] = queue.Queue()
    for idx in order:
        work_q.put(idx)

    waves: list[np.ndarray | None] = [None] * len(inputs)
    prog_lock = threading.Lock()
    done_utterances = 0
    done_samples = 0
    t_synth0 = time.perf_counter()

    pbar = tqdm(
        total=sum(token_lens),
        desc=f"Synthesizing ({voice}, {speed:g}x)",
        unit="tok",
        unit_scale=True,
        dynamic_ncols=True,
        disable=not show_progress,
    )

    def worker_fn(sess: ort.InferenceSession):
        nonlocal done_utterances, done_samples
        while True:
            try:
                idx = work_q.get_nowait()
            except queue.Empty:
                break
            out = sess.run(["audio"], inputs[idx])[0][0]
            if not np.isfinite(out).all():
                raise RuntimeError(f"model produced non-finite audio for utterance {idx}")
            waves[idx] = out
            if show_progress:
                with prog_lock:
                    done_utterances += 1
                    done_samples += len(out)
                    pbar.update(token_lens[idx])
                    elapsed = max(time.perf_counter() - t_synth0, 1e-6)
                    audio_s = done_samples / SAMPLE_RATE
                    pbar.set_postfix_str(
                        f"{done_utterances}/{len(inputs)} utt, {audio_s/60:.1f}m audio, {audio_s/elapsed:.0f}x RT",
                        refresh=False,
                    )

    try:
        if active_workers == 1:
            worker_fn(sessions[0])
        else:
            with cf.ThreadPoolExecutor(max_workers=active_workers) as ex:
                for f in [ex.submit(worker_fn, s) for s in sessions]:
                    f.result()
    finally:
        pbar.close()
    t_synth = time.perf_counter() - t_synth0

    t_post0 = time.perf_counter()
    sent_pause_n = int(round(sentence_pause * SAMPLE_RATE)) if sentence_pause > 0 else 0
    para_pause_n = int(round(paragraph_pause * SAMPLE_RATE)) if paragraph_pause > 0 else 0
    loudest = 0.0
    total_samples = 0
    last_idx = len(items) - 1
    for idx, (_, is_para_end) in enumerate(items):
        w = waves[idx]
        if w.size:
            peak_w = float(np.max(np.abs(w)))
            if peak_w > loudest:
                loudest = peak_w
        total_samples += len(w)
        if idx < last_idx:
            total_samples += para_pause_n if is_para_end else sent_pause_n

    scale = (PEAK / loudest) if loudest > PEAK else 1.0
    t_post = time.perf_counter() - t_post0
    t_total = time.perf_counter() - t_total0
    return (waves, items, scale, total_samples, sent_pause_n, para_pause_n), {
        "utterances": len(items),
        "workers": active_workers,
        "prep_s": t_prep,
        "synth_s": t_synth,
        "post_s": t_post,
        "total_s": t_total,
        "audio_s": total_samples / SAMPLE_RATE,
        "peak": loudest,
    }


def iter_audio_chunks(waves, items, scale: float, sent_pause_n: int, para_pause_n: int, chunk_samples: int = 240_000):
    sent_sil = np.zeros(sent_pause_n, dtype=np.float32) if sent_pause_n > 0 else None
    para_sil = np.zeros(para_pause_n, dtype=np.float32) if para_pause_n > 0 else None
    buf: list[np.ndarray] = []
    buf_n = 0
    last_idx = len(items) - 1
    for idx, (_, is_para_end) in enumerate(items):
        w = waves[idx] if scale == 1.0 else (waves[idx] * scale)
        buf.append(w)
        buf_n += len(w)
        if idx < last_idx:
            sil = para_sil if is_para_end else sent_sil
            if sil is not None:
                buf.append(sil)
                buf_n += len(sil)
        if buf_n >= chunk_samples:
            yield np.concatenate(buf)
            buf.clear()
            buf_n = 0
    if buf:
        yield np.concatenate(buf)


def save_audio_streaming(
    path: Path,
    bundle,
    subtype: str = "PCM_16",
    bitrate: str = "32k",
    metadata: dict[str, str] | None = None,
    show_progress: bool = True,
):
    waves, items, scale, total_samples, sent_pause_n, para_pause_n = bundle
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    total_sec_int = max(1, int(round(total_samples / SAMPLE_RATE)))
    written_samples = 0
    reported_sec = 0

    pbar = tqdm(
        total=total_sec_int,
        desc=f"Writing {path.name}",
        unit="s",
        dynamic_ncols=True,
        disable=not show_progress,
    )

    def advance(n_samples: int):
        nonlocal written_samples, reported_sec
        written_samples += n_samples
        sec = min(total_sec_int, int(written_samples // SAMPLE_RATE))
        if sec > reported_sec:
            pbar.update(sec - reported_sec)
            reported_sec = sec

    try:
        if ext in (".wav", ".flac", ".ogg") and not (ext == ".ogg" and metadata):
            with sf.SoundFile(
                str(path),
                mode="w",
                samplerate=SAMPLE_RATE,
                channels=1,
                subtype=subtype if ext == ".wav" else None,
            ) as f:
                for chunk in iter_audio_chunks(waves, items, scale, sent_pause_n, para_pause_n):
                    f.write(chunk)
                    advance(len(chunk))
        else:
            cmd = [
                "ffmpeg", "-y", "-v", "error",
                "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", "pipe:0",
            ]
            if ext == ".opus":
                cmd += [
                    "-c:a", "libopus",
                    "-b:a", bitrate,
                    "-vbr", "on",
                    "-compression_level", "10",
                    "-application", "voip",
                    "-ar", str(SAMPLE_RATE),
                ]
            elif ext in (".mp3", ".m4a", ".ogg") and bitrate:
                cmd += ["-b:a", bitrate]
            if metadata:
                for key, val in metadata.items():
                    if val:
                        cmd += ["-metadata", f"{key}={val}"]
            cmd.append(str(path))

            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            assert proc.stdin is not None
            try:
                for chunk in iter_audio_chunks(waves, items, scale, sent_pause_n, para_pause_n):
                    pcm16 = np.clip(chunk * 32767.0, -32768, 32767).astype(np.int16)
                    proc.stdin.write(pcm16.tobytes())
                    advance(len(chunk))
            finally:
                proc.stdin.close()
            ret = proc.wait()
            if ret != 0:
                raise subprocess.CalledProcessError(ret, cmd)
        if reported_sec < total_sec_int:
            pbar.update(total_sec_int - reported_sec)
    finally:
        pbar.close()


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    default_workers = physical_cpu_count()
    parser = argparse.ArgumentParser(prog=prog, description="Fast parallel ONNX tiny-voca TTS CLI (Opus default + tags)")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="Text to synthesize")
    src.add_argument("--text-file", type=Path, help="UTF-8 text file to synthesize")
    parser.add_argument("--lines", help="Optional 1-based line range START:END when using --text-file")
    parser.add_argument("--voice", choices=("F2", "M2"), default="M2", help="Voice: F2 (female) or M2 (male)")
    parser.add_argument("--weights-dir", type=Path, help="Path to n4ze3m/tiny-voca weights (auto-downloads if omitted)")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output audio path (default: <stem>.opus)")
    parser.add_argument("--bitrate", default="32k", help="Target bitrate for Opus/compressed formats (default: 32k)")
    parser.add_argument("--speed", type=float, default=1.0, help="Speaking rate (default: 1.0; try 0.85-0.90 for calmer pace)")
    parser.add_argument("--pitch-shift", type=float, default=0.0, help="Log-F0 shift (0.058 ~ 1 semitone)")
    parser.add_argument("--pitch-range", type=float, default=1.0, help="Pitch contour range multiplier")
    parser.add_argument("--sentence-pause", type=float, default=0.0, help="Extra seconds of silence between sentences (default: 0.0; try 0.15)")
    parser.add_argument("--paragraph-pause", type=float, default=0.3, help="Seconds of silence between paragraphs (default: 0.3)")
    parser.add_argument("--workers", type=int, default=default_workers, help=f"Parallel ONNX workers (default: {default_workers} physical cores)")
    parser.add_argument("--float-wav", action="store_true", help="Write 32-bit float WAV instead of 16-bit PCM WAV")
    parser.add_argument("--title", help="Track title tag (default: output stem)")
    parser.add_argument("--artist", help="Artist / author tag")
    parser.add_argument("--album", help="Album / book title tag (default: input stem)")
    parser.add_argument("--track", help="Track number tag (e.g. '1/12')")
    parser.add_argument("--genre", default="Audiobook", help="Genre tag (default: Audiobook)")
    parser.add_argument("--comment", help="Comment tag")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars")
    args = parser.parse_args(argv)

    if args.text is not None:
        text = args.text
        default_stem = "tiny_voca_output"
    else:
        raw_lines = args.text_file.read_text(encoding="utf-8").splitlines(keepends=True)
        if args.lines:
            start_s, end_s = args.lines.split(":")
            start = max(1, int(start_s)) - 1 if start_s else 0
            end = int(end_s) if end_s else len(raw_lines)
            raw_lines = raw_lines[start:end]
        text = "".join(raw_lines)
        default_stem = args.text_file.stem

    output_path = args.output if args.output is not None else Path(f"{default_stem}.opus")
    if not output_path.suffix:
        output_path = output_path.with_suffix(".opus")

    weights_dir = resolve_weights_dir(args.weights_dir)
    metadata = {
        "title": args.title or output_path.stem,
        "artist": args.artist or "",
        "album": args.album or default_stem,
        "track": args.track or "",
        "genre": args.genre or "",
        "comment": args.comment or f"Generated by tiny-voca (voice={args.voice}, speed={args.speed:g})",
    }

    show_progress = not args.no_progress
    bundle, stats = synthesize_parallel(
        text=text,
        weights_dir=weights_dir,
        voice=args.voice,
        speed=args.speed,
        pitch_shift=args.pitch_shift,
        pitch_range=args.pitch_range,
        workers=args.workers,
        sentence_pause=args.sentence_pause,
        paragraph_pause=args.paragraph_pause,
        show_progress=show_progress,
    )
    t_io0 = time.perf_counter()
    save_audio_streaming(
        output_path,
        bundle,
        subtype="FLOAT" if args.float_wav else "PCM_16",
        bitrate=args.bitrate,
        metadata=metadata,
        show_progress=show_progress,
    )
    t_io = time.perf_counter() - t_io0

    wall = stats["total_s"] + t_io
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(
        f"Saved {output_path} ({size_mb:.2f} MB, {stats['audio_s']:.2f}s audio, {stats['utterances']} utterances, "
        f"voice={args.voice}, speed={args.speed:g}, workers={stats['workers']}, peak={stats['peak']:.4f})\n"
        f"  prep+load : {stats['prep_s']*1000:6.1f} ms\n"
        f"  synthesis : {stats['synth_s']:6.3f} s  ({stats['audio_s']/stats['synth_s']:.1f}x real-time)\n"
        f"  encode+io : {(stats['post_s']+t_io):6.3f} s\n"
        f"  end-to-end: {wall:6.3f} s  ({stats['audio_s']/wall:.1f}x real-time)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
