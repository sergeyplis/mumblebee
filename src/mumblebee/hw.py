"""Hardware detection and XDG cache helpers (zero machine-specific hardcoded paths)."""

import os
from pathlib import Path
import subprocess


def physical_cpu_count() -> int:
    """Return physical CPU core count (best default for 1-thread-per-worker ONNX AVX-512/SIMD pools)."""
    try:
        # Linux: count unique (physical_package_id, core_id) pairs
        topo_root = Path("/sys/devices/system/cpu")
        cores: set[tuple[str, str]] = set()
        for cpu_dir in topo_root.glob("cpu[0-9]*"):
            pkg_f = cpu_dir / "topology" / "physical_package_id"
            core_f = cpu_dir / "topology" / "core_id"
            if pkg_f.is_file() and core_f.is_file():
                cores.add((pkg_f.read_text().strip(), core_f.read_text().strip()))
        if cores:
            return len(cores)
    except Exception:
        pass
    logical = os.cpu_count() or 4
    return max(1, logical // 2) if logical >= 4 else logical


def cache_dir() -> Path:
    """Return writable cache directory for pre-optimized ONNX graphs and downloaded weights."""
    override = os.environ.get("MUMBLEBEE_CACHE_DIR")
    if override:
        path = Path(override).expanduser()
    else:
        xdg = os.environ.get("XDG_CACHE_HOME")
        base = Path(xdg).expanduser() if xdg else (Path.home() / ".cache")
        path = base / "mumblebee"
    path.mkdir(parents=True, exist_ok=True)
    return path


def process_running(name: str) -> bool:
    """Best-effort check whether a process with exact binary name is running (Linux/macOS)."""
    try:
        res = subprocess.run(
            ["pgrep", "-x", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return res.returncode == 0
    except Exception:
        return False
