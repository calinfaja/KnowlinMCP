"""Cross-platform utilities for KnowlinMCP.

Provides cross-platform path handling and process management
using platformdirs and psutil.
"""

from __future__ import annotations

import getpass
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import platformdirs
import psutil

# Default port for knowledge server
DEFAULT_KB_PORT = 14000

# Knowledge DB directory name
KB_DIR_NAME = ".knowledge-db"

# Project markers in priority order
PROJECT_MARKERS = [".knowledge-db", ".serena", ".claude", ".git"]

HOST = "127.0.0.1"


# =============================================================================
# Path Utilities
# =============================================================================


def get_config_dir() -> Path:
    """Get the KnowlinMCP configuration directory (platform-specific)."""
    return Path(platformdirs.user_config_dir("knowlin-mcp", ensure_exists=True))


def get_cache_dir() -> Path:
    """Get the KnowlinMCP cache directory (platform-specific)."""
    return Path(platformdirs.user_cache_dir("knowlin-mcp", ensure_exists=True))


def get_runtime_dir() -> Path:
    """Get the KnowlinMCP runtime directory for PIDs and port files.

    Returns /tmp/knowlin-{username} (Linux/macOS) or %TEMP%/knowlin-{username} (Windows).
    """
    username = getpass.getuser()
    runtime_dir = Path(tempfile.gettempdir()) / f"knowlin-{username}"
    if runtime_dir.exists() and runtime_dir.is_symlink():
        raise RuntimeError(f"Runtime dir cannot be a symlink: {runtime_dir}")

    runtime_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    if runtime_dir.is_symlink():
        raise RuntimeError(f"Runtime dir cannot be a symlink: {runtime_dir}")

    if hasattr(os, "getuid"):
        current_uid = os.getuid()
        if runtime_dir.stat().st_uid != current_uid:
            raise PermissionError(f"Runtime dir is not owned by the current user: {runtime_dir}")

    if os.name != "nt":
        runtime_dir.chmod(0o700)

    return runtime_dir


def get_kb_port() -> int:
    """Get the knowledge server port from environment or default (14000)."""
    return int(os.environ.get("KNOWLIN_KB_PORT", str(DEFAULT_KB_PORT)))


def get_project_hash(project_path: Path) -> str:
    """Generate 8-char hex hash for a project path (unique port/file names)."""
    path_str = str(project_path.resolve())
    return hashlib.md5(path_str.encode()).hexdigest()[:8]


def get_kb_port_file(project_path: Path) -> Path:
    """Get path to the knowledge server port file for a project."""
    project_hash = get_project_hash(project_path)
    return get_runtime_dir() / f"kb-{project_hash}.port"


def get_kb_pid_file(project_path: Path) -> Path:
    """Get path to the knowledge server PID file for a project."""
    project_hash = get_project_hash(project_path)
    return get_runtime_dir() / f"kb-{project_hash}.pid"


def get_kb_token_file(project_path: Path) -> Path:
    """Get path to the knowledge server auth token file for a project."""
    project_hash = get_project_hash(project_path)
    return get_runtime_dir() / f"kb-{project_hash}.token"


def find_project_root(start_path: Path | None = None) -> Path | None:
    """Find the project root by looking for markers.

    Checks CLAUDE_PROJECT_DIR env var first, then walks up from start_path
    looking for .knowledge-db, .serena, .claude, or .git directories.
    """
    if env_dir := os.environ.get("CLAUDE_PROJECT_DIR"):
        p = Path(env_dir)
        if p.is_dir():
            return p

    current = Path(start_path or Path.cwd()).resolve()

    while current != current.parent:
        for marker in PROJECT_MARKERS:
            if (current / marker).exists():
                return current
        current = current.parent

    return None


# =============================================================================
# Process Utilities
# =============================================================================


def find_process(pattern: str) -> psutil.Process | None:
    """Find a process by command line pattern (cross-platform pgrep)."""
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = proc.info.get("cmdline")
            if cmdline and pattern in " ".join(cmdline):
                return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


def is_process_running(pid: int) -> bool:
    """Check if a process with the given PID is running."""
    try:
        proc = psutil.Process(pid)
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def spawn_background(
    cmd: list[str],
    cwd: Path | None = None,
    env: dict | None = None,
    log_file: Path | None = None,
) -> int:
    """Spawn a background process that survives parent exit. Returns PID."""
    log_handle = open(log_file, "a") if log_file else None  # noqa: SIM115

    kwargs: dict = {
        "stdout": log_handle if log_handle else subprocess.DEVNULL,
        "stderr": log_handle if log_handle else subprocess.DEVNULL,
        "stdin": subprocess.DEVNULL,
    }

    if cwd:
        kwargs["cwd"] = str(cwd)
    if env:
        kwargs["env"] = env

    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    else:
        kwargs["start_new_session"] = True

    try:
        proc = subprocess.Popen(cmd, **kwargs)
    except Exception:
        if log_handle:
            log_handle.close()
        raise
    return proc.pid


def kill_process_tree(pid: int, timeout: float = 5.0) -> bool:
    """Kill a process and all its children. Returns True if killed."""
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return False

    children = parent.children(recursive=True)

    try:
        parent.terminate()
    except psutil.NoSuchProcess:
        pass

    for child in children:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass

    gone, alive = psutil.wait_procs([parent] + children, timeout=timeout)

    for proc in alive:
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass

    return True


def read_pid_file(pid_file: Path) -> int | None:
    """Read a PID from a file. Returns None if missing or invalid."""
    try:
        if pid_file.exists():
            content = pid_file.read_text().strip()
            return int(content)
    except (ValueError, OSError):
        pass
    return None


def write_runtime_file(path: Path, content: str) -> None:
    """Write a runtime file without following symlinks."""
    runtime_dir = get_runtime_dir()
    if path.parent != runtime_dir:
        raise ValueError(f"Runtime file must live in {runtime_dir}: {path}")
    if path.exists() and path.is_symlink():
        raise RuntimeError(f"Refusing to write through symlink: {path}")

    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW

    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def write_pid_file(pid_file: Path, pid: int) -> None:
    """Write a PID to a file."""
    write_runtime_file(pid_file, str(pid))


def cleanup_stale_files(project_path: Path) -> None:
    """Remove stale PID and port files if process is dead."""
    pid_file = get_kb_pid_file(project_path)
    port_file = get_kb_port_file(project_path)
    token_file = get_kb_token_file(project_path)

    pid = read_pid_file(pid_file)
    if pid and not is_process_running(pid):
        try:
            pid_file.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            port_file.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            token_file.unlink(missing_ok=True)
        except OSError:
            pass
