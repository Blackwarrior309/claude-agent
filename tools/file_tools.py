"""
File system and code execution tools.
"""
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

ALLOWED_BASE_DIRS = ["data", "logs"]


def _safe_path(file_path: str) -> Path:
    path = Path(file_path)
    if path.is_absolute():
        raise ValueError("Absolute paths not allowed")
    resolved = path.resolve()
    cwd = Path.cwd().resolve()
    if not str(resolved).startswith(str(cwd)):
        raise ValueError("Path traversal detected")
    return path


def read_file(file_path: str) -> str:
    path = _safe_path(file_path)
    return path.read_text(encoding="utf-8")


def write_file(file_path: str, content: str) -> str:
    path = _safe_path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"Written: {file_path} ({len(content)} chars)"


def list_files(directory: str = "data") -> list[str]:
    path = _safe_path(directory)
    if not path.exists():
        return []
    return [str(p) for p in path.rglob("*") if p.is_file()]


def run_python_snippet(code: str, timeout_s: int = 10) -> dict:
    """Execute a small Python snippet in a subprocess sandbox."""
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env={**os.environ, "PYTHONPATH": str(Path.cwd())},
        )
        return {
            "stdout": result.stdout[:2000],
            "stderr": result.stderr[:500],
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "Timeout", "returncode": -1}
    finally:
        os.unlink(tmp_path)


def get_tool_definitions() -> list[dict]:
    return [
        {
            "name": "read_file",
            "description": "Read content of a file in the data directory",
            "schema": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]},
            "fn": read_file,
        },
        {
            "name": "write_file",
            "description": "Write content to a file in the data directory",
            "schema": {
                "type": "object",
                "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["file_path", "content"],
            },
            "fn": write_file,
        },
        {
            "name": "list_files",
            "description": "List files in the data directory",
            "schema": {"type": "object", "properties": {"directory": {"type": "string", "default": "data"}}},
            "fn": list_files,
        },
        {
            "name": "run_python",
            "description": "Execute a Python code snippet (sandboxed)",
            "schema": {
                "type": "object",
                "properties": {"code": {"type": "string"}, "timeout_s": {"type": "integer", "default": 10}},
                "required": ["code"],
            },
            "fn": run_python_snippet,
        },
    ]
