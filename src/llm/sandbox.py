"""Isolated subprocess sandbox for executing LLM-generated Pandas code."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SandboxResult:
    exec_status: str  # "ok", "timeout", "syntax_error", "runtime_error", "empty_output", "no_code"
    stdout: str
    stderr: str
    code_extracted: str
    elapsed_s: float


def extract_python_code(text: str) -> str:
    """Extract Python code from markdown fences or raw response."""
    if not text or not text.strip():
        return ""
    # Prefer ```python ... ``` blocks
    py_blocks = re.findall(r"```(?:python|py)\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if py_blocks:
        return py_blocks[-1].strip()
    # Fallback to generic ``` ... ``` blocks
    any_blocks = re.findall(r"```\s*\n?(.*?)```", text, re.DOTALL)
    if any_blocks:
        return any_blocks[-1].strip()
    # Fallback: if response looks like raw Python code (references df or print)
    stripped = text.strip()
    if any(tok in stripped for tok in ("df[", "df.", "print(", "import pandas")):
        return stripped
    return ""


_SANDBOX_RUNNER = r'''
import builtins
import io
import json
import resource
import sys
import traceback
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

payload = json.loads(sys.stdin.read())
csv_path = payload["csv_path"]
code = payload["code"]

# Load DataFrame BEFORE locking down file I/O and imports
df = pd.read_csv(csv_path)

# Block file writes at OS level (0 bytes max file creation)
try:
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
except Exception:
    pass

# Block network, subprocess, and filesystem mutation imports
BLOCKED_PREFIXES = (
    "socket", "urllib", "http", "requests", "ftplib", "smtplib", "ssl",
    "subprocess", "shutil", "ctypes", "multiprocessing", "webbrowser",
    "aiohttp", "httpx", "paramiko", "telnetlib", "xmlrpc",
)

class _ImportBlocker:
    def find_spec(self, fullname, path, target=None):
        root = fullname.split(".")[0]
        if root in BLOCKED_PREFIXES:
            raise ImportError(f"Module '{fullname}' is blocked in sandbox")
        return None

for mod_name in list(sys.modules.keys()):
    if mod_name.split(".")[0] in BLOCKED_PREFIXES:
        sys.modules[mod_name] = None

sys.meta_path.insert(0, _ImportBlocker())

def _blocked_open(*args, **kwargs):
    raise PermissionError("File I/O is disabled in sandbox; use the pre-loaded DataFrame 'df'.")

builtins.open = _blocked_open

namespace = {"pd": pd, "np": np, "df": df, "__builtins__": builtins}
buf = io.StringIO()
status = "ok"
err_msg = ""

try:
    compiled = compile(code, "<llm_code>", "exec")
except SyntaxError:
    status = "syntax_error"
    err_msg = traceback.format_exc()
else:
    try:
        with redirect_stdout(buf):
            exec(compiled, namespace, namespace)
    except Exception:
        status = "runtime_error"
        err_msg = traceback.format_exc()

out = buf.getvalue().strip()
if status == "ok" and not out:
    # Check if user assigned a 'result' or 'answer' variable without printing
    for var in ("answer", "result", "res", "ans"):
        if var in namespace and namespace[var] is not None:
            out = str(namespace[var]).strip()
            break
    if not out:
        status = "empty_output"
        err_msg = "Code executed without error but produced no stdout output. Use print(...) to output the final answer."

sys.stdout.write(json.dumps({"status": status, "stdout": out, "stderr": err_msg}))
'''


def execute_pandas_code(
    code: str,
    csv_path: str | Path,
    timeout: int = 5,
) -> SandboxResult:
    """Execute Python/Pandas code in an isolated subprocess with timeout and no network/writes."""
    if not code or not code.strip():
        return SandboxResult(
            exec_status="no_code",
            stdout="",
            stderr="No Python code found in response.",
            code_extracted="",
            elapsed_s=0.0,
        )

    payload = json.dumps({"csv_path": str(Path(csv_path).resolve()), "code": code})
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _SANDBOX_RUNNER],
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={"PATH": "", "HOME": "/tmp", "PYTHONNOUSERSITE": "1"},
        )
        elapsed = round(time.perf_counter() - t0, 4)
        if proc.returncode != 0:
            return SandboxResult(
                exec_status="runtime_error",
                stdout="",
                stderr=(proc.stderr or proc.stdout or f"Process exited with code {proc.returncode}")[-1500:],
                code_extracted=code,
                elapsed_s=elapsed,
            )
        data = json.loads(proc.stdout)
        return SandboxResult(
            exec_status=data["status"],
            stdout=data["stdout"],
            stderr=data["stderr"][-1500:],
            code_extracted=code,
            elapsed_s=elapsed,
        )
    except subprocess.TimeoutExpired:
        elapsed = round(time.perf_counter() - t0, 4)
        return SandboxResult(
            exec_status="timeout",
            stdout="",
            stderr=f"Execution timed out after {timeout}s.",
            code_extracted=code,
            elapsed_s=elapsed,
        )
    except Exception as e:
        elapsed = round(time.perf_counter() - t0, 4)
        return SandboxResult(
            exec_status="runtime_error",
            stdout="",
            stderr=str(e)[-1500:],
            code_extracted=code,
            elapsed_s=elapsed,
        )
