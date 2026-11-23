"""Ollama client with token, latency, and RAM instrumentation."""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

import ollama
import psutil

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    content: str
    prompt_tokens: int
    eval_tokens: int
    elapsed_s: float
    peak_ram_mb: float
    error: str = ""


def measure_ram_mb() -> float:
    """Measure combined RSS (MB) of Ollama server/runner processes + current Python process."""
    total_bytes = 0
    my_pid = os.getpid()
    for proc in psutil.process_iter(["pid", "name", "cmdline", "memory_info"]):
        try:
            info = proc.info
            pid = info["pid"]
            name = (info.get("name") or "").lower()
            cmd = " ".join(info.get("cmdline") or []).lower()
            if pid == my_pid or "ollama" in name or "ollama" in cmd:
                mem = info.get("memory_info")
                if mem is not None:
                    total_bytes += mem.rss
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return round(total_bytes / (1024 * 1024), 2)


class OllamaClient:
    def __init__(
        self,
        model: str = "qwen2.5:7b",
        temperature: float = 0.3,
        num_gpu: Optional[int] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.num_gpu = num_gpu

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        seed: Optional[int] = None,
        timeout: int = 120,
    ) -> LLMResponse:
        options: dict[str, Any] = {
            "temperature": self.temperature,
            "num_predict": max_tokens,
        }
        if seed is not None:
            options["seed"] = seed
        if self.num_gpu is not None:
            options["num_gpu"] = self.num_gpu

        ram_before = measure_ram_mb()
        t0 = time.perf_counter()
        try:
            resp = ollama.chat(
                model=self.model,
                messages=messages,
                options=options,
            )
            elapsed = time.perf_counter() - t0
            ram_after = measure_ram_mb()
            content = resp.get("message", {}).get("content", "") or ""
            prompt_tokens = int(resp.get("prompt_eval_count") or 0)
            eval_tokens = int(resp.get("eval_count") or 0)
            return LLMResponse(
                content=content,
                prompt_tokens=prompt_tokens,
                eval_tokens=eval_tokens,
                elapsed_s=round(elapsed, 4),
                peak_ram_mb=max(ram_before, ram_after),
            )
        except Exception as e:
            elapsed = time.perf_counter() - t0
            ram_after = measure_ram_mb()
            logger.error(f"Ollama query failed ({self.model}): {e}")
            return LLMResponse(
                content="",
                prompt_tokens=0,
                eval_tokens=0,
                elapsed_s=round(elapsed, 4),
                peak_ram_mb=max(ram_before, ram_after),
                error=str(e),
            )

    def query(
        self,
        prompt: str,
        max_tokens: int = 512,
        seed: Optional[int] = None,
        timeout: int = 120,
    ) -> LLMResponse:
        return self.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            seed=seed,
            timeout=timeout,
        )
