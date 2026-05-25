"""
Ollama local model client optimized for 4 GB VRAM GPUs (z.B. RTX 3060 4 GB).

Handles: health checks, model availability, GPU detection, request routing.
Falls back gracefully when Ollama is not running.

Modelle für 4 GB VRAM (nur Q4-quantisierte Modelle ≤ 3B Parameter passen sicher):
  phi3.5:mini      ~2.2 GB  — Klassifikation, Routing, Parsing (schnellstes Modell)
  llama3.2:3b      ~2.0 GB  — Zusammenfassungen, einfache Antworten
  gemma2:2b        ~1.5 GB  — Extraktion, einfache Aufgaben (Alternative zu llama3.2)
  nomic-embed-text ~0.3 GB  — Embeddings (ersetzt ChromaDB built-in)

NICHT geeignet für 4 GB VRAM:
  llama3.1:8b   ~4.7 GB  → zu groß
  mistral:7b    ~4.1 GB  → zu groß
  → Diese Tasks fallen auf Claude Haiku zurück (günstig aber nicht kostenlos)
"""
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

logger = logging.getLogger(__name__)

OLLAMA_BASE = "http://localhost:11434"
_AVAILABILITY_CACHE: dict[str, float] = {}  # model_name → last_check_ts
_CACHE_TTL = 60.0  # re-check every 60s


@dataclass
class LocalModelConfig:
    name: str            # e.g. "phi3.5:mini"
    vram_gb: float
    context_tokens: int
    use_for: list[str]
    temperature: float = 0.2


# Nur Modelle die sicher in 4 GB VRAM passen (mit etwas Puffer für System/OS)
LOCAL_MODELS = {
    "nano": LocalModelConfig(
        name="phi3.5:mini",
        vram_gb=2.2,
        context_tokens=4096,
        # Bestes Modell für strukturierte Outputs und Klassifikation auf kleiner GPU
        use_for=["classification", "filtering", "routing", "parsing", "simple_qa"],
    ),
    "small": LocalModelConfig(
        name="llama3.2:3b",
        vram_gb=2.0,
        context_tokens=8192,
        use_for=["summarization", "simple_answers", "extraction"],
    ),
    # 7B/8B Modelle (>4 GB) werden NICHT eingebunden — fallen auf Haiku zurück
    "embed": LocalModelConfig(
        name="nomic-embed-text",
        vram_gb=0.3,
        context_tokens=8192,
        use_for=["embeddings"],
    ),
}

# Tasks die lokal laufen (kostenlos) vs. die auf Haiku zurückfallen
TASK_TO_LOCAL_MODEL = {
    "classification": "nano",
    "filtering":      "nano",
    "routing":        "nano",
    "parsing":        "nano",
    "simple_qa":      "nano",
    "summarization":  "small",
    "extraction":     "small",
    "embeddings":     "embed",
    # analysis / writing / coding → kein lokales Modell für 4 GB → Haiku Fallback
}


def _http_post(url: str, body: dict, timeout: int = 60) -> Optional[dict]:
    data = json.dumps(body).encode()
    req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except (URLError, Exception):
        return None


def _http_get(url: str, timeout: int = 5) -> Optional[dict]:
    req = Request(url)
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except (URLError, Exception):
        return None


class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_BASE):
        self.base_url = base_url
        self._available: Optional[bool] = None
        self._pulled_models: set[str] = set()
        self._last_check = 0.0

    # ── Availability ──────────────────────────────────────────────────────────

    def is_running(self) -> bool:
        now = time.time()
        if now - self._last_check < 10:
            return self._available is True
        self._last_check = now
        resp = _http_get(f"{self.base_url}/api/tags", timeout=3)
        self._available = resp is not None
        if self._available:
            self._pulled_models = {m["name"].split(":")[0] for m in resp.get("models", [])}
        return self._available

    def list_models(self) -> list[str]:
        resp = _http_get(f"{self.base_url}/api/tags")
        if not resp:
            return []
        return [m["name"] for m in resp.get("models", [])]

    def model_is_pulled(self, model_name: str) -> bool:
        pulled = self.list_models()
        return any(model_name in m for m in pulled)

    def pull_model(self, model_name: str) -> bool:
        """Pull a model if not already present. Blocks until done."""
        if self.model_is_pulled(model_name):
            logger.info(f"Ollama: '{model_name}' already present")
            return True
        logger.info(f"Ollama: pulling '{model_name}' (this may take a while)…")
        resp = _http_post(f"{self.base_url}/api/pull", {"name": model_name, "stream": False}, timeout=600)
        success = resp is not None and resp.get("status") == "success"
        if success:
            logger.info(f"Ollama: pulled '{model_name}' successfully")
        else:
            logger.warning(f"Ollama: pull failed for '{model_name}': {resp}")
        return success

    # ── Inference ─────────────────────────────────────────────────────────────

    def generate(
        self,
        model: str,
        prompt: str,
        system: str = "",
        temperature: float = 0.2,
        max_tokens: int = 1000,
    ) -> Optional[str]:
        """Single-turn generation via /api/generate."""
        if not self.is_running():
            return None

        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": 4096,
            },
        }
        if system:
            body["system"] = system

        resp = _http_post(f"{self.base_url}/api/generate", body, timeout=120)
        if resp is None:
            logger.warning(f"Ollama: generate returned None for model '{model}'")
            return None
        return resp.get("response", "")

    def chat(
        self,
        model: str,
        messages: list[dict],
        system: str = "",
        temperature: float = 0.2,
        max_tokens: int = 1000,
    ) -> Optional[str]:
        """Multi-turn chat via /api/chat."""
        if not self.is_running():
            return None

        chat_messages = []
        if system:
            chat_messages.append({"role": "system", "content": system})
        chat_messages.extend(messages)

        body: dict[str, Any] = {
            "model": model,
            "messages": chat_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": 4096,
            },
        }
        resp = _http_post(f"{self.base_url}/api/chat", body, timeout=120)
        if resp is None:
            return None
        return resp.get("message", {}).get("content", "")

    def embed(self, model: str, text: str) -> Optional[list[float]]:
        """Generate embeddings via /api/embeddings."""
        if not self.is_running():
            return None
        resp = _http_post(f"{self.base_url}/api/embeddings", {"model": model, "prompt": text}, timeout=30)
        if resp is None:
            return None
        return resp.get("embedding")

    # ── GPU info ─────────────────────────────────────────────────────────────

    def gpu_info(self) -> dict:
        """Try to get VRAM info via nvidia-smi."""
        import subprocess
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
                text=True, timeout=5,
            )
            name, total, free = [x.strip() for x in out.strip().split(",")]
            return {"gpu": name, "vram_total_mb": int(total), "vram_free_mb": int(free)}
        except Exception:
            return {"gpu": "unknown"}


class LocalModelRouter:
    """Selects the best local model for a task and calls it via Ollama."""

    def __init__(self, client: OllamaClient):
        self.client = client

    def available(self) -> bool:
        return self.client.is_running()

    def best_model_for(self, task_type: str) -> Optional[LocalModelConfig]:
        tier = TASK_TO_LOCAL_MODEL.get(task_type)
        if not tier:
            return None
        return LOCAL_MODELS.get(tier)

    def call(
        self,
        task_type: str,
        messages: list[dict],
        system: str = "",
        max_tokens: int = 1000,
        temperature: float | None = None,
    ) -> Optional[str]:
        """Call the best local model for the task. Returns None on failure."""
        if not self.available():
            return None

        cfg = self.best_model_for(task_type)
        if cfg is None:
            return None

        temp = temperature if temperature is not None else cfg.temperature

        if not self.client.model_is_pulled(cfg.name):
            logger.info(f"LocalRouter: model '{cfg.name}' not pulled, skipping local")
            return None

        result = self.client.chat(cfg.name, messages, system=system, temperature=temp, max_tokens=max_tokens)
        if result:
            logger.debug(f"LocalRouter: '{cfg.name}' handled task '{task_type}' locally (0 cost)")
        return result

    def embed(self, text: str) -> Optional[list[float]]:
        cfg = LOCAL_MODELS["embed"]
        if not self.client.model_is_pulled(cfg.name):
            return None
        return self.client.embed(cfg.name, text)

    def status(self) -> dict:
        running = self.client.is_running()
        pulled = self.client.list_models() if running else []
        return {
            "ollama_running": running,
            "pulled_models": pulled,
            "gpu": self.client.gpu_info() if running else {},
            "recommended_pulls": [cfg.name for cfg in LOCAL_MODELS.values()],
        }
