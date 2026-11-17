"""Hardware detection and Ollama model recommendation engine for Relay AI."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
import json
import logging
import platform
import shutil
import subprocess
from typing import Any, Callable, Dict, Iterator, List, Optional

import httpx

logger = logging.getLogger("RelayAI.Hardware")

DEFAULT_OLLAMA_URL = "http://localhost:11434"

# Standard known model catalog with resource profiles
MODEL_CATALOG: Dict[str, Dict[str, Any]] = {
    "qwen3:4b": {
        "role": "Primary Chat",
        "size_gb": 2.6,
        "min_vram_gb": 3.5,
        "min_ram_gb": 6.0,
        "description": "Balanced high-performance local chat model with fast inference.",
    },
    "qwen3:8b": {
        "role": "Higher-Quality Candidate",
        "size_gb": 5.2,
        "min_vram_gb": 6.5,
        "min_ram_gb": 10.0,
        "description": "High reasoning quality for complex instruction-following.",
    },
    "gemma3:4b": {
        "role": "Alternative Chat",
        "size_gb": 2.8,
        "min_vram_gb": 3.8,
        "min_ram_gb": 6.0,
        "description": "Google Gemma 3 architecture optimized for concise assistance.",
    },
    "qwen2.5-coder:3b": {
        "role": "Coding Specialist",
        "size_gb": 2.0,
        "min_vram_gb": 3.0,
        "min_ram_gb": 5.0,
        "description": "Specialized code generation, debugging, and syntax reasoning.",
    },
    "llama3.2:3b": {
        "role": "Existing Fallback",
        "size_gb": 2.0,
        "min_vram_gb": 3.0,
        "min_ram_gb": 5.0,
        "description": "Lightweight, reliable local fallback model.",
    },
    "nomic-embed-text": {
        "role": "Local Embedding",
        "size_gb": 0.28,
        "min_vram_gb": 0.8,
        "min_ram_gb": 2.0,
        "description": "Local high-performance semantic text embedding model (768 dimensions).",
    },
}


@dataclass
class SystemMemoryInfo:
    total_ram_gb: float
    available_ram_gb: float
    used_ram_gb: float

    def __str__(self) -> str:
        return f"RAM: {self.available_ram_gb:.1f} GB available / {self.total_ram_gb:.1f} GB total"


@dataclass
class GpuInfo:
    has_nvidia: bool
    gpu_name: Optional[str] = None
    total_vram_mb: Optional[int] = None
    free_vram_mb: Optional[int] = None
    driver_version: Optional[str] = None
    cuda_available: bool = False

    @property
    def total_vram_gb(self) -> float:
        return (self.total_vram_mb or 0) / 1024.0

    @property
    def free_vram_gb(self) -> float:
        return (self.free_vram_mb or 0) / 1024.0

    def __str__(self) -> str:
        if not self.has_nvidia or not self.gpu_name:
            return "GPU: None / Integrated graphics"
        return f"GPU: {self.gpu_name} ({self.free_vram_mb} MB free / {self.total_vram_mb} MB total VRAM, Driver {self.driver_version})"


@dataclass
class OllamaInfo:
    is_running: bool
    version: Optional[str] = None
    installed_models: List[str] = field(default_factory=list)
    running_models: List[str] = field(default_factory=list)

    def is_installed(self, model_name: str) -> bool:
        clean = model_name.strip().lower()
        clean_base = clean.split(":")[0] if ":" in clean else clean
        for m in self.installed_models:
            m_lower = m.lower()
            if m_lower == clean or m_lower.startswith(f"{clean}:") or m_lower == f"{clean}:latest":
                return True
            if clean.endswith(":latest") and m_lower == clean_base:
                return True
        return False


@dataclass
class ModelRecommendation:
    role: str
    model_id: str
    size_gb: float
    min_vram_gb: float
    min_ram_gb: float
    fits_vram: bool
    is_installed: bool
    is_running: bool
    recommendation_level: str
    warning: Optional[str] = None
    description: str = ""


@dataclass
class HardwareReport:
    memory: SystemMemoryInfo
    gpu: GpuInfo
    ollama: OllamaInfo
    recommendations: List[ModelRecommendation] = field(default_factory=list)


class HardwareDetector:
    """Detects system RAM, GPU/VRAM, and Ollama status on Windows, computing model recommendations."""

    def __init__(self, ollama_base_url: str = DEFAULT_OLLAMA_URL):
        self.ollama_base_url = ollama_base_url.rstrip("/")

    def get_system_memory(self) -> SystemMemoryInfo:
        """Query system RAM using Windows Win32 API GlobalMemoryStatusEx with fallbacks."""
        if platform.system() == "Windows":
            try:
                class _MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                stat = _MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                    total_gb = stat.ullTotalPhys / (1024**3)
                    avail_gb = stat.ullAvailPhys / (1024**3)
                    return SystemMemoryInfo(
                        total_ram_gb=round(total_gb, 2),
                        available_ram_gb=round(avail_gb, 2),
                        used_ram_gb=round(total_gb - avail_gb, 2),
                    )
            except Exception as exc:
                logger.debug("Win32 memory query error: %s", exc)

        # Fallback query via psutil or default
        try:
            import psutil
            vm = psutil.virtual_memory()
            total_gb = vm.total / (1024**3)
            avail_gb = vm.available / (1024**3)
            return SystemMemoryInfo(
                total_ram_gb=round(total_gb, 2),
                available_ram_gb=round(avail_gb, 2),
                used_ram_gb=round(total_gb - avail_gb, 2),
            )
        except Exception:
            return SystemMemoryInfo(total_ram_gb=8.0, available_ram_gb=4.0, used_ram_gb=4.0)

    def get_gpu_info(self) -> GpuInfo:
        """Query NVIDIA GPU via nvidia-smi with graceful failure handling."""
        nvidia_smi_path = shutil.which("nvidia-smi")
        if not nvidia_smi_path:
            return GpuInfo(has_nvidia=False)

        try:
            cmd = [
                nvidia_smi_path,
                "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3.0, check=True)
            output = proc.stdout.strip()
            if not output:
                return GpuInfo(has_nvidia=False)

            # Parse primary GPU line: e.g. "NVIDIA GeForce RTX 2050, 4096, 3957, 581.86"
            lines = output.splitlines()
            parts = [p.strip() for p in lines[0].split(",")]
            if len(parts) >= 4:
                gpu_name = parts[0]
                total_vram = int(parts[1])
                free_vram = int(parts[2])
                driver_version = parts[3]
                return GpuInfo(
                    has_nvidia=True,
                    gpu_name=gpu_name,
                    total_vram_mb=total_vram,
                    free_vram_mb=free_vram,
                    driver_version=driver_version,
                    cuda_available=True,
                )
        except Exception as exc:
            logger.debug("nvidia-smi query execution failed: %s", exc)

        return GpuInfo(has_nvidia=False)

    def get_ollama_info(self, timeout: float = 2.0) -> OllamaInfo:
        """Query Ollama server for version, installed models, and running processes."""
        try:
            with httpx.Client(timeout=timeout) as client:
                # 1. Version check
                v_resp = client.get(f"{self.ollama_base_url}/api/version")
                if v_resp.status_code != 200:
                    return OllamaInfo(is_running=False)
                version = v_resp.json().get("version")

                # 2. Installed models check
                tags_resp = client.get(f"{self.ollama_base_url}/api/tags")
                installed_models: List[str] = []
                if tags_resp.status_code == 200:
                    for m in tags_resp.json().get("models", []):
                        name = m.get("name")
                        if name:
                            installed_models.append(name)

                # 3. Running models check
                running_models: List[str] = []
                try:
                    ps_resp = client.get(f"{self.ollama_base_url}/api/ps")
                    if ps_resp.status_code == 200:
                        for m in ps_resp.json().get("models", []):
                            name = m.get("name")
                            if name:
                                running_models.append(name)
                except Exception:
                    pass

                return OllamaInfo(
                    is_running=True,
                    version=version,
                    installed_models=installed_models,
                    running_models=running_models,
                )
        except Exception:
            return OllamaInfo(is_running=False)

    def generate_recommendations(
        self,
        memory: SystemMemoryInfo,
        gpu: GpuInfo,
        ollama: OllamaInfo,
    ) -> List[ModelRecommendation]:
        """Compute hardware-aware model recommendations without making assumptions."""
        recs: List[ModelRecommendation] = []
        has_vram = gpu.has_nvidia and gpu.total_vram_mb is not None
        vram_gb = gpu.total_vram_gb if has_vram else 0.0
        avail_ram_gb = memory.available_ram_gb
        total_ram_gb = memory.total_ram_gb

        for model_id, spec in MODEL_CATALOG.items():
            min_vram = spec["min_vram_gb"]
            min_ram = spec["min_ram_gb"]
            fits_vram = has_vram and (vram_gb >= min_vram)
            is_installed = ollama.is_installed(model_id)
            is_running = any(model_id in r for r in ollama.running_models)

            # Evaluation logic
            warning = None
            if fits_vram:
                rec_level = "Strongly Recommended" if is_installed else "Recommended"
            elif has_vram and (vram_gb + avail_ram_gb >= min_ram):
                rec_level = "Compatible (CPU/GPU Shared)"
                warning = (
                    f"Model size ({spec['size_gb']} GB) exceeds dedicated VRAM ({vram_gb:.1f} GB). "
                    "Inference will partially offload to system RAM."
                )
            elif total_ram_gb >= min_ram:
                rec_level = "Compatible (CPU Only)"
                warning = "Will run on system RAM (CPU). Inference may be slower."
            else:
                rec_level = "Requires More Memory"
                warning = f"Requires at least {min_ram} GB memory. System has {avail_ram_gb:.1f} GB available."

            recs.append(
                ModelRecommendation(
                    role=spec["role"],
                    model_id=model_id,
                    size_gb=spec["size_gb"],
                    min_vram_gb=min_vram,
                    min_ram_gb=min_ram,
                    fits_vram=fits_vram,
                    is_installed=is_installed,
                    is_running=is_running,
                    recommendation_level=rec_level,
                    warning=warning,
                    description=spec["description"],
                )
            )

        return recs

    def inspect_system(self) -> HardwareReport:
        """Run full hardware inspection and compute recommendations."""
        memory = self.get_system_memory()
        gpu = self.get_gpu_info()
        ollama = self.get_ollama_info()
        recommendations = self.generate_recommendations(memory, gpu, ollama)
        return HardwareReport(
            memory=memory,
            gpu=gpu,
            ollama=ollama,
            recommendations=recommendations,
        )

    def pull_model(
        self,
        model_name: str,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        timeout: float = 600.0,
    ) -> bool:
        """Stream pull a model from Ollama with progress reporting. Requires explicit user approval."""
        endpoint = f"{self.ollama_base_url}/api/pull"
        try:
            with httpx.Client(timeout=timeout) as client:
                with client.stream("POST", endpoint, json={"name": model_name}) as response:
                    if response.status_code != 200:
                        logger.error("Ollama pull failed with status %d", response.status_code)
                        return False
                    for line in response.iter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            if progress_callback:
                                progress_callback(data)
                            if data.get("status") == "success":
                                return True
                        except Exception:
                            continue
            return True
        except Exception as exc:
            logger.error("Error pulling model %s: %s", model_name, exc)
            return False
