"""Unit tests for HardwareDetector and model recommendations."""

import subprocess
import unittest
from unittest.mock import MagicMock, patch

from src.core.hardware import (
    GpuInfo,
    HardwareDetector,
    OllamaInfo,
    SystemMemoryInfo,
)


class TestHardwareDetector(unittest.TestCase):
    def setUp(self):
        self.detector = HardwareDetector(ollama_base_url="http://localhost:11434")

    def test_get_system_memory(self):
        mem = self.detector.get_system_memory()
        self.assertIsInstance(mem, SystemMemoryInfo)
        self.assertGreater(mem.total_ram_gb, 0.0)
        self.assertGreater(mem.available_ram_gb, 0.0)
        self.assertGreaterEqual(mem.used_ram_gb, 0.0)

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_gpu_info_nvidia_detected(self, mock_run, mock_which):
        mock_which.return_value = "C:\\Windows\\System32\\nvidia-smi.exe"
        mock_proc = MagicMock()
        mock_proc.stdout = "NVIDIA GeForce RTX 4080, 16384, 15000, 560.94\n"
        mock_run.return_value = mock_proc

        gpu = self.detector.get_gpu_info()
        self.assertTrue(gpu.has_nvidia)
        self.assertEqual(gpu.gpu_name, "NVIDIA GeForce RTX 4080")
        self.assertEqual(gpu.total_vram_mb, 16384)
        self.assertEqual(gpu.free_vram_mb, 15000)
        self.assertEqual(gpu.driver_version, "560.94")
        self.assertTrue(gpu.cuda_available)

    @patch("shutil.which")
    def test_gpu_info_no_nvidia_smi(self, mock_which):
        mock_which.return_value = None
        gpu = self.detector.get_gpu_info()
        self.assertFalse(gpu.has_nvidia)
        self.assertIsNone(gpu.gpu_name)
        self.assertFalse(gpu.cuda_available)

    @patch("httpx.Client")
    def test_ollama_info_running(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client

        v_resp = MagicMock(status_code=200)
        v_resp.json.return_value = {"version": "0.40.16"}
        tags_resp = MagicMock(status_code=200)
        tags_resp.json.return_value = {"models": [{"name": "llama3.2:3b"}, {"name": "qwen3:4b"}]}
        ps_resp = MagicMock(status_code=200)
        ps_resp.json.return_value = {"models": [{"name": "llama3.2:3b"}]}

        mock_client.get.side_effect = [v_resp, tags_resp, ps_resp]

        ollama = self.detector.get_ollama_info()
        self.assertTrue(ollama.is_running)
        self.assertEqual(ollama.version, "0.40.16")
        self.assertTrue(ollama.is_installed("llama3.2:3b"))
        self.assertTrue(ollama.is_installed("qwen3:4b"))
        self.assertFalse(ollama.is_installed("gemma3:4b"))

    @patch("httpx.Client")
    def test_ollama_info_unreachable(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_client.get.side_effect = Exception("Connection refused")

        ollama = self.detector.get_ollama_info()
        self.assertFalse(ollama.is_running)
        self.assertEqual(len(ollama.installed_models), 0)

    def test_recommendations_with_ample_vram(self):
        mem = SystemMemoryInfo(total_ram_gb=32.0, available_ram_gb=24.0, used_ram_gb=8.0)
        gpu = GpuInfo(has_nvidia=True, gpu_name="RTX 4096", total_vram_mb=16384, free_vram_mb=14000, cuda_available=True)
        ollama = OllamaInfo(is_running=True, installed_models=["llama3.2:3b"])

        recs = self.detector.generate_recommendations(mem, gpu, ollama)
        rec_map = {r.model_id: r for r in recs}

        self.assertIn("qwen3:4b", rec_map)
        self.assertTrue(rec_map["qwen3:4b"].fits_vram)
        self.assertEqual(rec_map["qwen3:4b"].recommendation_level, "Recommended")

        self.assertIn("qwen3:8b", rec_map)
        self.assertTrue(rec_map["qwen3:8b"].fits_vram)

        self.assertTrue(rec_map["llama3.2:3b"].is_installed)
        self.assertEqual(rec_map["llama3.2:3b"].recommendation_level, "Strongly Recommended")

    def test_recommendations_low_vram_or_cpu_only(self):
        mem = SystemMemoryInfo(total_ram_gb=16.0, available_ram_gb=8.0, used_ram_gb=8.0)
        gpu = GpuInfo(has_nvidia=False)
        ollama = OllamaInfo(is_running=True, installed_models=[])

        recs = self.detector.generate_recommendations(mem, gpu, ollama)
        rec_map = {r.model_id: r for r in recs}

        # On CPU only, large models get warning about running on CPU
        self.assertIn("qwen3:8b", rec_map)
        self.assertFalse(rec_map["qwen3:8b"].fits_vram)
        self.assertIn("CPU Only", rec_map["qwen3:8b"].recommendation_level)
        self.assertIsNotNone(rec_map["qwen3:8b"].warning)


if __name__ == "__main__":
    unittest.main()
