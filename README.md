# Relay AI — Phase 7 (Multi-Model Collaboration & Comparison)

Native Windows desktop AI application built using Python, PySide6, and a universal AI provider architecture.

## Supported Providers
- **Ollama (Local)**: High-performance local inference via `llama3.2:3b`. Zero cloud dependency.
- **Google Gemini (Cloud)**: Flagship Google Developer API integration using the official `google-genai` SDK (`gemini-3.5-flash`, `gemini-3.8-flash`, etc.).
- **Groq (Cloud)**: Ultra-low latency LPU cloud inference using the official `groq` SDK (`openai/gpt-oss-120b`, `qwen/qwen3.8-27b`, etc.).
- **OpenAI (Cloud)**: Industry standard reasoning and chat models via the official `openai` SDK (`gpt-4o-mini`, `gpt-4o`, `gpt-4.1-mini`, `gpt-4.1`, `o3`).

## Key Features
- **Side-by-Side Model Comparison (2 to 4 Models)**: Compare responses from different models (across local and cloud providers) concurrently on the same prompt with shared conversation history.
- **Responsive Layout Adaptation**: Dynamically switches between 2-column, 3-column, and 2×2 grid layouts depending on model count.
- **Independent Streaming & Error Isolation**: Each model card streams in its own thread with dedicated status badges (`Connecting…`, `Streaming…`, `Completed`, `Error`, `Canceled`). A failure or rate-limit in one provider never disrupts or stops others.
- **Safe Cooperative Cancellation**: Instant cancellation via "■ Stop All" that cooperatively aborts all running workers without destroying threads.
- **Live Performance & Cost Metrics**: Real-time display of response time (seconds), token counts, token generation speeds (tokens/sec), and estimated API costs (USD) with local inference marked as `$0.00 (Local)`.
- **Comparative Summary Banner**: Concise performance synthesis showing fastest model, speed, and token comparison across all models.
- **Selective Context Commitment**: Only the user's selected winning response ("✓ Commit to Chat") is appended to the main conversation history, preventing context bloat.
- **Seamless Single / Compare Mode Toggle**: Easily alternate between standard 1-on-1 chat and side-by-side comparison.
- **Secure Windows Credential Manager (`CredentialStore`)**: Gemini, Groq, and OpenAI API keys are persisted strictly inside the OS Keyring (`WinVaultKeyring`). No plaintext storage in `.env`, config files, or logs.
- **Settings Screen API Key Manager**: Directly manage provider credentials using password-masked fields.

## Setup & Running the Application
```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

## Running the Complete Test Suite
```powershell
.\.venv\Scripts\python.exe -m unittest discover tests
```
