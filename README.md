# Relay AI — Universal Context Engine & Multi-Agent Desktop Application

Native Windows desktop AI application built using Python 3.11+, PySide6, and a universal AI provider architecture.

## Supported Providers
- **Ollama (Local)**: High-performance local inference via `llama3.2:3b`. Zero cloud dependency.
- **Google Gemini (Cloud)**: Flagship Google Developer API integration using the official `google-genai` SDK (`gemini-3.5-flash`, `gemini-3.8-flash`, etc.).
- **Groq (Cloud)**: Ultra-low latency LPU cloud inference using the official `groq` SDK (`openai/gpt-oss-120b`, `qwen/qwen3.8-27b`, etc.).
- **OpenAI (Cloud)**: Industry standard reasoning and chat models via the official `openai` SDK (`gpt-4o-mini`, `gpt-4o`, `gpt-4.1-mini`, `gpt-4.1`, `o3`).

## Architecture & Capabilities

### Phase 7: Multi-Model Collaboration & Comparison
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

### Phase 8: Smart Routing & Model Selection
- Intent classification, query complexity evaluation, and automated cost-latency-accuracy trade-off optimization.
- Dynamic fallback routing between local and cloud providers based on network connectivity and credential availability.

### Phase 9: AI Teams & Multi-Agent Collaboration
- **Specialized Roles**: Planner, Researcher, Creator, Architect, Critic, Finalizer, and Custom roles.
- **Workflow Engine & Directed Execution**: Dependency validation, selective context transfer, step retries, skip-step policies, and fallback agent execution.
- **Iterative Critic Loops**: Automated evaluation with bounded revision loops and downstream state invalidation.
- **Teams UI**: Real-time agent step cards, revision selectors, timeline cards, and markdown deliverable viewer.

### Phase 10: Universal Context Engine (UCE) & Cloud Sync
- **Milestone 10.1: Supabase Authentication & Session Management**:
  - Email/password and signup flows backed by Supabase Auth (`gotrue`).
  - Access and refresh tokens encrypted and stored securely in Windows Credential Locker via OS Keyring (`DPAPI`/`WinVaultKeyring`).
  - Non-blocking asynchronous authentication worker thread with cooperative cancellation and auto-refresh.
- **Milestone 10.2: Database Schema & Local Sync Engine**:
  - Full relational PostgreSQL schema (`conversations`, `messages`, `documents`, `document_chunks`, `ai_teams`, `ai_team_runs`, `ai_team_steps`) with strict Row Level Security (RLS) policies isolating user data.
  - Supabase Storage bucket (`documents`) secured with private read/write policies (`auth.uid() = (storage.foldername(name))[1]`).
  - Transaction-safe SQLite sync queue (`sync_queue`) with exponential backoff retries, dead-letter logging, deduplication keys, and lease-based locking.
  - Background `SyncWorker` operating with authenticated user JWTs and offline resilience.
- **Milestone 10.3: Document Ingestion, Chunking & Private Storage**:
  - Multi-format document parser supporting PDF, TXT, MD, CSV, JSON, and source code files.
  - Configurable semantic chunking with configurable target chunk size (500 tokens) and overlap (50 tokens).
  - Client-side SHA-256 deduplication preventing redundant storage and indexing.
  - Supabase Storage client integration with private, authenticated bucket uploads.

## Setup & Running the Application
```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

## Running the Complete Test Suite
```powershell
python -m unittest discover -s tests
```
