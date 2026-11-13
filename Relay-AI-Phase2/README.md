# Relay AI — Chat v2

Native Windows desktop app built with Python + PySide6. This version advances the local Ollama chat experience with real-time streaming, session conversation context, safe Markdown rendering, and session control.

## Key Features in Chat v2
- **Real-Time Streaming**: Tokens stream live from Ollama's `/api/chat` endpoint.
- **Session Conversation Context**: Maintains multi-turn context (user and assistant messages) within the active chat session.
- **Markdown & Code Rendering**: Formats headings (`#`, `##`, `###`), bold, italics, inline code, bullet and numbered lists, and fenced code blocks with theme-adaptive styling. Handles unclosed code blocks during active streaming without UI disruption.
- **Clear Conversation**: Dedicated action to reset active session context and clear the transcript.
- **Responsive Asynchronous UI**: Generation runs on a dedicated `QThread` background worker, keeping the UI fully interactive and smooth.
- **Connection & State Management**: Real-time thinking/responding indicators, disabled inputs during generation to prevent overlapping queries, and graceful error recovery (unanswered prompts are rolled back from context).
- **Design & Theme Parity**: Retains full branding, dashboard metrics, sidebar navigation, and seamless light/dark appearance switching.

## Requirements
- Windows 10/11
- Python 3.11+
- Ollama installed and running (`http://localhost:11434`)
- `llama3.2:3b` model downloaded (`ollama pull llama3.2:3b`)

## Setup & Run (PowerShell)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

## Running Tests
```powershell
.\.venv\Scripts\python.exe -m unittest tests/test_chat_v2.py
```
