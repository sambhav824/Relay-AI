# Relay AI Chat v2

This version improves the local Ollama chat experience:
- Streaming responses (text appears as it is generated)
- Conversation context within the current chat session
- Markdown rendering for headings, lists and code blocks
- Clear chat action
- Background worker keeps the UI responsive

## Requirements
- Windows 10/11
- Python 3.11+
- Ollama running locally
- `llama3.2:3b` installed (`ollama list`)

## Run (PowerShell)
Open PowerShell in the extracted project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

If Python 3.11 is not installed but `python --version` reports a supported Python 3 version, you can use:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## Notes
- This version keeps conversation messages in memory for the current running session. Persistent SQLite history is a later milestone.
- The local Ollama API is expected at `http://localhost:11434`.
- Do not expose Ollama's local API to untrusted networks.
