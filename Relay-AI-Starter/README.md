# Relay AI — Desktop Starter

Native Windows desktop starter built with Python and PySide6.

## Requirements
- Windows 10/11
- Python 3.11+
- Optional: Ollama for local models

## Setup (PowerShell)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
python app.py
```

## Current starter scope
- Native PySide6 desktop window
- Sidebar navigation
- Dashboard, Chat, Model Manager, History and Settings pages
- Dark/light theme toggle
- Placeholder chat interface (provider integration comes in Phase 2)

## Safety
Do not put API keys in source code or commit `.env`. Provider integrations will use OS credential storage.
