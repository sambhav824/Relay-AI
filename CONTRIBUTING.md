# Contributing to Relay AI

Thank you for your interest in contributing to Relay AI! We welcome bug reports, feature suggestions, and code contributions.

---

## Development Workflow

### 1. Prerequisites
- Python 3.11+ (64-bit recommended)
- Node.js 18+ and npm 9+
- Optional: [Ollama](https://ollama.ai) for local on-device inference.

### 2. Environment Setup
```powershell
# Clone the repository
git clone https://github.com/sambhav824/Relay-AI.git
cd Relay-Ai

# Set up Python virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# Set up Web Frontend
cd web
npm install
cd ..
```

### 3. Running Test Suites
Before submitting pull requests, run the test suites to ensure zero regressions:

```powershell
# Run all unit and regression tests
python -m unittest discover -s tests -p "test_*.py"

# Verify web production build
cd web
npm run build
cd ..
```

---

## Coding Standards

- **Python:** PEP 8 compliance, explicit type annotations, no bare `except:` clauses.
- **TypeScript:** Strict type checking, ESLint compliance, Next.js App Router conventions.
- **Design System:** Respect the strict monochromatic design tokens:
  - Raven: `#202124`
  - Charcoal: `#36454F`
  - Pure White: `#FFFFFF`
  - Mid-tone Gray: `#757575`
- **Security:** Never commit API keys, secrets, or `.env` files. Ensure all `.env*` files remain in `.gitignore`.
