# Relay AI — Technical Architecture Specification

## Overview

Relay AI is a dual-platform (Native Windows Desktop + Cloud Web) multi-model collaboration system engineered with Python, PySide6, Next.js, and FastAPI. It unifies local inference (Ollama) with frontier cloud inference (Groq, Gemini, OpenAI) behind a single context and orchestration bus.

---

## 1. High-Level System Architecture

```mermaid
graph TD
    subgraph Client Layer
        DesktopUI["Desktop App (Python / PySide6)"]
        WebUI["Web Platform (Next.js 14 / Tailwind CSS)"]
    end

    subgraph API & Gateway Layer
        FastAPI["FastAPI Production API (Uvicorn / Render)"]
    end

    subgraph Core AI Engine
        ProviderMgr["Provider Manager Singleton"]
        SmartRouter["Deterministic Intent & Complexity Router"]
        CompareCoord["Model Comparison Coordinator"]
        TeamOrch["Autonomous AI Teams Engine"]
        ContextEngine["Universal Context Engine (UCE)"]
    end

    subgraph AI Provider Layer
        Groq["Groq LPU (GPT-OSS 120B, Qwen 3.8 27B)"]
        Gemini["Google Gemini (2.5/3.5 Flash & Pro)"]
        OpenAI["OpenAI (GPT-4o, GPT-4o-mini, o3)"]
        Ollama["Ollama Local Daemon (Llama 3.2 3B)"]
    end

    subgraph Persistence & Retrieval Layer
        SQLite[".relay/context.db (Local Transactional Outbox)"]
        Supabase["Supabase Cloud (PostgreSQL + pgvector + RLS)"]
        HybridRetriever["Hybrid RAG (RRF + Sparse FTS + Vector)"]
    end

    DesktopUI --> ProviderMgr
    DesktopUI --> SmartRouter
    DesktopUI --> CompareCoord
    DesktopUI --> TeamOrch
    DesktopUI --> ContextEngine

    WebUI --> FastAPI
    FastAPI --> ProviderMgr
    FastAPI --> SmartRouter
    FastAPI --> CompareCoord
    FastAPI --> TeamOrch

    ProviderMgr --> Groq
    ProviderMgr --> Gemini
    ProviderMgr --> OpenAI
    ProviderMgr --> Ollama

    ContextEngine --> SQLite
    ContextEngine --> HybridRetriever
    HybridRetriever --> SQLite
    HybridRetriever --> Supabase
```

---

## 2. Autonomous AI Teams: Sequential 5-Agent Pipeline

Relay AI organizes multi-agent execution in a directed, feedback-driven pipeline where each agent is constrained to its specialization:

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Planner as 🧭 Planner Agent
    participant Researcher as 🔍 Researcher Agent
    participant Creator as ✍️ Creator Agent
    participant Critic as ⚖️ Critic Agent
    participant Finalizer as ✨ Finalizer Agent

    User->>Planner: Submit complex task prompt
    activate Planner
    Planner->>Planner: Deconstruct prompt into sequential milestones & constraints
    Planner-->>Researcher: Strategic plan & research brief
    deactivate Planner

    activate Researcher
    Researcher->>Researcher: Gather domain knowledge, APIs, architectural patterns
    Researcher-->>Creator: Comprehensive factual context
    deactivate Researcher

    activate Creator
    Creator->>Creator: Synthesize full initial draft & code implementation
    Creator-->>Critic: Complete draft deliverable
    deactivate Creator

    activate Critic
    Critic->>Critic: Audit against requirements, performance, & safety
    Critic-->>Finalizer: Peer review critiques & refinement recommendations
    deactivate Critic

    activate Finalizer
    Finalizer->>Finalizer: Polish syntax, reconcile critiques, format production deliverable
    Finalizer-->>User: Complete production-ready deliverable
    deactivate Finalizer
```

---

## 3. Intelligent Model Routing

The `ModelRouter` evaluates user prompts using deterministic lexical cues and complexity scoring:

- **Coding Tasks (`TaskType.CODING`)**: Dispatches to models with high code benchmarks and long context (e.g., `openai/gpt-oss-120b` or `gpt-4o-mini`).
- **Complex Reasoning (`TaskType.COMPLEX_REASONING`)**: Routes to high-depth reasoning engines.
- **Vision Tasks (`TaskType.VISION`)**: Automatically identifies base64 image attachments and filters for models with native multimodal support (`gemini-2.5-flash`, `gpt-4o-mini`).
- **Local-Only Guardrail**: When enabled, guarantees queries never exit the local network, isolating execution to local Ollama inference.

---

## 4. Security & Credential Isolation

- **OS Keyring Integration (`WinVaultKeyring` / `DPAPI`)**: Windows desktop stores all API keys inside the encrypted Windows Credential Locker. Plaintext keys are never stored in `.env` files, configuration files, or logs.
- **Server-Side Environment Fallback**: On Linux / Render deployment, `CredentialStore` falls back to process environment variables (`GROQ_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY`) without client exposure.
- **Zero Frontend Leakage**: The Next.js frontend interacts strictly via backend API proxy or public anon keys; secret tokens (`SUPABASE_SERVICE_ROLE_KEY`) are never exposed to browser bundles.
