import json
import httpx

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"

def stream_chat(messages, model="llama3.2:3b"):
    """Yield response text chunks from Ollama's streaming chat endpoint."""
    payload = {"model": model, "messages": messages, "stream": True}
    try:
        with httpx.stream(
            "POST",
            OLLAMA_CHAT_URL,
            json=payload,
            timeout=httpx.Timeout(180.0, connect=5.0),
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                chunk = data.get("message", {}).get("content", "")
                if chunk:
                    yield chunk
                if data.get("done"):
                    break
    except httpx.ConnectError as exc:
        raise RuntimeError(
            "Cannot connect to Ollama at localhost:11434. Make sure Ollama is running."
        ) from exc
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:500]
        raise RuntimeError(f"Ollama API error ({exc.response.status_code}): {detail}") from exc
    except httpx.TimeoutException as exc:
        raise RuntimeError("Ollama took too long to respond. Please try again.") from exc
