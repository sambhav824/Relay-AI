// Relay AI Universal Web Client Library
// Cross-Device Production Architecture with Resilient Backend Routing

import { supabase } from "./supabase";

/**
 * Verified production Render backend endpoint for Relay AI.
 */
export const PRODUCTION_RENDER_BACKEND = "https://relay-ai-0r5f.onrender.com";

/**
 * Checks whether a valid backend API target is available for this client.
 */
export function isBackendConfigured(): boolean {
  if (typeof window !== "undefined") {
    try {
      const custom = localStorage.getItem("relay_backend_url");
      if (custom && custom.trim() && !custom.includes("relay-ai-x92e.onrender.com")) return true;
    } catch {}

    const hostname = window.location.hostname;
    if (
      hostname === "localhost" ||
      hostname === "127.0.0.1" ||
      hostname.endsWith(".local") ||
      hostname.startsWith("192.168.") ||
      hostname.startsWith("10.") ||
      hostname.startsWith("172.16.")
    ) {
      return true;
    }

    const envUrl = process.env.NEXT_PUBLIC_API_URL;
    if (envUrl && envUrl.trim() && !envUrl.includes("localhost") && !envUrl.includes("127.0.0.1")) {
      return true;
    }
    // Verified production Render backend
    if (PRODUCTION_RENDER_BACKEND) {
      return true;
    }
    return false;
  }
  return Boolean(process.env.NEXT_PUBLIC_API_URL || PRODUCTION_RENDER_BACKEND);
}

/**
 * Dynamically resolves the active Backend API base URL:
 * 1. Checks localStorage for a user-specified custom URL (useful for testing Render services / local IPs).
 * 2. Checks process.env.NEXT_PUBLIC_API_URL (provided at Vercel build time).
 * 3. On browser loopback (localhost / 127.0.0.1) -> defaults to "http://localhost:8000".
 * 4. On local network Wi-Fi (192.168.x.x / 10.x.x.x) -> dynamically points to that device's port 8000.
 * 5. On public production domains (*.vercel.app / custom domain) -> connects to the verified production Render service.
 */
export function getApiBase(): string {
  if (typeof window !== "undefined") {
    try {
      const custom = localStorage.getItem("relay_backend_url");
      if (custom && custom.trim()) {
        const cleaned = custom.trim().replace(/\/+$/, "");
        // Clean up outdated or invalid legacy Render endpoints
        if (cleaned.includes("relay-ai-x92e.onrender.com")) {
          localStorage.removeItem("relay_backend_url");
        } else {
          return cleaned;
        }
      }
    } catch {}

    const hostname = window.location.hostname;
    // Local loopback
    if (hostname === "localhost" || hostname === "127.0.0.1") {
      return process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    }

    // Local area network Wi-Fi testing (e.g. 192.168.x.x, 10.x.x.x, or .local)
    if (
      hostname.startsWith("192.168.") ||
      hostname.startsWith("10.") ||
      hostname.startsWith("172.16.") ||
      hostname.endsWith(".local")
    ) {
      return process.env.NEXT_PUBLIC_API_URL || `http://${hostname}:8000`;
    }

    // Public production environment (Vercel, custom domains)
    const envUrl = process.env.NEXT_PUBLIC_API_URL;
    if (
      envUrl &&
      envUrl.trim() &&
      !envUrl.includes("localhost") &&
      !envUrl.includes("127.0.0.1")
    ) {
      return envUrl.trim().replace(/\/+$/, "");
    }

    return PRODUCTION_RENDER_BACKEND;
  }

  const envUrl = process.env.NEXT_PUBLIC_API_URL;
  if (envUrl && envUrl.trim() && !envUrl.includes("localhost")) {
    return envUrl.trim().replace(/\/+$/, "");
  }
  return PRODUCTION_RENDER_BACKEND;
}

export function setCustomApiBase(url: string): void {
  if (typeof window !== "undefined") {
    try {
      if (!url || !url.trim()) {
        localStorage.removeItem("relay_backend_url");
      } else {
        localStorage.setItem("relay_backend_url", url.trim().replace(/\/+$/, ""));
      }
    } catch {}
  }
}

export function resetApiBase(): void {
  if (typeof window !== "undefined") {
    try {
      localStorage.removeItem("relay_backend_url");
    } catch {}
  }
}

export interface ModelOption {
  id: string;
  provider: string;
  provider_name: string;
  display_name: string;
  context_window: number;
  is_local: boolean;
  configured: boolean;
  supports_vision: boolean;
}

export interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
  images?: string[];
}

export interface ChatSession {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  provider: string;
  model: string;
  messages: ChatMessage[];
}

export interface RouteResponse {
  task_type: string;
  complexity: string;
  confidence: number;
  recommended_provider: string;
  recommended_model: string;
  reasoning: string;
}

export interface CompareResult {
  provider: string;
  model: string;
  status: "success" | "error";
  content: string;
  duration_sec: number;
  tokens?: number;
  speed_tps?: number;
  error?: string;
}

export interface TeamStep {
  role_id: string;
  role_name: string;
  icon: string;
  status: "completed" | "error" | "running";
  output: string;
  duration_sec: number;
  provider: string;
  model: string;
}

/**
 * Format cross-device network exceptions into clear, actionable messages.
 */
function formatNetworkError(err: unknown): string {
  const base = getApiBase();
  const rawMsg = err instanceof Error ? err.message : String(err);
  if (!base) {
    return "Backend API URL is not configured. Please enter your Render backend service URL in Settings or configure NEXT_PUBLIC_API_URL in Vercel.";
  }
  if (rawMsg.includes("Failed to fetch") || rawMsg.includes("NetworkError") || rawMsg.includes("fetch failed")) {
    if (typeof window !== "undefined" && window.location.hostname !== "localhost" && base.includes("localhost")) {
      return `Cross-Device Network Error: Cannot reach "${base}" from ${window.location.hostname}. Please open Settings and update your deployed Render backend URL.`;
    }
    return `Unable to reach backend at ${base}. If using a Render free-tier instance, it may be waking up (cold start ~30-50s). Please check your Render URL or test connection in Settings.`;
  }
  return rawMsg;
}

/**
 * Diagnostic ping to test live backend connectivity and measure latency.
 */
export async function pingBackend(customUrl?: string): Promise<{
  ok: boolean;
  status: number;
  latencyMs: number;
  apiUrl: string;
  error?: string;
  data?: any;
}> {
  const base = (customUrl !== undefined ? customUrl : getApiBase()).replace(/\/+$/, "");
  if (!base) {
    return {
      ok: false,
      status: 0,
      latencyMs: 0,
      apiUrl: "",
      error: "Backend API URL is empty. Please supply a valid URL (e.g. https://your-service.onrender.com).",
    };
  }

  const start = Date.now();
  try {
    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), 12000);
    const res = await fetch(`${base}/health`, {
      signal: controller.signal,
      cache: "no-store",
    });
    clearTimeout(id);
    const latencyMs = Date.now() - start;
    if (res.ok) {
      const data = await res.json().catch(() => ({}));
      return { ok: true, status: res.status, latencyMs, apiUrl: base, data };
    }
    return {
      ok: false,
      status: res.status,
      latencyMs,
      apiUrl: base,
      error: `HTTP ${res.status} ${res.statusText}`,
    };
  } catch (err: any) {
    const latencyMs = Date.now() - start;
    return {
      ok: false,
      status: 0,
      latencyMs,
      apiUrl: base,
      error: formatNetworkError(err),
    };
  }
}

export async function fetchHealth() {
  const base = getApiBase();
  if (!base) {
    return {
      status: "unconfigured",
      app: "Relay AI Web Platform",
      version: "1.0.0",
      api_url: "Not Configured",
      error: "Backend API URL is not set. Please configure in Settings.",
    };
  }

  try {
    const res = await fetch(`${base}/health`, { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
    return res.json();
  } catch (err) {
    return {
      status: "offline",
      app: "Relay AI (Cloud Disconnected)",
      version: "1.0.0",
      api_url: base,
      error: formatNetworkError(err),
    };
  }
}

export async function fetchModels(): Promise<ModelOption[]> {
  const base = getApiBase();
  if (!base) {
    return [
      {
        id: "openai/gpt-oss-120b",
        provider: "groq",
        provider_name: "Groq LPU",
        display_name: "GPT-OSS 120B",
        context_window: 131072,
        is_local: false,
        configured: true,
        supports_vision: false,
      },
      {
        id: "qwen/qwen3.8-27b",
        provider: "groq",
        provider_name: "Groq LPU",
        display_name: "Qwen 3.8 27B",
        context_window: 131072,
        is_local: false,
        configured: true,
        supports_vision: true,
      },
      {
        id: "gemini-3.5-flash-lite",
        provider: "gemini",
        provider_name: "Google Gemini",
        display_name: "Gemini 3.5 Flash Lite",
        context_window: 1048576,
        is_local: false,
        configured: true,
        supports_vision: true,
      },
      {
        id: "gpt-4o-mini",
        provider: "openai",
        provider_name: "OpenAI",
        display_name: "GPT-4o Mini",
        context_window: 128000,
        is_local: false,
        configured: true,
        supports_vision: true,
      },
    ];
  }

  try {
    const res = await fetch(`${base}/api/models`, { cache: "no-store" });
    if (!res.ok) throw new Error("Failed to fetch models");
    const data = await res.json();
    return data.models || [];
  } catch {
    return [
      {
        id: "openai/gpt-oss-120b",
        provider: "groq",
        provider_name: "Groq LPU",
        display_name: "GPT-OSS 120B",
        context_window: 131072,
        is_local: false,
        configured: true,
        supports_vision: false,
      },
      {
        id: "qwen/qwen3.8-27b",
        provider: "groq",
        provider_name: "Groq LPU",
        display_name: "Qwen 3.8 27B",
        context_window: 131072,
        is_local: false,
        configured: true,
        supports_vision: true,
      },
      {
        id: "gemini-3.5-flash-lite",
        provider: "gemini",
        provider_name: "Google Gemini",
        display_name: "Gemini 3.5 Flash Lite",
        context_window: 1048576,
        is_local: false,
        configured: true,
        supports_vision: true,
      },
      {
        id: "gpt-4o-mini",
        provider: "openai",
        provider_name: "OpenAI",
        display_name: "GPT-4o Mini",
        context_window: 128000,
        is_local: false,
        configured: true,
        supports_vision: true,
      },
    ];
  }
}

export async function streamChat(
  messages: ChatMessage[],
  provider: string,
  model: string,
  onDelta: (delta: string) => void,
  onDone: () => void,
  onError: (err: string) => void
) {
  const base = getApiBase();
  if (!base) {
    onError("Backend API URL is not configured. Please enter your Render backend URL in Settings.");
    return;
  }

  try {
    const res = await fetch(`${base}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages, provider, model, stream: true }),
    });

    if (!res.ok) {
      const err = await res.text().catch(() => "HTTP Error " + res.status);
      onError(err || `Failed to communicate with provider (${res.status})`);
      return;
    }

    const reader = res.body?.getReader();
    if (!reader) {
      onError("Streaming not supported by browser");
      return;
    }

    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          try {
            const data = JSON.parse(line.slice(6));
            if (data.delta) {
              onDelta(data.delta);
            }
            if (data.is_final) {
              onDone();
            }
            if (data.error) {
              onError(data.error);
            }
          } catch {
            // parse error
          }
        }
      }
    }
    onDone();
  } catch (err: unknown) {
    onError(formatNetworkError(err));
  }
}

export async function routeQuery(prompt: string): Promise<RouteResponse> {
  const base = getApiBase();
  if (!base) {
    throw new Error("Backend API URL is not configured. Please enter your Render backend URL in Settings.");
  }
  try {
    const res = await fetch(`${base}/api/route`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt }),
    });
    if (!res.ok) {
      const err = await res.text().catch(() => "HTTP Error " + res.status);
      throw new Error(err || "Routing request failed");
    }
    return res.json();
  } catch (err) {
    throw new Error(formatNetworkError(err));
  }
}

export async function compareModels(
  prompt: string,
  slots: { provider: string; model: string }[]
): Promise<{ summary?: string; results?: CompareResult[] }> {
  const base = getApiBase();
  if (!base) {
    throw new Error("Backend API URL is not configured. Please enter your Render backend URL in Settings.");
  }
  try {
    const res = await fetch(`${base}/api/compare`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, slots }),
    });
    if (!res.ok) {
      const err = await res.text().catch(() => "HTTP Error " + res.status);
      throw new Error(err || "Comparison request failed");
    }
    return res.json();
  } catch (err) {
    throw new Error(formatNetworkError(err));
  }
}

export async function runAITeam(
  task: string,
  provider: string = "groq",
  model: string = "openai/gpt-oss-120b"
): Promise<{ deliverable?: string; final_deliverable?: string; steps?: TeamStep[]; duration_sec?: number }> {
  const base = getApiBase();
  if (!base) {
    throw new Error("Backend API URL is not configured. Please enter your Render backend URL in Settings.");
  }
  try {
    const res = await fetch(`${base}/api/teams/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task, provider, model }),
    });
    if (!res.ok) {
      const err = await res.text().catch(() => "HTTP Error " + res.status);
      throw new Error(err || "Team run failed");
    }
    const data = await res.json();
    if (data.final_deliverable && !data.deliverable) {
      data.deliverable = data.final_deliverable;
    }
    return data;
  } catch (err) {
    throw new Error(formatNetworkError(err));
  }
}

export async function authLogin(email: string, pass: string) {
  if (supabase) {
    const { data, error } = await supabase.auth.signInWithPassword({
      email,
      password: pass,
    });
    if (error) throw new Error(error.message);
    if (data.user) {
      return {
        user: { id: data.user.id, email: data.user.email || email },
        token: data.session?.access_token,
        notice: "Authenticated with Supabase Cloud Auth.",
      };
    }
  }

  const base = getApiBase();
  if (!base) {
    throw new Error("Backend API URL is not configured. Please enter your Render backend URL in Settings or configure NEXT_PUBLIC_API_URL in Vercel.");
  }

  try {
    const res = await fetch(`${base}/api/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password: pass }),
    });
    if (res.ok) {
      return res.json();
    }
    const err = await res.json().catch(() => ({ detail: "Authentication failed" }));
    throw new Error(err.detail || `Authentication failed (HTTP ${res.status})`);
  } catch (err: any) {
    throw new Error(formatNetworkError(err));
  }
}

export async function authSignup(email: string, pass: string) {
  if (supabase) {
    const { data, error } = await supabase.auth.signUp({
      email,
      password: pass,
    });
    if (error) throw new Error(error.message);
    if (data.user) {
      return {
        user: { id: data.user.id, email: data.user.email || email },
        token: data.session?.access_token,
        message: "Sign up successful! Launching workspace...",
      };
    }
  }

  const base = getApiBase();
  if (!base) {
    throw new Error("Backend API URL is not configured. Please enter your Render backend URL in Settings or configure NEXT_PUBLIC_API_URL in Vercel.");
  }

  try {
    const res = await fetch(`${base}/api/auth/signup`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password: pass }),
    });
    if (res.ok) {
      return res.json();
    }
    const err = await res.json().catch(() => ({ detail: "Sign up failed" }));
    throw new Error(err.detail || `Sign up failed (HTTP ${res.status})`);
  } catch (err: any) {
    throw new Error(formatNetworkError(err));
  }
}

export async function authSignOut() {
  if (supabase) {
    await supabase.auth.signOut().catch(() => {});
  }
  if (typeof window !== "undefined") {
    localStorage.removeItem("relay_user");
    localStorage.removeItem("relay_token");
  }
}

export async function fetchSavedConversations(): Promise<ChatSession[]> {
  const base = getApiBase();
  if (!base) return [];
  try {
    const res = await fetch(`${base}/api/conversations`, { cache: "no-store" });
    if (!res.ok) return [];
    const data = await res.json();
    return data.conversations || [];
  } catch {
    return [];
  }
}

export async function saveConversationApi(session: ChatSession): Promise<boolean> {
  const base = getApiBase();
  if (!base) return false;
  try {
    const res = await fetch(`${base}/api/conversations`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(session),
    });
    return res.ok;
  } catch {
    return false;
  }
}

export async function deleteConversationApi(conversationId: string): Promise<boolean> {
  const base = getApiBase();
  if (!base) return false;
  try {
    const res = await fetch(`${base}/api/conversations/${conversationId}`, {
      method: "DELETE",
    });
    return res.ok;
  } catch {
    return false;
  }
}
