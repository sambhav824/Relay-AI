// Relay AI Web Client Library

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

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

export async function fetchHealth() {
  const res = await fetch(`${API_BASE}/health`, { cache: "no-store" });
  if (!res.ok) throw new Error("API offline");
  return res.json();
}

export async function fetchModels(): Promise<ModelOption[]> {
  try {
    const res = await fetch(`${API_BASE}/api/models`, { cache: "no-store" });
    if (!res.ok) throw new Error("Failed to fetch models");
    const data = await res.json();
    return data.models || [];
  } catch {
    // Return sensible defaults if backend offline
    return [
      { id: "openai/gpt-oss-120b", provider: "groq", provider_name: "Groq LPU", display_name: "GPT-OSS 120B", context_window: 131072, is_local: false, configured: true, supports_vision: false },
      { id: "qwen/qwen3.8-27b", provider: "groq", provider_name: "Groq LPU", display_name: "Qwen 3.8 27B", context_window: 131072, is_local: false, configured: true, supports_vision: true },
      { id: "gemini-3.5-flash-lite", provider: "gemini", provider_name: "Google Gemini", display_name: "Gemini 3.5 Flash Lite", context_window: 1048576, is_local: false, configured: true, supports_vision: true },
      { id: "gpt-4o-mini", provider: "openai", provider_name: "OpenAI", display_name: "GPT-4o Mini", context_window: 128000, is_local: false, configured: true, supports_vision: true },
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
  try {
    const res = await fetch(`${API_BASE}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages, provider, model, stream: true }),
    });

    if (!res.ok) {
      const err = await res.text();
      onError(err || "Failed to communicate with provider");
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
    const msg = err instanceof Error ? err.message : "Network error";
    onError(msg);
  }
}

export async function routeQuery(prompt: string): Promise<RouteResponse> {
  const res = await fetch(`${API_BASE}/api/route`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt }),
  });
  if (!res.ok) throw new Error("Routing request failed");
  return res.json();
}

export async function compareModels(prompt: string, slots: { provider: string; model: string }[]) {
  const res = await fetch(`${API_BASE}/api/compare`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, slots }),
  });
  if (!res.ok) throw new Error("Comparison request failed");
  return res.json();
}

export async function runAITeam(task: string, provider: string = "groq", model: string = "openai/gpt-oss-120b") {
  const res = await fetch(`${API_BASE}/api/teams/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ task, provider, model }),
  });
  if (!res.ok) throw new Error("Team run failed");
  return res.json();
}

import { supabase } from "./supabase";

export async function authLogin(email: string, pass: string) {
  if (supabase) {
    const { data, error } = await supabase.auth.signInWithPassword({
      email,
      password: pass,
    });
    if (error) throw new Error(error.message);
    return {
      user: { id: data.user?.id || "usr_remote", email: data.user?.email || email },
      token: data.session?.access_token,
      notice: "Authenticated with Supabase Cloud Auth.",
    };
  }

  const res = await fetch(`${API_BASE}/api/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password: pass }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Authentication failed");
  }
  return res.json();
}

export async function authSignup(email: string, pass: string) {
  if (supabase) {
    const { data, error } = await supabase.auth.signUp({
      email,
      password: pass,
    });
    if (error) throw new Error(error.message);
    return {
      user: { id: data.user?.id || "usr_remote", email: data.user?.email || email },
      token: data.session?.access_token,
      message: "Sign up successful! You can now sign in.",
    };
  }

  const res = await fetch(`${API_BASE}/api/auth/signup`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password: pass }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Sign up failed");
  }
  return res.json();
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
  try {
    const res = await fetch(`${API_BASE}/api/conversations`, { cache: "no-store" });
    if (!res.ok) return [];
    const data = await res.json();
    return data.conversations || [];
  } catch {
    return [];
  }
}

export async function saveConversationApi(session: ChatSession): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/api/conversations`, {
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
  try {
    const res = await fetch(`${API_BASE}/api/conversations/${conversationId}`, {
      method: "DELETE",
    });
    return res.ok;
  } catch {
    return false;
  }
}
