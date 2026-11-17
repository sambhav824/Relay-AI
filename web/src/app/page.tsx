"use client";

import React, { useState, useEffect, useRef } from "react";
import {
  fetchHealth,
  fetchModels,
  streamChat,
  routeQuery,
  compareModels,
  runAITeam,
  authLogin,
  ModelOption,
  ChatMessage,
  RouteResponse,
  CompareResult,
  TeamStep,
} from "../lib/api";

type Tab = "dashboard" | "chat" | "routing" | "compare" | "teams" | "settings";

interface HealthData {
  status?: string;
  app?: string;
  version?: string;
  timestamp?: number;
  providers?: Record<string, { name?: string; configured?: boolean; cloud?: boolean }>;
  supabase_configured?: boolean;
}

export default function RelayAIWeb() {
  const [activeTab, setActiveTab] = useState<Tab>("dashboard");
  const [isDarkMode, setIsDarkMode] = useState<boolean>(true);
  const [sidebarOpen, setSidebarOpen] = useState<boolean>(true);

  // Auth State
  const [user, setUser] = useState<{ id: string; email: string } | null>({
    id: "usr_guest",
    email: "guest@relay.ai",
  });
  const [showAuthModal, setShowAuthModal] = useState<boolean>(false);
  const [authEmail, setAuthEmail] = useState<string>("");
  const [authPassword, setAuthPassword] = useState<string>("");
  const [authLoading, setAuthLoading] = useState<boolean>(false);
  const [authNotice, setAuthNotice] = useState<string>("");

  // Health & Models
  const [healthData, setHealthData] = useState<HealthData | null>(null);
  const [models, setModels] = useState<ModelOption[]>([]);
  const [selectedProvider, setSelectedProvider] = useState<string>("groq");
  const [selectedModel, setSelectedModel] = useState<string>("openai/gpt-oss-120b");

  // Chat State
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    {
      role: "assistant",
      content: "Hello! I am Relay AI, your universal multi-model workspace. How can I assist you today?",
    },
  ]);
  const [chatInput, setChatInput] = useState<string>("");
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Routing State
  const [routingPrompt, setRoutingPrompt] = useState<string>("");
  const [routingResult, setRoutingResult] = useState<RouteResponse | null>(null);
  const [routingLoading, setRoutingLoading] = useState<boolean>(false);

  // Compare State
  const [comparePrompt, setComparePrompt] = useState<string>("");
  const [compareSlots] = useState<{ provider: string; model: string }[]>([
    { provider: "groq", model: "openai/gpt-oss-120b" },
    { provider: "gemini", model: "gemini-2.5-flash" },
  ]);
  const [compareResults, setCompareResults] = useState<CompareResult[]>([]);
  const [compareLoading, setCompareLoading] = useState<boolean>(false);
  const [compareSummary, setCompareSummary] = useState<string>("");

  // AI Teams State
  const [teamTask, setTeamTask] = useState<string>("");
  const [teamSteps, setTeamSteps] = useState<TeamStep[]>([]);
  const [teamDeliverable, setTeamDeliverable] = useState<string>("");
  const [teamRunning, setTeamRunning] = useState<boolean>(false);

  // Initialize
  useEffect(() => {
    fetchHealth().then(setHealthData).catch(() => {});
    fetchModels().then((m) => {
      setModels(m);
      if (m.length > 0) {
        setSelectedProvider(m[0].provider);
        setSelectedModel(m[0].id);
      }
    });
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatMessages, isStreaming]);

  // Handle Send Chat
  const handleSendChat = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!chatInput.trim() || isStreaming) return;

    const userMsg: ChatMessage = { role: "user", content: chatInput.trim() };
    const newHistory = [...chatMessages, userMsg];
    setChatMessages(newHistory);
    setChatInput("");
    setIsStreaming(true);

    const assistantMsg: ChatMessage = { role: "assistant", content: "" };
    setChatMessages([...newHistory, assistantMsg]);

    await streamChat(
      newHistory,
      selectedProvider,
      selectedModel,
      (delta) => {
        setChatMessages((prev) => {
          const updated = [...prev];
          const lastIdx = updated.length - 1;
          if (lastIdx >= 0 && updated[lastIdx].role === "assistant") {
            updated[lastIdx] = {
              ...updated[lastIdx],
              content: updated[lastIdx].content + delta,
            };
          }
          return updated;
        });
      },
      () => setIsStreaming(false),
      (err) => {
        setChatMessages((prev) => {
          const updated = [...prev];
          const lastIdx = updated.length - 1;
          if (lastIdx >= 0) {
            updated[lastIdx] = {
              ...updated[lastIdx],
              content: updated[lastIdx].content + `\n\n*[Error: ${err}]*`,
            };
          }
          return updated;
        });
        setIsStreaming(false);
      }
    );
  };

  // Handle Smart Routing
  const handleRunRouting = async () => {
    if (!routingPrompt.trim() || routingLoading) return;
    setRoutingLoading(true);
    setRoutingResult(null);
    try {
      const res = await routeQuery(routingPrompt);
      setRoutingResult(res);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Routing failed";
      alert("Routing failed: " + msg);
    } finally {
      setRoutingLoading(false);
    }
  };

  // Handle Model Compare
  const handleRunCompare = async () => {
    if (!comparePrompt.trim() || compareLoading) return;
    setCompareLoading(true);
    setCompareResults([]);
    setCompareSummary("");
    try {
      const data = await compareModels(comparePrompt, compareSlots);
      setCompareResults(data.results || []);
      setCompareSummary(data.summary || "");
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Comparison failed";
      alert("Comparison failed: " + msg);
    } finally {
      setCompareLoading(false);
    }
  };

  // Handle AI Teams
  const handleRunTeam = async () => {
    if (!teamTask.trim() || teamRunning) return;
    setTeamRunning(true);
    setTeamSteps([]);
    setTeamDeliverable("");
    try {
      const data = await runAITeam(teamTask, selectedProvider, selectedModel);
      setTeamSteps(data.steps || []);
      setTeamDeliverable(data.final_deliverable || "");
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "AI Team execution failed";
      alert("AI Team execution failed: " + msg);
    } finally {
      setTeamRunning(false);
    }
  };

  // Handle Auth Login
  const handleAuthSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setAuthLoading(true);
    setAuthNotice("");
    try {
      const res = await authLogin(authEmail, authPassword);
      setUser(res.user);
      setShowAuthModal(false);
      if (res.notice) {
        alert(res.notice);
      }
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Failed to sign in";
      setAuthNotice(msg);
    } finally {
      setAuthLoading(false);
    }
  };

  return (
    <div className={`min-h-screen flex flex-col md:flex-row ${isDarkMode ? "bg-[#202124] text-white" : "bg-[#F8F9FA] text-[#202124]"}`}>
      {/* Sidebar Navigation */}
      <aside
        className={`${
          sidebarOpen ? "w-64" : "w-16"
        } transition-all duration-200 border-r ${
          isDarkMode ? "bg-[#1b1c1e] border-[#36373b]" : "bg-[#EEEEEE] border-[#E0E0E0]"
        } flex flex-col justify-between p-4 shrink-0`}
      >
        <div>
          {/* Brand */}
          <div className="flex items-center justify-between mb-6 pb-2 border-b border-gray-700/30">
            {sidebarOpen ? (
              <div>
                <h1 className="text-lg font-bold tracking-tight flex items-center gap-2">
                  <span className="text-sm">✦</span> RELAY AI
                </h1>
                <p className="text-[11px] text-gray-400">Think Together. Work Smarter.</p>
              </div>
            ) : (
              <span className="text-xl mx-auto">✦</span>
            )}
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-1 rounded hover:bg-gray-700/20 text-xs hidden md:block"
              title="Toggle sidebar"
            >
              {sidebarOpen ? "◀" : "▶"}
            </button>
          </div>

          {/* Navigation Items */}
          <nav className="space-y-1">
            {[
              { id: "dashboard", label: "Dashboard", icon: "⌂" },
              { id: "chat", label: "AI Chat", icon: "✧" },
              { id: "routing", label: "Smart Route", icon: "✦" },
              { id: "compare", label: "Compare Models", icon: "⚖" },
              { id: "teams", label: "AI Teams", icon: "👥" },
              { id: "settings", label: "Settings", icon: "⚙" },
            ].map((item) => (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id as Tab)}
                className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition ${
                  activeTab === item.id
                    ? isDarkMode
                      ? "bg-[#36454F] text-white font-semibold"
                      : "bg-[#202124] text-white"
                    : isDarkMode
                    ? "text-gray-300 hover:bg-[#28292c]"
                    : "text-gray-700 hover:bg-gray-200"
                }`}
              >
                <span className="text-base">{item.icon}</span>
                {sidebarOpen && <span>{item.label}</span>}
              </button>
            ))}
          </nav>
        </div>

        {/* User & Theme Footer */}
        <div className="pt-4 border-t border-gray-700/30 space-y-2">
          {sidebarOpen && (
            <div className="flex items-center justify-between px-2 py-1.5 text-xs text-gray-400">
              <span className="truncate">{user ? user.email : "Not signed in"}</span>
              <button
                onClick={() => setShowAuthModal(true)}
                className="text-xs underline hover:text-white"
              >
                {user ? "Account" : "Sign In"}
              </button>
            </div>
          )}
          <button
            onClick={() => setIsDarkMode(!isDarkMode)}
            className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg text-xs font-medium border border-gray-600/40 hover:bg-gray-700/20"
          >
            <span>{isDarkMode ? "☼ Light Mode" : "☽ Dark Mode"}</span>
          </button>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col h-screen overflow-y-auto">
        {/* Top Header */}
        <header
          className={`h-14 border-b flex items-center justify-between px-6 shrink-0 ${
            isDarkMode ? "bg-[#202124] border-[#36373b]" : "bg-white border-[#E0E0E0]"
          }`}
        >
          <div className="flex items-center gap-3">
            <h2 className="text-base font-semibold capitalize">
              {activeTab === "chat"
                ? "AI Chat"
                : activeTab === "routing"
                ? "Smart Routing Engine"
                : activeTab === "compare"
                ? "Side-by-Side Model Comparison"
                : activeTab === "teams"
                ? "Autonomous Multi-Agent Teams"
                : activeTab === "settings"
                ? "Settings & System Status"
                : "Overview & Workspaces"}
            </h2>
            <span className="px-2 py-0.5 rounded text-[11px] font-mono border border-gray-600/40 text-gray-400">
              {healthData?.providers?.groq?.configured ? "✦ Cloud Active" : "Local Ready"}
            </span>
          </div>

          <div className="flex items-center gap-3">
            {/* Quick Provider Picker in Chat */}
            {activeTab === "chat" && (
              <div className="flex items-center gap-2 text-xs">
                <select
                  value={selectedProvider}
                  onChange={(e) => {
                    setSelectedProvider(e.target.value);
                    const matching = models.filter((m) => m.provider === e.target.value);
                    if (matching.length > 0) setSelectedModel(matching[0].id);
                  }}
                  className={`px-2.5 py-1 rounded border text-xs ${
                    isDarkMode
                      ? "bg-[#28292c] border-[#36373b] text-white"
                      : "bg-white border-gray-300 text-gray-900"
                  }`}
                >
                  <option value="groq">Groq LPU</option>
                  <option value="gemini">Google Gemini</option>
                  <option value="openai">OpenAI</option>
                  <option value="ollama">Ollama (Local)</option>
                </select>

                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className={`px-2.5 py-1 rounded border text-xs max-w-[180px] truncate ${
                    isDarkMode
                      ? "bg-[#28292c] border-[#36373b] text-white"
                      : "bg-white border-gray-300 text-gray-900"
                  }`}
                >
                  {models
                    .filter((m) => m.provider === selectedProvider)
                    .map((m) => (
                      <option key={m.id} value={m.id}>
                        {m.display_name || m.id}
                      </option>
                    ))}
                </select>
              </div>
            )}
          </div>
        </header>

        {/* View Router */}
        <div className="flex-1 p-6 max-w-6xl w-full mx-auto">
          {/* 1. DASHBOARD */}
          {activeTab === "dashboard" && (
            <div className="space-y-6">
              <div className="p-6 rounded-xl border border-gray-700/40 bg-gradient-to-br from-gray-800/40 to-transparent">
                <h3 className="text-xl font-bold mb-2">Welcome to Relay AI Web</h3>
                <p className="text-sm text-gray-400 max-w-2xl">
                  Unified AI collaboration platform. Chat with frontier models, intelligently auto-route queries based on complexity, compare multi-model outputs concurrently, and deploy autonomous 5-agent AI teams.
                </p>
                <div className="mt-4 flex gap-3">
                  <button
                    onClick={() => setActiveTab("chat")}
                    className="px-4 py-2 rounded-lg bg-white text-black font-semibold text-xs hover:bg-gray-200 transition"
                  >
                    Open AI Chat →
                  </button>
                  <button
                    onClick={() => setActiveTab("teams")}
                    className="px-4 py-2 rounded-lg border border-gray-600 text-xs font-semibold hover:bg-gray-800/40 transition"
                  >
                    Launch AI Teams
                  </button>
                </div>
              </div>

              {/* Status Cards */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="p-4 rounded-xl border border-gray-700/40 bg-[#28292c]/50">
                  <div className="text-xs font-medium text-gray-400 mb-1">CONNECTED PROVIDERS</div>
                  <div className="text-base font-semibold">Groq, Gemini & OpenAI</div>
                  <div className="text-xs text-green-400 mt-2 flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-green-500"></span> Active Cloud Infrastructure
                  </div>
                </div>

                <div className="p-4 rounded-xl border border-gray-700/40 bg-[#28292c]/50">
                  <div className="text-xs font-medium text-gray-400 mb-1">SMART ROUTER</div>
                  <div className="text-base font-semibold">Intent & Complexity Engine</div>
                  <div className="text-xs text-gray-400 mt-2">Cost & Accuracy Balanced</div>
                </div>

                <div className="p-4 rounded-xl border border-gray-700/40 bg-[#28292c]/50">
                  <div className="text-xs font-medium text-gray-400 mb-1">SECURITY & PRIVACY</div>
                  <div className="text-base font-semibold">Zero Telemetry</div>
                  <div className="text-xs text-gray-400 mt-2">Server-Side Key Vault</div>
                </div>
              </div>

              {/* Quick Actions Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
                <div
                  onClick={() => setActiveTab("routing")}
                  className="p-5 rounded-xl border border-gray-700/40 hover:border-gray-500/60 cursor-pointer transition bg-[#28292c]/30"
                >
                  <div className="text-lg font-bold mb-1">✦ Smart Routing Analysis</div>
                  <p className="text-xs text-gray-400">
                    Submit prompts to automatically analyze task domain (Code, Reasoning, Creative) and dispatch to the optimal model.
                  </p>
                </div>

                <div
                  onClick={() => setActiveTab("compare")}
                  className="p-5 rounded-xl border border-gray-700/40 hover:border-gray-500/60 cursor-pointer transition bg-[#28292c]/30"
                >
                  <div className="text-lg font-bold mb-1">⚖ Side-by-Side Model Comparison</div>
                  <p className="text-xs text-gray-400">
                    Compare responses from multiple models in parallel with real-time token per second speed metrics.
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* 2. AI CHAT */}
          {activeTab === "chat" && (
            <div className="flex flex-col h-[calc(100vh-8.5rem)]">
              {/* Message Transcript Area */}
              <div className="flex-1 overflow-y-auto space-y-4 pr-2">
                {chatMessages.map((msg, idx) => (
                  <div
                    key={idx}
                    className={`flex gap-3 ${
                      msg.role === "user" ? "justify-end" : "justify-start"
                    }`}
                  >
                    {msg.role === "assistant" && (
                      <div className="w-7 h-7 rounded-full bg-white text-black font-bold text-xs flex items-center justify-center shrink-0">
                        ✦
                      </div>
                    )}
                    <div
                      className={`max-w-[75%] px-4 py-3 rounded-2xl text-sm leading-relaxed ${
                        msg.role === "user"
                          ? "bg-white text-black font-medium"
                          : isDarkMode
                          ? "bg-[#28292c] border border-gray-700/50 text-gray-100"
                          : "bg-white border border-gray-200 text-gray-900"
                      }`}
                    >
                      <div className="whitespace-pre-wrap">{msg.content}</div>
                    </div>
                  </div>
                ))}
                {isStreaming && (
                  <div className="flex items-center gap-2 text-xs text-gray-400 italic">
                    <span className="animate-spin">✦</span> Generating response...
                  </div>
                )}
                <div ref={chatEndRef} />
              </div>

              {/* Chat Input Bar */}
              <form onSubmit={handleSendChat} className="mt-4 pt-2">
                <div
                  className={`flex items-center gap-2 px-3 py-2 rounded-xl border ${
                    isDarkMode ? "bg-[#28292c] border-[#36373b]" : "bg-white border-gray-300"
                  }`}
                >
                  <input
                    type="text"
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    placeholder="Message Relay AI... (Press Enter to send)"
                    disabled={isStreaming}
                    className="flex-1 bg-transparent border-none outline-none text-sm px-2 py-1 text-inherit placeholder-gray-500"
                  />
                  <button
                    type="submit"
                    disabled={isStreaming || !chatInput.trim()}
                    className="px-4 py-1.5 rounded-lg bg-white text-black font-semibold text-xs hover:bg-gray-200 disabled:opacity-40 transition"
                  >
                    Send
                  </button>
                </div>
                <div className="text-center mt-2">
                  <span className="text-[11px] text-gray-500">
                    Relay AI can make mistakes. Verify important information.
                  </span>
                </div>
              </form>
            </div>
          )}

          {/* 3. SMART ROUTING */}
          {activeTab === "routing" && (
            <div className="space-y-6 max-w-3xl">
              <div>
                <h3 className="text-lg font-bold">Intelligent Model Routing</h3>
                <p className="text-xs text-gray-400">
                  Analyze prompts to determine complexity, domain classification, and select the optimal cost-to-performance model.
                </p>
              </div>

              <div className="space-y-3">
                <textarea
                  value={routingPrompt}
                  onChange={(e) => setRoutingPrompt(e.target.value)}
                  placeholder="Enter a prompt to analyze and route (e.g., 'Write an async Python pipeline with retry logic')..."
                  rows={4}
                  className={`w-full p-3 rounded-xl border text-sm outline-none ${
                    isDarkMode ? "bg-[#28292c] border-[#36373b]" : "bg-white border-gray-300"
                  }`}
                />
                <button
                  onClick={handleRunRouting}
                  disabled={routingLoading || !routingPrompt.trim()}
                  className="px-5 py-2.5 rounded-lg bg-white text-black font-semibold text-xs hover:bg-gray-200 disabled:opacity-50 transition"
                >
                  {routingLoading ? "Evaluating Task..." : "✦ Analyze & Route"}
                </button>
              </div>

              {routingResult && (
                <div className="p-5 rounded-xl border border-gray-700/50 bg-[#28292c]/40 space-y-4">
                  <div className="flex items-center justify-between pb-3 border-b border-gray-700/40">
                    <div>
                      <div className="text-xs text-gray-400">RECOMMENDED MODEL</div>
                      <div className="text-base font-bold text-white">
                        {routingResult.recommended_model} ({routingResult.recommended_provider.toUpperCase()})
                      </div>
                    </div>
                    <span className="px-2.5 py-1 rounded-full text-xs font-semibold bg-gray-700/60 text-white">
                      Confidence: {(routingResult.confidence * 100).toFixed(0)}%
                    </span>
                  </div>

                  <div className="grid grid-cols-2 gap-4 text-xs">
                    <div>
                      <span className="text-gray-400">Task Type:</span>{" "}
                      <span className="font-semibold uppercase">{routingResult.task_type}</span>
                    </div>
                    <div>
                      <span className="text-gray-400">Complexity:</span>{" "}
                      <span className="font-semibold uppercase">{routingResult.complexity}</span>
                    </div>
                  </div>

                  <div className="p-3 rounded-lg bg-black/30 text-xs text-gray-300 leading-relaxed">
                    <span className="font-semibold text-white">Decision Reasoning:</span>{" "}
                    {routingResult.reasoning}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* 4. COMPARE MODELS */}
          {activeTab === "compare" && (
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-bold">Side-by-Side Model Comparison</h3>
                <p className="text-xs text-gray-400">
                  Execute identical prompts across multiple models simultaneously to benchmark speed, tokens, and output quality.
                </p>
              </div>

              <div className="flex gap-2">
                <input
                  type="text"
                  value={comparePrompt}
                  onChange={(e) => setComparePrompt(e.target.value)}
                  placeholder="Enter prompt to compare across models..."
                  className={`flex-1 p-3 rounded-xl border text-sm outline-none ${
                    isDarkMode ? "bg-[#28292c] border-[#36373b]" : "bg-white border-gray-300"
                  }`}
                />
                <button
                  onClick={handleRunCompare}
                  disabled={compareLoading || !comparePrompt.trim()}
                  className="px-6 py-3 rounded-xl bg-white text-black font-semibold text-xs hover:bg-gray-200 disabled:opacity-50 transition shrink-0"
                >
                  {compareLoading ? "Comparing..." : "Compare Models"}
                </button>
              </div>

              {compareSummary && (
                <div className="px-4 py-2.5 rounded-lg border border-gray-600/40 text-xs font-mono text-gray-300">
                  ✦ {compareSummary}
                </div>
              )}

              {/* Comparison Output Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {compareSlots.map((slot, idx) => {
                  const res = compareResults.find(
                    (r) => r.provider === slot.provider && r.model === slot.model
                  );
                  return (
                    <div
                      key={idx}
                      className="p-4 rounded-xl border border-gray-700/50 bg-[#28292c]/40 flex flex-col h-96"
                    >
                      <div className="flex items-center justify-between pb-2 mb-3 border-b border-gray-700/40">
                        <div className="font-bold text-sm truncate">
                          {slot.provider.toUpperCase()} · {slot.model}
                        </div>
                        {res && (
                          <div className="text-[11px] font-mono text-gray-400">
                            {res.duration_sec}s · {res.speed_tps || 0} tok/s
                          </div>
                        )}
                      </div>
                      <div className="flex-1 overflow-y-auto text-xs leading-relaxed text-gray-200 whitespace-pre-wrap">
                        {compareLoading && !res ? (
                          <div className="text-gray-500 italic">Generating in parallel...</div>
                        ) : res?.error ? (
                          <div className="text-red-400">{res.error}</div>
                        ) : (
                          res?.content || <span className="text-gray-500 italic">Ready for benchmark</span>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* 5. AI TEAMS */}
          {activeTab === "teams" && (
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-bold">Autonomous Multi-Agent Collaboration</h3>
                <p className="text-xs text-gray-400">
                  Sequential 5-stage agent workflow: Planner → Researcher → Creator → Critic → Finalizer.
                </p>
              </div>

              <div className="flex gap-2">
                <input
                  type="text"
                  value={teamTask}
                  onChange={(e) => setTeamTask(e.target.value)}
                  placeholder="Describe goal (e.g., 'Design an event-driven architecture for a fintech wallet')..."
                  className={`flex-1 p-3 rounded-xl border text-sm outline-none ${
                    isDarkMode ? "bg-[#28292c] border-[#36373b]" : "bg-white border-gray-300"
                  }`}
                />
                <button
                  onClick={handleRunTeam}
                  disabled={teamRunning || !teamTask.trim()}
                  className="px-6 py-3 rounded-xl bg-white text-black font-semibold text-xs hover:bg-gray-200 disabled:opacity-50 transition shrink-0"
                >
                  {teamRunning ? "Executing Pipeline..." : "Deploy Team"}
                </button>
              </div>

              {/* Steps Timeline */}
              {teamSteps.length > 0 && (
                <div className="space-y-3">
                  <h4 className="text-sm font-semibold text-gray-300">Execution Timeline</h4>
                  <div className="space-y-2">
                    {teamSteps.map((s, idx) => (
                      <div
                        key={idx}
                        className="p-4 rounded-xl border border-gray-700/50 bg-[#28292c]/40"
                      >
                        <div className="flex items-center justify-between mb-2">
                          <span className="font-bold text-sm flex items-center gap-2">
                            <span>{s.icon}</span> {s.role_name}
                          </span>
                          <span className="text-xs font-mono text-gray-400">
                            {s.duration_sec}s · {s.status.toUpperCase()}
                          </span>
                        </div>
                        <div className="text-xs text-gray-300 whitespace-pre-wrap leading-relaxed max-h-48 overflow-y-auto">
                          {s.output}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Final Deliverable */}
              {teamDeliverable && (
                <div className="p-6 rounded-xl border border-gray-700 bg-[#28292c] space-y-3">
                  <h4 className="text-sm font-bold flex items-center gap-2 text-white">
                    <span>✨</span> Final Deliverable
                  </h4>
                  <div className="text-xs text-gray-200 whitespace-pre-wrap leading-relaxed">
                    {teamDeliverable}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* 6. SETTINGS */}
          {activeTab === "settings" && (
            <div className="space-y-6 max-w-2xl">
              <div>
                <h3 className="text-lg font-bold">System Diagnostics & Configuration</h3>
                <p className="text-xs text-gray-400">
                  Inspect cloud inference providers, API credentials, and database synchronizers.
                </p>
              </div>

              <div className="p-5 rounded-xl border border-gray-700/50 bg-[#28292c]/40 space-y-4">
                <h4 className="text-sm font-bold">Cloud Inference Providers</h4>
                <div className="space-y-2 text-xs">
                  {healthData?.providers ? (
                    Object.entries(healthData.providers).map(([k, v]) => (
                      <div key={k} className="flex items-center justify-between py-1.5 border-b border-gray-700/30">
                        <span className="font-semibold uppercase">{k}</span>
                        <span className={`px-2 py-0.5 rounded text-[11px] font-mono ${v.configured ? "bg-green-900/40 text-green-400" : "bg-gray-800 text-gray-500"}`}>
                          {v.configured ? "✓ Configured & Active" : "Missing API Key"}
                        </span>
                      </div>
                    ))
                  ) : (
                    <div className="text-gray-500">Checking provider status...</div>
                  )}
                </div>
              </div>

              <div className="p-5 rounded-xl border border-gray-700/50 bg-[#28292c]/40 space-y-3">
                <h4 className="text-sm font-bold">Authentication & Database</h4>
                <div className="text-xs text-gray-400 space-y-1">
                  <div>Supabase Status: {healthData?.supabase_configured ? "Connected" : "Local Vault Active"}</div>
                  <div>Sync Mode: Outbox Lease Queue</div>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>

      {/* Auth Modal */}
      {showAuthModal && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center p-4 z-50">
          <div className="w-full max-w-md p-6 rounded-2xl bg-[#202124] border border-gray-700 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-base font-bold">Sign in to Relay AI</h3>
              <button onClick={() => setShowAuthModal(false)} className="text-gray-400 hover:text-white">✕</button>
            </div>

            {authNotice && (
              <div className="p-2.5 rounded bg-yellow-900/40 text-yellow-300 text-xs">
                {authNotice}
              </div>
            )}

            <form onSubmit={handleAuthSubmit} className="space-y-3">
              <div>
                <label className="text-xs text-gray-400 mb-1 block">Email</label>
                <input
                  type="email"
                  value={authEmail}
                  onChange={(e) => setAuthEmail(e.target.value)}
                  required
                  placeholder="user@example.com"
                  className="w-full px-3 py-2 rounded-lg bg-[#28292c] border border-gray-700 text-xs text-white outline-none"
                />
              </div>

              <div>
                <label className="text-xs text-gray-400 mb-1 block">Password</label>
                <input
                  type="password"
                  value={authPassword}
                  onChange={(e) => setAuthPassword(e.target.value)}
                  required
                  placeholder="••••••••"
                  className="w-full px-3 py-2 rounded-lg bg-[#28292c] border border-gray-700 text-xs text-white outline-none"
                />
              </div>

              <button
                type="submit"
                disabled={authLoading}
                className="w-full py-2.5 rounded-lg bg-white text-black font-semibold text-xs hover:bg-gray-200 transition"
              >
                {authLoading ? "Authenticating..." : "Continue"}
              </button>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
