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
  authSignup,
  authSignOut,
  ModelOption,
  ChatMessage,
  ChatSession,
  RouteResponse,
  CompareResult,
  TeamStep,
  fetchSavedConversations,
  saveConversationApi,
  deleteConversationApi,
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

  // Authentication State
  const [user, setUser] = useState<{ id: string; email: string } | null>(null);
  const [isAuthLoaded, setIsAuthLoaded] = useState<boolean>(false);
  const [authMode, setAuthMode] = useState<"signin" | "signup">("signin");
  const [authEmail, setAuthEmail] = useState<string>("");
  const [authPassword, setAuthPassword] = useState<string>("");
  const [authConfirmPassword, setAuthConfirmPassword] = useState<string>("");
  const [showPassword, setShowPassword] = useState<boolean>(false);
  const [rememberMe, setRememberMe] = useState<boolean>(true);
  const [authLoading, setAuthLoading] = useState<boolean>(false);
  const [authError, setAuthError] = useState<string>("");
  const [authSuccess, setAuthSuccess] = useState<string>("");

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

  // Chat Persistence & Sessions State
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string>("session_init");
  const [showHistoryDrawer, setShowHistoryDrawer] = useState<boolean>(true);

  // Routing State
  const [routingPrompt, setRoutingPrompt] = useState<string>("");
  const [routingResult, setRoutingResult] = useState<RouteResponse | null>(null);
  const [routingLoading, setRoutingLoading] = useState<boolean>(false);

  // Compare State
  const [comparePrompt, setComparePrompt] = useState<string>("");
  const [compareSlots] = useState<{ provider: string; model: string }[]>([
    { provider: "groq", model: "openai/gpt-oss-120b" },
    { provider: "gemini", model: "gemini-3.5-flash-lite" },
  ]);
  const [compareResults, setCompareResults] = useState<CompareResult[]>([]);
  const [compareLoading, setCompareLoading] = useState<boolean>(false);
  const [compareSummary, setCompareSummary] = useState<string>("");

  // AI Teams State
  const [teamTask, setTeamTask] = useState<string>("");
  const [teamSteps, setTeamSteps] = useState<TeamStep[]>([]);
  const [teamDeliverable, setTeamDeliverable] = useState<string>("");
  const [teamRunning, setTeamRunning] = useState<boolean>(false);

  // Attachment & Voice Dictation State
  interface AttachmentItem {
    name: string;
    type: "image" | "file";
    sizeKb: number;
    dataUrl?: string;
    content?: string;
  }
  const [attachments, setAttachments] = useState<AttachmentItem[]>([]);
  const [showAttachMenu, setShowAttachMenu] = useState<boolean>(false);
  const [isListening, setIsListening] = useState<boolean>(false);
  const recognitionRef = useRef<any>(null);
  const cameraInputRef = useRef<HTMLInputElement>(null);
  const photoInputRef = useRef<HTMLInputElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const attachMenuRef = useRef<HTMLDivElement>(null);

  // Handle Speech-to-Text Voice Dictation
  const toggleVoiceDictation = () => {
    if (isListening) {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
      setIsListening(false);
      return;
    }

    const SpeechRec =
      (window as any).SpeechRecognition ||
      (window as any).webkitSpeechRecognition;

    if (!SpeechRec) {
      alert("Speech-to-text dictation is not supported by your current browser. Please open in Google Chrome or Microsoft Edge.");
      return;
    }

    try {
      const recognition = new SpeechRec();
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = "en-US";

      recognition.onstart = () => {
        setIsListening(true);
      };

      recognition.onresult = (event: any) => {
        let transcript = "";
        for (let i = event.resultIndex; i < event.results.length; i++) {
          if (event.results[i].isFinal) {
            transcript += event.results[i][0].transcript;
          }
        }
        if (transcript) {
          setChatInput((prev) => (prev ? `${prev.trim()} ${transcript.trim()}` : transcript.trim()));
        }
      };

      recognition.onerror = () => {
        setIsListening(false);
      };

      recognition.onend = () => {
        setIsListening(false);
      };

      recognitionRef.current = recognition;
      recognition.start();
    } catch {
      setIsListening(false);
    }
  };

  // Close attachment menu on outside click
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (attachMenuRef.current && !attachMenuRef.current.contains(e.target as Node)) {
        setShowAttachMenu(false);
      }
    };
    if (showAttachMenu) {
      document.addEventListener("mousedown", handleClickOutside);
    }
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [showAttachMenu]);

  // Handle File Selection
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>, isImage: boolean) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    Array.from(files).forEach((file) => {
      const sizeKb = Math.round(file.size / 1024);
      if (isImage || file.type.startsWith("image/")) {
        const reader = new FileReader();
        reader.onload = () => {
          setAttachments((prev) => [
            ...prev,
            {
              name: file.name,
              type: "image",
              sizeKb,
              dataUrl: reader.result as string,
            },
          ]);
        };
        reader.readAsDataURL(file);
      } else {
        const reader = new FileReader();
        reader.onload = () => {
          setAttachments((prev) => [
            ...prev,
            {
              name: file.name,
              type: "file",
              sizeKb,
              content: reader.result as string,
            },
          ]);
        };
        reader.readAsText(file);
      }
    });

    e.target.value = "";
    setShowAttachMenu(false);
  };

  const removeAttachment = (index: number) => {
    setAttachments((prev) => prev.filter((_, i) => i !== index));
  };

  // Initialize
  useEffect(() => {
    try {
      const saved = localStorage.getItem("relay_user");
      if (saved) {
        setUser(JSON.parse(saved));
      }
    } catch {}
    setIsAuthLoaded(true);

    fetchHealth().then(setHealthData).catch(() => {});
    fetchModels().then((m) => {
      setModels(m);
      if (m.length > 0) {
        setSelectedProvider(m[0].provider);
        setSelectedModel(m[0].id);
      }
    });
  }, []);

  // Load and Restore Saved Chat Sessions
  useEffect(() => {
    if (!isAuthLoaded) return;
    const storageKey = `relay_sessions_${user?.id || "guest"}`;
    try {
      const stored = localStorage.getItem(storageKey);
      if (stored) {
        const parsed: ChatSession[] = JSON.parse(stored);
        if (parsed.length > 0) {
          setSessions(parsed);
          const active = parsed[0];
          setCurrentSessionId(active.id);
          setChatMessages(active.messages);
          if (active.provider) setSelectedProvider(active.provider);
          if (active.model) setSelectedModel(active.model);
          return;
        }
      }
    } catch {}

    // Initial default session if none exists
    const defaultSession: ChatSession = {
      id: "session_" + Date.now(),
      title: "Welcome to Relay AI",
      createdAt: Date.now(),
      updatedAt: Date.now(),
      provider: selectedProvider || "groq",
      model: selectedModel || "openai/gpt-oss-120b",
      messages: [
        {
          role: "assistant",
          content: "Hello! I am Relay AI, your universal multi-model workspace. How can I assist you today?",
        },
      ],
    };
    setSessions([defaultSession]);
    setCurrentSessionId(defaultSession.id);
    setChatMessages(defaultSession.messages);
  }, [user?.id, isAuthLoaded]);

  const persistSessions = (updated: ChatSession[]) => {
    setSessions(updated);
    const storageKey = `relay_sessions_${user?.id || "guest"}`;
    try {
      localStorage.setItem(storageKey, JSON.stringify(updated));
    } catch {}
  };

  const handleNewChat = () => {
    const newSession: ChatSession = {
      id: "session_" + Date.now() + "_" + Math.random().toString(36).substring(2, 7),
      title: "New Conversation",
      createdAt: Date.now(),
      updatedAt: Date.now(),
      provider: selectedProvider,
      model: selectedModel,
      messages: [
        {
          role: "assistant",
          content: "How can I assist you in this conversation?",
        },
      ],
    };
    const updated = [newSession, ...sessions];
    persistSessions(updated);
    setCurrentSessionId(newSession.id);
    setChatMessages(newSession.messages);
    saveConversationApi(newSession).catch(() => {});
  };

  const handleSelectSession = (session: ChatSession) => {
    setCurrentSessionId(session.id);
    setChatMessages(session.messages);
    if (session.provider) setSelectedProvider(session.provider);
    if (session.model) setSelectedModel(session.model);
  };

  const handleDeleteSession = (sessionId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    const updated = sessions.filter((s) => s.id !== sessionId);
    deleteConversationApi(sessionId).catch(() => {});
    if (updated.length === 0) {
      const fallback: ChatSession = {
        id: "session_" + Date.now(),
        title: "New Conversation",
        createdAt: Date.now(),
        updatedAt: Date.now(),
        provider: selectedProvider,
        model: selectedModel,
        messages: [
          {
            role: "assistant",
            content: "How can I assist you today?",
          },
        ],
      };
      persistSessions([fallback]);
      setCurrentSessionId(fallback.id);
      setChatMessages(fallback.messages);
    } else {
      persistSessions(updated);
      if (currentSessionId === sessionId) {
        setCurrentSessionId(updated[0].id);
        setChatMessages(updated[0].messages);
      }
    }
  };

  // Handle Send Chat
  const handleSendChat = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if ((!chatInput.trim() && attachments.length === 0) || isStreaming) return;

    let promptText = chatInput.trim();
    if (!promptText && attachments.length > 0) {
      promptText = "Please analyze the attached document(s) and information.";
    }

    const currentAttachments = [...attachments];
    setAttachments([]);
    setChatInput("");
    setIsStreaming(true);

    const docSnippets: string[] = [];
    const imageList: string[] = [];
    const attachedBadges: string[] = [];

    for (const att of currentAttachments) {
      if (att.type === "image" && att.dataUrl) {
        imageList.push(att.dataUrl);
        attachedBadges.push(`🖼 ${att.name}`);
      } else {
        attachedBadges.push(`📄 ${att.name}`);
        if (att.content) {
          const preview = att.content.length > 6000 ? att.content.slice(0, 6000) + "\n...[truncated]" : att.content;
          docSnippets.push(`\n--- Attached Document (${att.name}) ---\n${preview}\n---`);
        }
      }
    }

    let fullPrompt = promptText;
    if (docSnippets.length > 0) {
      fullPrompt = `${promptText}\n${docSnippets.join("\n")}`.trim();
    }

    let displayMsg = promptText;
    if (attachedBadges.length > 0) {
      displayMsg = `${promptText}\n\n*📎 Attached: ${attachedBadges.join(", ")}*`.trim();
    }

    const userMsg: ChatMessage = {
      role: "user",
      content: displayMsg,
      images: imageList.length > 0 ? imageList : undefined,
    };
    const apiMsg: ChatMessage = {
      role: "user",
      content: fullPrompt,
      images: imageList.length > 0 ? imageList : undefined,
    };

    const newDisplayHistory = [...chatMessages, userMsg];
    const newApiHistory = [...chatMessages, apiMsg];
    setChatMessages(newDisplayHistory);

    const assistantMsg: ChatMessage = { role: "assistant", content: "" };
    setChatMessages([...newDisplayHistory, assistantMsg]);

    let accumulatedResponse = "";
    await streamChat(
      newApiHistory,
      selectedProvider,
      selectedModel,
      (delta) => {
        accumulatedResponse += delta;
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
      () => {
        setIsStreaming(false);
        const finalHistory: ChatMessage[] = [
          ...newDisplayHistory,
          { role: "assistant", content: accumulatedResponse },
        ];
        setSessions((prev) => {
          const next = prev.map((s) => {
            if (s.id === currentSessionId) {
              const autoTitle =
                s.title === "New Conversation" || s.title === "Welcome to Relay AI"
                  ? promptText.slice(0, 36) + (promptText.length > 36 ? "…" : "")
                  : s.title;
              const updatedSession: ChatSession = {
                ...s,
                title: autoTitle,
                messages: finalHistory,
                updatedAt: Date.now(),
                provider: selectedProvider,
                model: selectedModel,
              };
              saveConversationApi(updatedSession).catch(() => {});
              return updatedSession;
            }
            return s;
          });
          const storageKey = `relay_sessions_${user?.id || "guest"}`;
          try {
            localStorage.setItem(storageKey, JSON.stringify(next));
          } catch {}
          return next;
        });
      },
      (err) => {
        setIsStreaming(false);
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

  // Handle Authentication (Login / Signup)
  const handleAuthSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setAuthError("");
    setAuthSuccess("");

    const emailTrimmed = authEmail.trim();
    if (!emailTrimmed || !emailTrimmed.includes("@") || !emailTrimmed.includes(".")) {
      setAuthError("Please enter a valid email address.");
      return;
    }

    if (authPassword.length < 6) {
      setAuthError("Password must be at least 6 characters.");
      return;
    }

    if (authMode === "signup" && authPassword !== authConfirmPassword) {
      setAuthError("Passwords do not match. Please verify.");
      return;
    }

    setAuthLoading(true);
    try {
      if (authMode === "signin") {
        const res = await authLogin(emailTrimmed, authPassword);
        setUser(res.user);
        if (rememberMe) {
          localStorage.setItem("relay_user", JSON.stringify(res.user));
        }
      } else {
        const res = await authSignup(emailTrimmed, authPassword);
        setAuthSuccess("Account created successfully! Launching Relay AI workspace...");
        setTimeout(() => {
          setUser(res.user);
          if (rememberMe) {
            localStorage.setItem("relay_user", JSON.stringify(res.user));
          }
        }, 600);
      }
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Authentication failed";
      setAuthError(msg);
    } finally {
      setAuthLoading(false);
    }
  };

  // Instant Guest Access (Demo Mode)
  const handleGuestLogin = () => {
    const guestUser = {
      id: "usr_guest_" + Math.random().toString(36).substring(2, 9),
      email: "demo.guest@relay.ai",
    };
    setUser(guestUser);
    if (rememberMe) {
      localStorage.setItem("relay_user", JSON.stringify(guestUser));
    }
  };

  // Sign Out
  const handleSignOut = async () => {
    await authSignOut();
    setUser(null);
    setAuthEmail("");
    setAuthPassword("");
    setAuthConfirmPassword("");
    setAuthError("");
    setAuthSuccess("");
  };

  // Splash Screen while restoring saved authentication session
  if (!isAuthLoaded) {
    return (
      <div className={`min-h-screen flex flex-col items-center justify-center ${isDarkMode ? "bg-[#202124] text-white" : "bg-[#F8F9FA] text-[#202124]"}`}>
        <div className="w-14 h-14 rounded-2xl bg-white text-black text-2xl font-bold flex items-center justify-center animate-pulse mb-4 shadow-xl">
          ✦
        </div>
        <h1 className="text-base font-bold tracking-wider">RELAY AI</h1>
        <p className="text-xs text-gray-400 mt-1">Initializing secure multi-model workspace...</p>
      </div>
    );
  }

  // Full-Screen Dedicated Login & Sign-Up Page (Shown on initial visit before authentication)
  if (!user) {
    return (
      <div className={`min-h-screen flex flex-col justify-between ${isDarkMode ? "bg-[#202124] text-white" : "bg-[#F8F9FA] text-[#202124]"}`}>
        {/* Top Navbar */}
        <header className={`h-16 px-6 md:px-12 flex items-center justify-between border-b ${isDarkMode ? "border-[#36373b] bg-[#202124]/80" : "border-gray-200 bg-white/80"} backdrop-blur-md sticky top-0 z-40`}>
          <div className="flex items-center gap-3">
            <span className="w-8 h-8 rounded-lg bg-white text-black font-bold text-sm flex items-center justify-center shadow-md">
              ✦
            </span>
            <div>
              <h1 className="text-sm font-bold tracking-tight">RELAY AI</h1>
              <p className="text-[10px] text-gray-400">Think Together. Work Smarter.</p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={() => setIsDarkMode(!isDarkMode)}
              className={`p-2 rounded-lg text-xs border transition ${isDarkMode ? "border-gray-700 hover:bg-gray-800 text-gray-300" : "border-gray-300 hover:bg-gray-100 text-gray-700"}`}
              title="Toggle theme"
            >
              {isDarkMode ? "☼ Light Mode" : "☽ Dark Mode"}
            </button>
          </div>
        </header>

        {/* Main Split Grid */}
        <main className="flex-1 max-w-7xl w-full mx-auto p-4 md:p-8 lg:p-12 flex items-center justify-center">
          <div className="w-full grid grid-cols-1 lg:grid-cols-12 gap-8 lg:gap-12 items-center">
            {/* Left: Dedicated Auth Form Card */}
            <div className="lg:col-span-6 xl:col-span-5 w-full max-w-md mx-auto">
              <div className={`p-6 sm:p-8 rounded-3xl border shadow-2xl transition-all ${isDarkMode ? "bg-[#28292c] border-[#36373b]" : "bg-white border-gray-200"}`}>
                {/* Form Header */}
                <div className="text-center mb-6">
                  <div className="w-10 h-10 rounded-2xl bg-white text-black font-bold text-lg flex items-center justify-center mx-auto mb-3 shadow">
                    ✦
                  </div>
                  <h2 className="text-xl font-bold tracking-tight">
                    {authMode === "signin" ? "Sign in to Relay AI" : "Create your account"}
                  </h2>
                  <p className="text-xs text-gray-400 mt-1">
                    {authMode === "signin"
                      ? "Access multi-model orchestration, voice dictation, and AI teams."
                      : "Start collaborating with frontier AI models in a unified workspace."}
                  </p>
                </div>

                {/* Segmented Mode Switcher */}
                <div className={`flex p-1 rounded-xl mb-6 border ${isDarkMode ? "bg-[#202124] border-gray-700/60" : "bg-gray-100 border-gray-200"}`}>
                  <button
                    type="button"
                    onClick={() => { setAuthMode("signin"); setAuthError(""); setAuthSuccess(""); }}
                    className={`flex-1 py-2 text-xs font-semibold rounded-lg transition-all ${
                      authMode === "signin"
                        ? isDarkMode
                          ? "bg-[#36373b] text-white shadow"
                          : "bg-white text-black shadow"
                        : "text-gray-400 hover:text-inherit"
                    }`}
                  >
                    Sign In
                  </button>
                  <button
                    type="button"
                    onClick={() => { setAuthMode("signup"); setAuthError(""); setAuthSuccess(""); }}
                    className={`flex-1 py-2 text-xs font-semibold rounded-lg transition-all ${
                      authMode === "signup"
                        ? isDarkMode
                          ? "bg-[#36373b] text-white shadow"
                          : "bg-white text-black shadow"
                        : "text-gray-400 hover:text-inherit"
                    }`}
                  >
                    Create Account
                  </button>
                </div>

                {/* Error Banner */}
                {authError && (
                  <div className="mb-4 p-3 rounded-xl bg-red-900/30 border border-red-700/50 text-red-300 text-xs flex items-start gap-2">
                    <span className="font-bold">⚠</span>
                    <span>{authError}</span>
                  </div>
                )}

                {/* Success Banner */}
                {authSuccess && (
                  <div className="mb-4 p-3 rounded-xl bg-green-900/30 border border-green-700/50 text-green-300 text-xs flex items-start gap-2">
                    <span className="font-bold">✓</span>
                    <span>{authSuccess}</span>
                  </div>
                )}

                {/* Auth Form */}
                <form onSubmit={handleAuthSubmit} className="space-y-4">
                  <div>
                    <label className="block text-xs font-medium text-gray-400 mb-1.5">
                      Email Address
                    </label>
                    <div className={`flex items-center px-3.5 py-2.5 rounded-xl border transition-all ${isDarkMode ? "bg-[#202124] border-gray-700 focus-within:border-white" : "bg-gray-50 border-gray-300 focus-within:border-black"}`}>
                      <span className="text-gray-400 mr-2 text-sm">✉</span>
                      <input
                        type="email"
                        value={authEmail}
                        onChange={(e) => setAuthEmail(e.target.value)}
                        placeholder="you@company.com"
                        required
                        className="w-full bg-transparent border-none outline-none text-xs text-inherit placeholder-gray-500"
                      />
                    </div>
                  </div>

                  <div>
                    <label className="block text-xs font-medium text-gray-400 mb-1.5">
                      Password
                    </label>
                    <div className={`flex items-center px-3.5 py-2.5 rounded-xl border transition-all ${isDarkMode ? "bg-[#202124] border-gray-700 focus-within:border-white" : "bg-gray-50 border-gray-300 focus-within:border-black"}`}>
                      <span className="text-gray-400 mr-2 text-sm">🔒</span>
                      <input
                        type={showPassword ? "text" : "password"}
                        value={authPassword}
                        onChange={(e) => setAuthPassword(e.target.value)}
                        placeholder="••••••••"
                        required
                        minLength={6}
                        className="w-full bg-transparent border-none outline-none text-xs text-inherit placeholder-gray-500"
                      />
                      <button
                        type="button"
                        onClick={() => setShowPassword(!showPassword)}
                        className="text-gray-400 hover:text-inherit text-xs transition"
                        title={showPassword ? "Hide password" : "Show password"}
                      >
                        {showPassword ? "Hide" : "Show"}
                      </button>
                    </div>
                  </div>

                  {authMode === "signup" && (
                    <div>
                      <label className="block text-xs font-medium text-gray-400 mb-1.5">
                        Confirm Password
                      </label>
                      <div className={`flex items-center px-3.5 py-2.5 rounded-xl border transition-all ${isDarkMode ? "bg-[#202124] border-gray-700 focus-within:border-white" : "bg-gray-50 border-gray-300 focus-within:border-black"}`}>
                        <span className="text-gray-400 mr-2 text-sm">🔒</span>
                        <input
                          type={showPassword ? "text" : "password"}
                          value={authConfirmPassword}
                          onChange={(e) => setAuthConfirmPassword(e.target.value)}
                          placeholder="••••••••"
                          required
                          minLength={6}
                          className="w-full bg-transparent border-none outline-none text-xs text-inherit placeholder-gray-500"
                        />
                      </div>
                    </div>
                  )}

                  <div className="flex items-center justify-between text-xs">
                    <label className="flex items-center gap-2 cursor-pointer text-gray-400 hover:text-inherit">
                      <input
                        type="checkbox"
                        checked={rememberMe}
                        onChange={(e) => setRememberMe(e.target.checked)}
                        className="rounded border-gray-600 accent-white"
                      />
                      <span>Remember this device</span>
                    </label>
                    <span className="text-gray-500 hover:text-gray-400 cursor-pointer" onClick={() => alert("Please use your registered credentials or sign in using guest access.")}>
                      Need help?
                    </span>
                  </div>

                  {/* Primary Submit Button */}
                  <button
                    type="submit"
                    disabled={authLoading}
                    className="w-full py-3 rounded-xl bg-white text-black font-bold text-xs hover:bg-gray-200 transition-all shadow-md disabled:opacity-50 flex items-center justify-center gap-2 mt-2"
                  >
                    {authLoading ? (
                      <>
                        <span className="animate-spin text-sm">✦</span>
                        <span>{authMode === "signin" ? "Signing In..." : "Creating Account..."}</span>
                      </>
                    ) : (
                      <span>{authMode === "signin" ? "Sign In to Relay AI →" : "Create Free Account →"}</span>
                    )}
                  </button>
                </form>

                {/* Divider */}
                <div className="relative my-6 text-center">
                  <div className="absolute inset-0 flex items-center">
                    <div className={`w-full border-t ${isDarkMode ? "border-gray-700/50" : "border-gray-200"}`}></div>
                  </div>
                  <span className={`relative px-3 text-[11px] ${isDarkMode ? "bg-[#28292c] text-gray-500" : "bg-white text-gray-400"}`}>
                    or quick explore
                  </span>
                </div>

                {/* Guest / Demo Bypass Option */}
                <button
                  type="button"
                  onClick={handleGuestLogin}
                  className={`w-full py-2.5 px-4 rounded-xl border text-xs font-semibold transition flex items-center justify-between ${
                    isDarkMode
                      ? "border-gray-700 hover:bg-gray-700/30 text-gray-300 hover:text-white"
                      : "border-gray-300 hover:bg-gray-100 text-gray-700 hover:text-black"
                  }`}
                >
                  <span className="flex items-center gap-2">
                    <span>⚡</span> Continue as Guest (Demo Access)
                  </span>
                  <span>→</span>
                </button>
              </div>
            </div>

            {/* Right: Feature Highlights & Showcase */}
            <div className="hidden lg:block lg:col-span-6 xl:col-span-7 space-y-6 pl-4">
              <div className="space-y-3">
                <span className="px-3 py-1 rounded-full text-[11px] font-mono border border-gray-600/50 text-gray-400 inline-block">
                  Unified AI Operating System
                </span>
                <h2 className="text-3xl xl:text-4xl font-extrabold tracking-tight leading-tight">
                  One workspace for frontier models and autonomous teams.
                </h2>
                <p className="text-sm text-gray-400 max-w-lg leading-relaxed">
                  Switch effortlessly between Groq LPU speed, Google Gemini 3.5 reasoning, and OpenAI GPT-4o with integrated voice dictation, smart intent routing, and 5-agent pipeline execution.
                </p>
              </div>

              {/* Showcase Cards Grid */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-2">
                <div className={`p-4 rounded-2xl border ${isDarkMode ? "bg-[#28292c]/40 border-gray-700/40" : "bg-white border-gray-200 shadow-sm"}`}>
                  <div className="text-base mb-1">🧭</div>
                  <h4 className="text-xs font-bold mb-1">Autonomous AI Teams</h4>
                  <p className="text-[11px] text-gray-400 leading-normal">
                    5-agent pipeline: Planner, Researcher, Creator, Critic, and Finalizer collaborate synchronously.
                  </p>
                </div>

                <div className={`p-4 rounded-2xl border ${isDarkMode ? "bg-[#28292c]/40 border-gray-700/40" : "bg-white border-gray-200 shadow-sm"}`}>
                  <div className="text-base mb-1">✦</div>
                  <h4 className="text-xs font-bold mb-1">Intelligent Router</h4>
                  <p className="text-[11px] text-gray-400 leading-normal">
                    Classifies prompt complexity and domain in real time to dispatch queries to optimal models.
                  </p>
                </div>

                <div className={`p-4 rounded-2xl border ${isDarkMode ? "bg-[#28292c]/40 border-gray-700/40" : "bg-white border-gray-200 shadow-sm"}`}>
                  <div className="text-base mb-1">⚖️</div>
                  <h4 className="text-xs font-bold mb-1">Side-by-Side Compare</h4>
                  <p className="text-[11px] text-gray-400 leading-normal">
                    Concurrently evaluate response quality, token throughput, and latency across providers.
                  </p>
                </div>

                <div className={`p-4 rounded-2xl border ${isDarkMode ? "bg-[#28292c]/40 border-gray-700/40" : "bg-white border-gray-200 shadow-sm"}`}>
                  <div className="text-base mb-1">🎙</div>
                  <h4 className="text-xs font-bold mb-1">Speech & Multimodal</h4>
                  <p className="text-[11px] text-gray-400 leading-normal">
                    Real-time speech-to-text dictation, camera capture, and photo attachments.
                  </p>
                </div>
              </div>

              {/* System Badge */}
              <div className={`p-4 rounded-2xl border flex items-center justify-between ${isDarkMode ? "bg-[#28292c]/60 border-gray-700/50" : "bg-gray-100 border-gray-200"}`}>
                <div className="flex items-center gap-3">
                  <span className="w-2.5 h-2.5 rounded-full bg-green-500 animate-pulse"></span>
                  <span className="text-xs font-medium">Production Cloud Active · Zero Telemetry Vault</span>
                </div>
                <span className="text-[11px] text-gray-400 font-mono">v1.0.0</span>
              </div>
            </div>
          </div>
        </main>

        {/* Footer */}
        <footer className={`py-4 text-center text-[11px] border-t ${isDarkMode ? "border-gray-800 text-gray-500" : "border-gray-200 text-gray-400"}`}>
          Relay AI · Monochromatic Intelligence Platform · Encrypted Session Storage
        </footer>
      </div>
    );
  }

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
              <div className="flex items-center gap-2 truncate max-w-[130px]">
                <div className="w-6 h-6 rounded-full bg-white text-black font-bold flex items-center justify-center text-[10px] shrink-0 shadow">
                  {user.email.charAt(0).toUpperCase()}
                </div>
                <span className="truncate font-medium text-gray-300">{user.email}</span>
              </div>
              <button
                onClick={handleSignOut}
                className="text-xs text-red-400 hover:text-red-300 transition shrink-0 underline font-medium"
                title="Sign Out of Relay AI"
              >
                Sign Out
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
            <div className="flex h-[calc(100vh-8.5rem)] gap-4">
              {/* Collapsible Saved Chats Drawer */}
              {showHistoryDrawer && (
                <div
                  className={`w-64 shrink-0 rounded-2xl border flex flex-col p-3 transition-all ${
                    isDarkMode ? "bg-[#1b1c1e] border-gray-800" : "bg-white border-gray-200 shadow-sm"
                  }`}
                >
                  <div className="flex items-center justify-between pb-2 mb-2 border-b border-gray-700/20">
                    <span className="text-xs font-bold uppercase tracking-wider text-gray-400">Saved Chats</span>
                    <button
                      onClick={handleNewChat}
                      className="px-2.5 py-1 rounded-lg bg-white text-black font-semibold text-xs hover:bg-gray-200 transition flex items-center gap-1 shadow-sm"
                      title="Start a new chat session"
                    >
                      <span>+</span> New
                    </button>
                  </div>

                  <div className="flex-1 overflow-y-auto space-y-1.5 pr-1">
                    {sessions.map((sess) => {
                      const isActive = sess.id === currentSessionId;
                      return (
                        <div
                          key={sess.id}
                          onClick={() => handleSelectSession(sess)}
                          className={`group w-full p-2.5 rounded-xl text-left cursor-pointer transition flex items-center justify-between text-xs ${
                            isActive
                              ? isDarkMode
                                ? "bg-[#36454F] text-white font-medium"
                                : "bg-[#202124] text-white font-medium"
                              : isDarkMode
                              ? "text-gray-300 hover:bg-[#28292c]"
                              : "text-gray-700 hover:bg-gray-100"
                          }`}
                        >
                          <div className="truncate flex-1 pr-2">
                            <div className="truncate font-medium">{sess.title || "New Conversation"}</div>
                            <div className="text-[10px] text-gray-400 opacity-80">
                              {new Date(sess.updatedAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })} · {sess.messages.length} msgs
                            </div>
                          </div>
                          <button
                            onClick={(e) => handleDeleteSession(sess.id, e)}
                            className="opacity-0 group-hover:opacity-100 p-1 hover:text-red-400 text-gray-400 transition"
                            title="Delete conversation"
                          >
                            ✕
                          </button>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Main Chat Interface */}
              <div className="flex-1 flex flex-col min-w-0">
                {/* Chat Top Controls Bar */}
                <div className="flex items-center justify-between pb-2 mb-2 border-b border-gray-700/20">
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => setShowHistoryDrawer(!showHistoryDrawer)}
                      className="px-2.5 py-1 rounded-lg border border-gray-700/40 text-xs font-medium hover:bg-gray-700/20 transition flex items-center gap-1.5"
                      title="Toggle Saved Chats sidebar"
                    >
                      <span>💬</span>
                      <span>{showHistoryDrawer ? "Hide History" : `Saved Chats (${sessions.length})`}</span>
                    </button>
                    <button
                      onClick={handleNewChat}
                      className="px-2.5 py-1 rounded-lg bg-gray-700/30 hover:bg-gray-700/50 text-xs font-medium transition"
                      title="Start a new chat"
                    >
                      + New Chat
                    </button>
                  </div>

                  <div className="flex items-center gap-2 text-xs">
                    <span className="text-gray-400 hidden sm:inline">Engine:</span>
                    <select
                      value={`${selectedProvider}:${selectedModel}`}
                      onChange={(e) => {
                        const [p, m] = e.target.value.split(":");
                        setSelectedProvider(p);
                        setSelectedModel(m);
                      }}
                      className={`px-2.5 py-1 rounded-lg text-xs outline-none border ${
                        isDarkMode ? "bg-[#28292c] border-gray-700 text-white" : "bg-white border-gray-300 text-gray-900"
                      }`}
                    >
                      {models.map((m) => (
                        <option key={`${m.provider}:${m.id}`} value={`${m.provider}:${m.id}`}>
                          {m.provider_name} · {m.display_name}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>

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

              {/* Hidden File Inputs for Camera, Photos, Files */}
              <input
                type="file"
                ref={cameraInputRef}
                accept="image/*"
                capture="environment"
                onChange={(e) => handleFileUpload(e, true)}
                className="hidden"
              />
              <input
                type="file"
                ref={photoInputRef}
                accept="image/*"
                multiple
                onChange={(e) => handleFileUpload(e, true)}
                className="hidden"
              />
              <input
                type="file"
                ref={fileInputRef}
                accept=".pdf,.docx,.txt,.md,.csv,.json,.log,text/*,application/pdf"
                multiple
                onChange={(e) => handleFileUpload(e, false)}
                className="hidden"
              />

              {/* Attachment Preview Tray */}
              {attachments.length > 0 && (
                <div className="flex flex-wrap gap-2 mb-2 px-1">
                  {attachments.map((att, idx) => (
                    <div
                      key={idx}
                      className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs border ${
                        isDarkMode
                          ? "bg-[#28292c] border-gray-700 text-gray-200"
                          : "bg-gray-100 border-gray-300 text-gray-800"
                      }`}
                    >
                      <span>{att.type === "image" ? "🖼" : "📎"}</span>
                      <span className="max-w-[140px] truncate">{att.name}</span>
                      <span className="text-[10px] text-gray-500">({att.sizeKb} KB)</span>
                      <button
                        type="button"
                        onClick={() => removeAttachment(idx)}
                        className="ml-1 text-gray-400 hover:text-red-400 font-bold"
                        title="Remove attachment"
                      >
                        ✕
                      </button>
                    </div>
                  ))}
                </div>
              )}

              {/* Chat Input Bar */}
              <form onSubmit={handleSendChat} className="mt-2 pt-1">
                <div
                  className={`relative flex items-center gap-2 px-3 py-2 rounded-2xl border transition-all ${
                    isDarkMode ? "bg-[#28292c] border-[#36373b]" : "bg-white border-gray-300"
                  }`}
                >
                  {/* '+' Attachment Trigger Button */}
                  <div className="relative" ref={attachMenuRef}>
                    <button
                      type="button"
                      onClick={() => setShowAttachMenu(!showAttachMenu)}
                      className={`w-8 h-8 rounded-full flex items-center justify-center text-lg font-light transition ${
                        showAttachMenu
                          ? "bg-white text-black"
                          : isDarkMode
                          ? "text-gray-300 hover:bg-[#36373b]"
                          : "text-gray-600 hover:bg-gray-200"
                      }`}
                      title="Attach photos or camera capture"
                    >
                      +
                    </button>

                    {/* Reference Attachment Popup Menu (Camera, Photos, Files) */}
                    {showAttachMenu && (
                      <div
                        className={`absolute bottom-12 left-0 w-48 rounded-2xl shadow-2xl border p-2 z-50 backdrop-blur-md ${
                          isDarkMode
                            ? "bg-[#202124]/95 border-gray-700/80 text-white"
                            : "bg-white/95 border-gray-200 text-gray-900"
                        }`}
                      >
                        <button
                          type="button"
                          onClick={() => {
                            setShowAttachMenu(false);
                            cameraInputRef.current?.click();
                          }}
                          className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl hover:bg-gray-700/30 text-left text-sm font-medium transition"
                        >
                          <span className="w-8 h-8 rounded-full bg-gray-700/40 flex items-center justify-center text-base">
                            📷
                          </span>
                          Camera
                        </button>
                        <button
                          type="button"
                          onClick={() => {
                            setShowAttachMenu(false);
                            photoInputRef.current?.click();
                          }}
                          className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl hover:bg-gray-700/30 text-left text-sm font-medium transition"
                        >
                          <span className="w-8 h-8 rounded-full bg-gray-700/40 flex items-center justify-center text-base">
                            🖼
                          </span>
                          Photos
                        </button>
                        <button
                          type="button"
                          onClick={() => {
                            setShowAttachMenu(false);
                            fileInputRef.current?.click();
                          }}
                          className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl hover:bg-gray-700/30 text-left text-sm font-medium transition"
                        >
                          <span className="w-8 h-8 rounded-full bg-gray-700/40 flex items-center justify-center text-base">
                            📎
                          </span>
                          Files
                        </button>
                      </div>
                    )}
                  </div>

                  {/* Main Text Input Field */}
                  <input
                    type="text"
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    placeholder={isListening ? "Listening... Speak now..." : "Ask Relay AI…"}
                    disabled={isStreaming}
                    className="flex-1 bg-transparent border-none outline-none text-sm px-2 py-1 text-inherit placeholder-gray-500"
                  />

                  {/* Dedicated Speech-to-Text Microphone Button */}
                  <button
                    type="button"
                    onClick={toggleVoiceDictation}
                    className={`p-2 rounded-full transition flex items-center justify-center ${
                      isListening
                        ? "bg-red-600 text-white animate-pulse"
                        : isDarkMode
                        ? "text-gray-300 hover:bg-[#36373b]"
                        : "text-gray-600 hover:bg-gray-200"
                    }`}
                    title={isListening ? "Listening... Click to stop" : "Voice Dictation (Speech to Text)"}
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      width="18"
                      height="18"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
                      <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                      <line x1="12" y1="19" x2="12" y2="23" />
                      <line x1="8" y1="23" x2="16" y2="23" />
                    </svg>
                  </button>

                  {/* Send Button */}
                  <button
                    type="submit"
                    disabled={isStreaming || (!chatInput.trim() && attachments.length === 0)}
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

              {/* Current Account Session Card */}
              <div className="p-5 rounded-xl border border-gray-700/50 bg-[#28292c]/40 space-y-3">
                <h4 className="text-sm font-bold">Current Account Session</h4>
                <div className="text-xs text-gray-400 space-y-1">
                  <div>Email: <span className="text-white font-medium">{user.email}</span></div>
                  <div>User ID: <span className="font-mono text-gray-300">{user.id}</span></div>
                  <div>Session Status: <span className="text-green-400 font-medium">✓ Active</span></div>
                </div>
                <button
                  onClick={handleSignOut}
                  className="mt-2 px-4 py-2 rounded-lg bg-red-600/20 hover:bg-red-600/30 text-red-400 text-xs font-semibold border border-red-500/30 transition"
                >
                  Sign Out of Relay AI
                </button>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
