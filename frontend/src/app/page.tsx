"use client";

import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { motion } from "motion/react";
import {
  BrainCircuit,
  ChevronDown,
  Crosshair,
  Database,
  MessageSquarePlus,
  PanelLeftClose,
  PanelLeftOpen,
  Search,
  SearchX,
  Sword,
  Target,
  Trash2,
  Trophy,
  Zap,
  Bot,
  User,
} from "lucide-react";

import {
  Conversation,
  ConversationContent,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation";
import {
  Message,
  MessageContent,
  MessageResponse,
} from "@/components/ai-elements/message";
import {
  PromptInput,
  PromptInputTextarea,
  PromptInputBody,
  PromptInputFooter,
  PromptInputSubmit,
} from "@/components/ai-elements/prompt-input";
import {
  Reasoning,
  ReasoningTrigger,
  ReasoningContent,
} from "@/components/ai-elements/reasoning";
import {
  Sources,
  SourcesTrigger,
  SourcesContent,
} from "@/components/ai-elements/sources";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Models & RAG config
// ---------------------------------------------------------------------------

type RagMode = "auto" | "always" | "off";

interface ModelOption {
  id: string;
  name: string;
  supportsReasoning: boolean;
  supportsTools: boolean;
}

const MODELS: ModelOption[] = [
  { id: "qwen2.5:7b-instruct-q4_K_M", name: "Qwen 2.5 7B", supportsReasoning: false, supportsTools: true },
  { id: "qwen3:8b", name: "Qwen3 8B", supportsReasoning: true, supportsTools: true },
  { id: "deepseek-r1:8b", name: "DeepSeek R1 8B", supportsReasoning: true, supportsTools: false },
];

// ---------------------------------------------------------------------------
// Chat sessions
// ---------------------------------------------------------------------------

interface ChatSession {
  id: string;
  title: string;
  createdAt: Date;
}

// ---------------------------------------------------------------------------
// Suggestion pills
// ---------------------------------------------------------------------------

const SUGGESTIONS = [
  {
    icon: Sword,
    label: "Dano de Armas",
    query: "Quais armas causam mais dano por round em média?",
  },
  {
    icon: Trophy,
    label: "Jogadores",
    query: "Quais são os melhores jogadores de CS:GO?",
  },
  {
    icon: Target,
    label: "Eventos de Round",
    query: "Quais são os eventos mais comuns de fim de round?",
  },
  {
    icon: Zap,
    label: "Economia",
    query: "Como a economia afeta o resultado dos rounds?",
  },
];

// ---------------------------------------------------------------------------
// RAG mode labels
// ---------------------------------------------------------------------------

const RAG_MODE_CONFIG: Record<RagMode, { label: string; tip: string }> = {
  auto: { label: "Auto", tip: "Modelo decide quando buscar" },
  always: { label: "RAG", tip: "Sempre buscar no banco" },
  off: { label: "Off", tip: "Sem busca, apenas conhecimento geral" },
};

const RAG_MODES: RagMode[] = ["auto", "always", "off"];

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function Home() {
  const [input, setInput] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [chatSessions, setChatSessions] = useState<ChatSession[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);
  const shouldSaveRef = useRef(false);

  // AI settings
  const [selectedModel, setSelectedModel] = useState(MODELS[0].id);
  const [ragMode, setRagMode] = useState<RagMode>("auto");

  const currentModel = MODELS.find((m) => m.id === selectedModel) ?? MODELS[0];

  // Force RAG off when model doesn't support tools
  const effectiveRagMode = currentModel.supportsTools ? ragMode : "off";

  // Refs so the transport body function always reads current values
  const selectedModelRef = useRef(selectedModel);
  const ragModeRef = useRef(effectiveRagMode);
  selectedModelRef.current = selectedModel;
  ragModeRef.current = effectiveRagMode;

  // Stable transport — body is a function that reads from refs
  const transport = useMemo(
    () =>
      new DefaultChatTransport({
        api: "/api/chat",
        body: () => ({ model: selectedModelRef.current, ragMode: ragModeRef.current }),
      }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    []
  );

  const { messages, status, sendMessage, stop, setMessages } = useChat({
    transport,
  });

  const isGenerating = status === "streaming" || status === "submitted";
  const hasMessages = messages.length > 0;

  // Load sessions from DB on mount
  useEffect(() => {
    fetch("/api/sessions")
      .then((r) => r.json())
      .then((sessions: { id: string; title: string; created_at: string }[]) => {
        if (sessions.length === 0) {
          const id = Date.now().toString();
          fetch("/api/sessions", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ id, title: "Nova Conversa" }),
          });
          setChatSessions([{ id, title: "Nova Conversa", createdAt: new Date() }]);
          setActiveChatId(id);
        } else {
          setChatSessions(
            sessions.map((s) => ({ id: s.id, title: s.title, createdAt: new Date(s.created_at) }))
          );
          setActiveChatId(sessions[0].id);
        }
        setLoaded(true);
      });
  }, []);

  // Load messages when active session changes
  useEffect(() => {
    if (!activeChatId) return;
    fetch(`/api/sessions/${activeChatId}/messages`)
      .then((r) => r.json())
      .then((msgs: { id: string; role: string; parts: unknown[] }[]) => {
        if (msgs.length > 0) {
          setMessages(msgs as Parameters<typeof setMessages>[0]);
        }
      });
  }, [activeChatId, setMessages]);

  // Persist messages after AI response completes
  useEffect(() => {
    if (shouldSaveRef.current && status === "ready" && messages.length > 0 && activeChatId) {
      shouldSaveRef.current = false;
      fetch(`/api/sessions/${activeChatId}/messages`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages }),
      });
    }
  }, [status, messages, activeChatId]);

  const updateSessionTitle = useCallback(
    (title: string) => {
      if (!activeChatId) return;
      setChatSessions((prev) =>
        prev.map((s) => (s.id === activeChatId ? { ...s, title } : s))
      );
      fetch(`/api/sessions/${activeChatId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title }),
      });
    },
    [activeChatId]
  );

  const handleSubmit = useCallback(() => {
    if (!input.trim() || isGenerating) return;
    if (messages.length === 0) {
      const title = input.length > 40 ? input.substring(0, 40) + "..." : input;
      updateSessionTitle(title);
    }
    shouldSaveRef.current = true;
    sendMessage({ text: input });
    setInput("");
  }, [input, isGenerating, messages.length, updateSessionTitle, sendMessage]);

  const handleSuggestionClick = useCallback(
    (query: string) => {
      if (isGenerating) return;
      if (messages.length === 0) {
        const title = query.length > 40 ? query.substring(0, 40) + "..." : query;
        updateSessionTitle(title);
      }
      shouldSaveRef.current = true;
      sendMessage({ text: query });
    },
    [isGenerating, messages.length, updateSessionTitle, sendMessage]
  );

  const handleNewChat = useCallback(() => {
    const newId = Date.now().toString();
    const title = "Nova Conversa";
    fetch("/api/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: newId, title }),
    });
    setChatSessions((prev) => [{ id: newId, title, createdAt: new Date() }, ...prev]);
    setActiveChatId(newId);
    setMessages([]);
    setInput("");
  }, [setMessages]);

  const handleDeleteChat = useCallback(
    (id: string) => {
      fetch(`/api/sessions/${id}`, { method: "DELETE" });
      setChatSessions((prev) => {
        const filtered = prev.filter((s) => s.id !== id);
        if (filtered.length === 0) {
          const newId = Date.now().toString();
          const newSession = { id: newId, title: "Nova Conversa", createdAt: new Date() };
          fetch("/api/sessions", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ id: newId, title: "Nova Conversa" }),
          });
          setActiveChatId(newId);
          setMessages([]);
          return [newSession];
        }
        if (id === activeChatId) {
          setActiveChatId(filtered[0].id);
        }
        return filtered;
      });
    },
    [activeChatId, setMessages]
  );

  // -------------------------------------------------------------------------
  // Shared input controls JSX
  // -------------------------------------------------------------------------

  const inputControls = (
    <PromptInputFooter className="justify-between px-3 pb-2.5">
      <div className="flex items-center gap-1.5">
        {/* Model selector */}
        <Select value={selectedModel} onValueChange={setSelectedModel}>
          <SelectTrigger
            size="sm"
            className="h-7 gap-1 rounded-lg border-0 bg-transparent px-2 text-xs text-muted-foreground hover:text-foreground hover:bg-accent"
          >
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {MODELS.map((m) => (
              <SelectItem key={m.id} value={m.id}>
                <span className="flex items-center gap-1.5">
                  {m.name}
                  {m.supportsReasoning && (
                    <BrainCircuit className="size-3 text-primary/60" />
                  )}
                </span>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        {/* RAG mode toggle */}
        <div className={cn(
          "flex items-center rounded-lg",
          !currentModel.supportsTools && "opacity-40"
        )}>
          {RAG_MODES.map((mode) => {
            const isActive = effectiveRagMode === mode;
            return (
              <Tooltip key={mode}>
                <TooltipTrigger asChild>
                  <button
                    type="button"
                    disabled={!currentModel.supportsTools}
                    onClick={() => setRagMode(mode)}
                    className={cn(
                      "flex items-center gap-1 px-2 py-1 text-xs font-medium transition-colors first:rounded-l-[7px] last:rounded-r-[7px]",
                      "disabled:cursor-not-allowed",
                      isActive
                        ? mode === "off"
                          ? "bg-destructive/15 text-destructive"
                          : "bg-primary/15 text-primary"
                        : "text-muted-foreground hover:text-foreground"
                    )}
                  >
                    {mode === "always" && <Search className="size-3" />}
                    {mode === "off" && <SearchX className="size-3" />}
                    {RAG_MODE_CONFIG[mode].label}
                  </button>
                </TooltipTrigger>
                <TooltipContent side="top" className="text-xs">
                  {!currentModel.supportsTools
                    ? "Modelo não suporta busca"
                    : RAG_MODE_CONFIG[mode].tip}
                </TooltipContent>
              </Tooltip>
            );
          })}
        </div>
      </div>

      <PromptInputSubmit status={status} onStop={stop} />
    </PromptInputFooter>
  );

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------

  if (!loaded) {
    return (
      <div className="flex h-dvh items-center justify-center bg-background">
        <Crosshair className="size-5 animate-pulse text-primary/50" />
      </div>
    );
  }

  return (
    <div className="flex h-dvh bg-background">
      {/* ================================================================= */}
      {/* Sidebar                                                           */}
      {/* ================================================================= */}
      <aside
        className={cn(
          "flex flex-col border-r border-border/50 bg-card/40 transition-all duration-300 ease-in-out",
          sidebarOpen ? "w-64" : "w-0 overflow-hidden border-r-0"
        )}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-4">
          <span className="text-sm font-medium tracking-tight text-foreground/60">
            RAG Intelligence
          </span>
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={() => setSidebarOpen(false)}
            className="text-muted-foreground/60 hover:text-foreground"
          >
            <PanelLeftClose className="size-4" />
          </Button>
        </div>

        {/* New Chat */}
        <div className="px-3 pb-2">
          <Button
            variant="ghost"
            className="w-full justify-start gap-2 rounded-lg text-muted-foreground hover:text-foreground"
            onClick={handleNewChat}
          >
            <MessageSquarePlus className="size-4" />
            Nova Conversa
          </Button>
        </div>

        <Separator className="opacity-30" />

        {/* History */}
        <div className="px-3 py-2">
          <p className="px-2 text-[11px] font-medium text-muted-foreground/50 uppercase tracking-widest">
            Histórico
          </p>
        </div>
        <ScrollArea className="flex-1 px-3">
          <div className="space-y-0.5 pb-4">
            {chatSessions.map((session) => (
              <div
                key={session.id}
                className={cn(
                  "group flex items-center justify-between rounded-lg px-3 py-2 text-sm cursor-pointer transition-colors",
                  session.id === activeChatId
                    ? "bg-accent/80 text-foreground"
                    : "text-muted-foreground hover:bg-accent/40 hover:text-foreground"
                )}
                onClick={() => {
                  if (session.id !== activeChatId) {
                    setMessages([]);
                    setActiveChatId(session.id);
                  }
                }}
              >
                <span className="truncate">{session.title}</span>
                <Button
                  variant="ghost"
                  size="icon-xs"
                  className="opacity-0 group-hover:opacity-100 transition-opacity shrink-0"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleDeleteChat(session.id);
                  }}
                >
                  <Trash2 className="size-3" />
                </Button>
              </div>
            ))}
          </div>
        </ScrollArea>
      </aside>

      {/* ================================================================= */}
      {/* Main Content                                                      */}
      {/* ================================================================= */}
      <main className="relative flex flex-1 flex-col min-w-0">
        {/* Floating sidebar toggle */}
        {!sidebarOpen && (
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={() => setSidebarOpen(true)}
            className="absolute top-3 left-3 z-10 text-muted-foreground/50 hover:text-foreground"
          >
            <PanelLeftOpen className="size-4" />
          </Button>
        )}

        {!hasMessages ? (
          /* ============================================================= */
          /* Empty State — centered greeting + input + pills               */
          /* ============================================================= */
          <div className="flex flex-1 flex-col items-center justify-center gap-6 px-4 pb-16">
            {/* Greeting */}
            <motion.div
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, ease: [0.25, 0.46, 0.45, 0.94] }}
              className="flex items-center gap-3"
            >
              <Crosshair className="size-7 text-primary/40" />
              <h1 className="font-serif text-5xl tracking-tight text-foreground/70">
                Analista CS:GO
              </h1>
            </motion.div>

            {/* Centered Input */}
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.1, ease: [0.25, 0.46, 0.45, 0.94] }}
              className="w-full max-w-[640px]"
            >
              <PromptInput
                onSubmit={handleSubmit}
                className={cn(
                  "rounded-2xl border-border/30 bg-card/50 backdrop-blur-xl",
                  "shadow-lg shadow-black/10",
                  "transition-all duration-200",
                  "focus-within:!border-primary/25 focus-within:!shadow-xl focus-within:!shadow-primary/5"
                )}
              >
                <PromptInputBody>
                  <PromptInputTextarea
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Pergunte sobre partidas, armas, jogadores..."
                    className="!min-h-[52px] !py-3.5 !px-4 !text-[15px] placeholder:text-muted-foreground/35"
                  />
                </PromptInputBody>
                {inputControls}
              </PromptInput>
            </motion.div>

            {/* Suggestion Pills */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.5, delay: 0.25 }}
              className="flex flex-wrap justify-center gap-2"
            >
              {SUGGESTIONS.map((s) => (
                <button
                  key={s.label}
                  onClick={() => handleSuggestionClick(s.query)}
                  className={cn(
                    "flex items-center gap-2 rounded-full border border-border/40 bg-card/30 px-4 py-2",
                    "text-sm text-muted-foreground/70",
                    "transition-all duration-200",
                    "hover:bg-accent/60 hover:text-foreground hover:border-border/60"
                  )}
                >
                  <s.icon className="size-3.5" />
                  {s.label}
                </button>
              ))}
            </motion.div>
          </div>
        ) : (
          /* ============================================================= */
          /* Chat State — messages + bottom input                          */
          /* ============================================================= */
          <>
            <Conversation className="flex-1 overflow-hidden">
              <ConversationContent className="space-y-6 px-4 py-6 max-w-3xl mx-auto w-full">
                {messages.map((message) => (
                  <Message key={message.id} from={message.role}>
                    <div
                      className={cn(
                        "flex gap-3",
                        message.role === "user" ? "flex-row-reverse" : "flex-row"
                      )}
                    >
                      {/* Avatar */}
                      <div
                        className={cn(
                          "flex size-7 shrink-0 items-center justify-center rounded-full",
                          message.role === "user"
                            ? "bg-primary/15 text-primary"
                            : "bg-muted/80 text-muted-foreground"
                        )}
                      >
                        {message.role === "user" ? (
                          <User className="size-3.5" />
                        ) : (
                          <Bot className="size-3.5" />
                        )}
                      </div>

                      {/* Content */}
                      <MessageContent
                        className={cn(
                          "!text-[15px] leading-relaxed",
                          message.role === "user" && "!max-w-[80%]"
                        )}
                      >
                        {[...message.parts]
                          .sort((a, b) => {
                            if (a.type === "reasoning" && b.type !== "reasoning") return -1;
                            if (a.type !== "reasoning" && b.type === "reasoning") return 1;
                            return 0;
                          })
                          .map((part, i) => {
                          // --- Reasoning tokens ---
                          if (part.type === "reasoning") {
                            const isReasoningDone = message.parts.some(
                              (p, j) => j > i && p.type === "text"
                            );
                            const isActivelyStreaming =
                              status === "streaming" &&
                              message.role === "assistant" &&
                              !isReasoningDone;

                            return (
                              <Reasoning key={i} isStreaming={isActivelyStreaming}>
                                <ReasoningTrigger
                                  getThinkingMessage={(streaming, duration) => {
                                    if (streaming || duration === 0) {
                                      return (
                                        <span className="flex items-center gap-1.5 text-primary/70">
                                          <BrainCircuit className="size-3.5 animate-pulse" />
                                          Pensando...
                                        </span>
                                      );
                                    }
                                    if (duration === undefined) {
                                      return <span>Pensou por alguns segundos</span>;
                                    }
                                    return <span>Pensou por {duration}s</span>;
                                  }}
                                />
                                <ReasoningContent>{part.text}</ReasoningContent>
                              </Reasoning>
                            );
                          }

                          // --- Text parts ---
                          if (part.type === "text") {
                            if (message.role === "user") {
                              return <p key={i}>{part.text}</p>;
                            }
                            return (
                              <MessageResponse key={i}>
                                {part.text}
                              </MessageResponse>
                            );
                          }

                          // --- Tool: searchMatchData ---
                          if (part.type === "tool-searchMatchData") {
                            if (part.state === "input-streaming") {
                              return (
                                <div
                                  key={i}
                                  className="flex items-center gap-2 rounded-xl border border-border/40 bg-muted/30 px-3.5 py-2.5 text-sm text-muted-foreground"
                                >
                                  <Search className="size-4 animate-pulse text-primary/60" />
                                  <span>Preparando busca...</span>
                                </div>
                              );
                            }

                            if (part.state === "input-available") {
                              const toolInput = part.input as {
                                query?: string;
                                event_type?: string;
                                map_name?: string;
                              };
                              return (
                                <div
                                  key={i}
                                  className="flex items-center gap-2 rounded-xl border border-border/40 bg-muted/30 px-3.5 py-2.5 text-sm text-muted-foreground"
                                >
                                  <Search className="size-4 animate-pulse text-primary/60" />
                                  <div className="flex flex-col gap-0.5">
                                    <span>Buscando no banco de dados...</span>
                                    {toolInput.query && (
                                      <span className="text-xs text-muted-foreground/60 italic truncate max-w-md">
                                        &quot;{toolInput.query}&quot;
                                        {toolInput.event_type && ` [${toolInput.event_type}]`}
                                        {toolInput.map_name && ` [${toolInput.map_name}]`}
                                      </span>
                                    )}
                                  </div>
                                </div>
                              );
                            }

                            if (part.state === "output-available") {
                              const output = part.output as {
                                results_returned?: number;
                                retrieval_ms?: number;
                                results?: {
                                  rank: number;
                                  score: number | null;
                                  text: string;
                                  event_type: string | null;
                                  map: string | null;
                                  round: number | string | null;
                                }[];
                              };
                              const results = output.results ?? [];
                              const count = output.results_returned ?? results.length;
                              if (count === 0) {
                                return (
                                  <div
                                    key={i}
                                    className="flex items-center gap-2 rounded-xl border border-border/40 bg-muted/30 px-3.5 py-2.5 text-sm text-muted-foreground"
                                  >
                                    <Database className="size-4 text-muted-foreground/60" />
                                    <span>Nenhum resultado encontrado</span>
                                  </div>
                                );
                              }
                              return (
                                <Sources key={i} className="!text-sm">
                                  <SourcesTrigger count={count}>
                                    <div className="flex items-center gap-2 rounded-xl border border-border/40 bg-muted/30 px-3.5 py-2.5 text-sm text-muted-foreground cursor-pointer hover:bg-muted/50 transition-colors">
                                      <Database className="size-4 text-primary/60" />
                                      <span className="font-medium">
                                        {count} documentos recuperados
                                        {output.retrieval_ms
                                          ? ` em ${output.retrieval_ms}ms`
                                          : ""}
                                      </span>
                                      <ChevronDown className="size-4 ml-auto transition-transform [[data-state=open]_&]:rotate-180" />
                                    </div>
                                  </SourcesTrigger>
                                  <SourcesContent className="w-full max-w-full">
                                    <div className="mt-2 space-y-3">
                                      {results.map((doc, j) => (
                                        <div
                                          key={j}
                                          className="rounded-xl border border-border/40 bg-card/60 p-4 space-y-2.5"
                                        >
                                          <div className="flex items-center gap-2 flex-wrap">
                                            {doc.event_type && (
                                              <Badge variant="secondary" className="text-xs px-2 py-0.5">
                                                {doc.event_type}
                                              </Badge>
                                            )}
                                            {doc.map && (
                                              <Badge variant="outline" className="text-xs px-2 py-0.5">
                                                {doc.map}
                                              </Badge>
                                            )}
                                            {doc.round != null && (
                                              <Badge variant="outline" className="text-xs px-2 py-0.5">
                                                Round {doc.round}
                                              </Badge>
                                            )}
                                            {doc.score != null && (
                                              <span className="ml-auto text-xs text-muted-foreground/60">
                                                {(doc.score * 100).toFixed(1)}% relevância
                                              </span>
                                            )}
                                          </div>
                                          <p className="text-sm text-muted-foreground leading-relaxed line-clamp-3">
                                            {doc.text}
                                          </p>
                                        </div>
                                      ))}
                                    </div>
                                  </SourcesContent>
                                </Sources>
                              );
                            }
                          }
                          return null;
                        })}
                      </MessageContent>
                    </div>
                  </Message>
                ))}

                {/* Thinking indicator */}
                {status === "submitted" && (
                  <div className="flex gap-3">
                    <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-muted/80 text-muted-foreground">
                      <Bot className="size-3.5" />
                    </div>
                    <div className="flex items-center gap-1.5 py-3">
                      <span className="size-1.5 rounded-full bg-primary/50 animate-bounce [animation-delay:0ms]" />
                      <span className="size-1.5 rounded-full bg-primary/50 animate-bounce [animation-delay:150ms]" />
                      <span className="size-1.5 rounded-full bg-primary/50 animate-bounce [animation-delay:300ms]" />
                      <span className="ml-2 text-sm text-muted-foreground/60">
                        {currentModel.supportsReasoning
                          ? "Iniciando raciocínio..."
                          : "Analisando dados..."}
                      </span>
                    </div>
                  </div>
                )}
              </ConversationContent>
              <ConversationScrollButton />
            </Conversation>

            {/* Bottom Input */}
            <div className="border-t border-border/20 bg-background/80 backdrop-blur-sm">
              <div className="mx-auto max-w-3xl px-4 py-4">
                <PromptInput
                  onSubmit={handleSubmit}
                  className={cn(
                    "rounded-2xl border-border/30 bg-card/50 backdrop-blur-xl",
                    "shadow-md shadow-black/5",
                    "transition-all duration-200",
                    "focus-within:!border-primary/25 focus-within:!shadow-lg focus-within:!shadow-primary/5"
                  )}
                >
                  <PromptInputBody>
                    <PromptInputTextarea
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      placeholder="Pergunte sobre partidas, armas, jogadores..."
                      className="!min-h-[48px] !py-3 !px-4 !text-[15px] placeholder:text-muted-foreground/35"
                    />
                  </PromptInputBody>
                  {inputControls}
                </PromptInput>
              </div>
            </div>
          </>
        )}
      </main>
    </div>
  );
}
