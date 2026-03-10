"use client";

import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  ChevronDown,
  Crosshair,
  Database,
  MessageSquarePlus,
  PanelLeftClose,
  PanelLeftOpen,
  Search,
  Sword,
  Target,
  Trash2,
  Trophy,
  Users,
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
  Sources,
  SourcesTrigger,
  SourcesContent,
} from "@/components/ai-elements/sources";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";

const transport = new DefaultChatTransport({
  api: "/api/chat",
});

interface ChatSession {
  id: string;
  title: string;
  createdAt: Date;
}

const SUGGESTION_CARDS = [
  {
    icon: Sword,
    title: "Dano de Armas",
    query: "Quais armas causam mais dano por round em média?",
    color: "text-red-400",
  },
  {
    icon: Trophy,
    title: "Desempenho de Jogadores",
    query: "Quais são os melhores jogadores por razão de kill/death?",
    color: "text-amber-400",
  },
  {
    icon: Target,
    title: "Eventos de Partida",
    query: "Quais são os eventos mais comuns de fim de round em partidas competitivas?",
    color: "text-blue-400",
  },
  {
    icon: Users,
    title: "Estratégias de Time",
    query: "Quais composições de time têm as maiores taxas de vitória?",
    color: "text-emerald-400",
  },
  {
    icon: Zap,
    title: "Análise de Economia",
    query: "Como a economia do time afeta a probabilidade de vitória no round?",
    color: "text-purple-400",
  },
  {
    icon: Crosshair,
    title: "Estatísticas de Headshot",
    query: "Qual a porcentagem média de headshot entre os diferentes tipos de arma?",
    color: "text-orange-400",
  },
];

export default function Home() {
  const [input, setInput] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [chatSessions, setChatSessions] = useState<ChatSession[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);
  const shouldSaveRef = useRef(false);

  const { messages, status, sendMessage, stop, setMessages } = useChat({
    transport,
  });

  const isGenerating = status === "streaming" || status === "submitted";

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

  if (!loaded) {
    return (
      <div className="flex h-dvh items-center justify-center bg-background">
        <div className="flex items-center gap-2 text-muted-foreground">
          <Crosshair className="size-5 animate-pulse text-primary" />
          <span>Carregando...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-dvh bg-background">
      {/* Sidebar */}
      <aside
        className={cn(
          "flex flex-col border-r border-border bg-card transition-all duration-300 ease-in-out",
          sidebarOpen ? "w-72" : "w-0 overflow-hidden border-r-0"
        )}
      >
        {/* Sidebar Header */}
        <div className="flex items-center justify-between px-4 py-4">
          <div className="flex items-center gap-2">
            <div className="flex size-8 items-center justify-center rounded-lg bg-primary">
              <Crosshair className="size-4 text-primary-foreground" />
            </div>
            <span className="text-base font-semibold tracking-tight">
              RAG Intel
            </span>
          </div>
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={() => setSidebarOpen(false)}
            className="text-muted-foreground hover:text-foreground"
          >
            <PanelLeftClose className="size-4" />
          </Button>
        </div>

        {/* New Chat Button */}
        <div className="px-3 pb-2">
          <Button
            variant="outline"
            className="w-full justify-start gap-2 rounded-lg"
            onClick={handleNewChat}
          >
            <MessageSquarePlus className="size-4" />
            Nova Conversa
          </Button>
        </div>

        <Separator />

        {/* Chat History */}
        <div className="px-3 py-2">
          <p className="px-2 text-xs font-medium text-muted-foreground uppercase tracking-wider">
            Histórico
          </p>
        </div>
        <ScrollArea className="flex-1 px-3">
          <div className="space-y-1 pb-4">
            {chatSessions.map((session) => (
              <div
                key={session.id}
                className={cn(
                  "group flex items-center justify-between rounded-lg px-3 py-2.5 text-sm transition-colors cursor-pointer",
                  session.id === activeChatId
                    ? "bg-accent text-accent-foreground"
                    : "text-muted-foreground hover:bg-accent/50 hover:text-foreground"
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

      {/* Main Content */}
      <main className="flex flex-1 flex-col min-w-0">
        {/* Header */}
        <header className="flex items-center gap-3 border-b border-border px-4 py-3">
          {!sidebarOpen && (
            <Button
              variant="ghost"
              size="icon-sm"
              onClick={() => setSidebarOpen(true)}
              className="text-muted-foreground hover:text-foreground"
            >
              <PanelLeftOpen className="size-4" />
            </Button>
          )}
          <div className="flex items-center gap-2">
            <h1 className="text-lg font-semibold tracking-tight">
              RAG Intelligence
            </h1>
            <span className="rounded-md bg-primary/10 px-2 py-0.5 text-sm font-medium text-primary">
              CS:GO
            </span>
          </div>
        </header>

        {/* Conversation Area */}
        <Conversation className="flex-1 overflow-hidden">
          <ConversationContent className="space-y-6 px-4 py-6 max-w-4xl mx-auto w-full">
            {messages.length === 0 ? (
              /* Empty State */
              <div className="flex size-full flex-col items-center justify-center gap-8 px-4">
                {/* Hero */}
                <div className="flex flex-col items-center gap-4 text-center">
                  <div className="flex size-24 items-center justify-center rounded-2xl bg-gradient-to-br from-primary/20 to-primary/5 ring-1 ring-primary/10">
                    <Crosshair className="size-12 text-primary" />
                  </div>
                  <div className="space-y-3">
                    <h2 className="text-4xl font-bold tracking-tight">
                      Analista de Partidas CS:GO
                    </h2>
                    <p className="text-lg text-muted-foreground max-w-md">
                      Faça perguntas sobre eventos de partida, dano de armas,
                      desempenho de jogadores, economia e muito mais.
                    </p>
                  </div>
                </div>

                {/* Suggestion Cards Grid */}
                <div className="grid w-full max-w-2xl grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
                  {SUGGESTION_CARDS.map((card) => (
                    <button
                      key={card.title}
                      onClick={() => handleSuggestionClick(card.query)}
                      className="group flex flex-col gap-2.5 rounded-xl border border-border bg-card p-5 text-left cursor-pointer transition-all hover:border-primary/30 hover:bg-accent/50 hover:shadow-md hover:shadow-primary/5"
                    >
                      <div className="flex items-center gap-2.5">
                        <card.icon className={cn("size-5", card.color)} />
                        <span className="text-base font-medium">
                          {card.title}
                        </span>
                      </div>
                      <p className="text-sm text-muted-foreground leading-relaxed line-clamp-2">
                        {card.query}
                      </p>
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              /* Messages */
              messages.map((message) => (
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
                        "flex size-8 shrink-0 items-center justify-center rounded-lg",
                        message.role === "user"
                          ? "bg-primary text-primary-foreground"
                          : "bg-muted text-muted-foreground"
                      )}
                    >
                      {message.role === "user" ? (
                        <User className="size-4" />
                      ) : (
                        <Bot className="size-4" />
                      )}
                    </div>

                    {/* Content */}
                    <MessageContent
                      className={cn(
                        "!text-base leading-relaxed",
                        message.role === "user" && "!max-w-[80%]"
                      )}
                    >
                      {message.parts.map((part, i) => {
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
                        if (part.type === "tool-searchMatchData") {
                          if (
                            part.state === "input-streaming" ||
                            part.state === "input-available"
                          ) {
                            return (
                              <div
                                key={i}
                                className="flex items-center gap-2 rounded-lg border border-border bg-muted/50 px-3.5 py-2.5 text-sm text-muted-foreground"
                              >
                                <Search className="size-4 animate-pulse text-primary" />
                                <span>Buscando no banco de dados...</span>
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
                                  className="flex items-center gap-2 rounded-lg border border-border bg-muted/50 px-3.5 py-2.5 text-sm text-muted-foreground"
                                >
                                  <Database className="size-4 text-muted-foreground" />
                                  <span>Nenhum resultado encontrado</span>
                                </div>
                              );
                            }
                            return (
                              <Sources key={i} className="!text-sm">
                                <SourcesTrigger count={count}>
                                  <div className="flex items-center gap-2 rounded-lg border border-border bg-muted/50 px-3.5 py-2.5 text-sm text-muted-foreground cursor-pointer hover:bg-muted transition-colors">
                                    <Database className="size-4 text-emerald-400" />
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
                                        className="rounded-xl border border-border bg-card p-4 space-y-2.5"
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
                                            <span className="ml-auto text-xs text-muted-foreground">
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
              ))
            )}

            {/* Thinking indicator */}
            {status === "submitted" && (
              <div className="flex gap-3">
                <div className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-muted text-muted-foreground">
                  <Bot className="size-4" />
                </div>
                <div className="flex items-center gap-1.5 py-3">
                  <span className="size-2 rounded-full bg-primary/60 animate-bounce [animation-delay:0ms]" />
                  <span className="size-2 rounded-full bg-primary/60 animate-bounce [animation-delay:150ms]" />
                  <span className="size-2 rounded-full bg-primary/60 animate-bounce [animation-delay:300ms]" />
                  <span className="ml-2 text-sm text-muted-foreground">
                    Analisando dados da partida...
                  </span>
                </div>
              </div>
            )}
          </ConversationContent>
          <ConversationScrollButton />
        </Conversation>

        {/* Input Area */}
        <div className="border-t border-border bg-background/80 backdrop-blur-sm">
          <div className="mx-auto max-w-4xl px-4 py-5">
            <PromptInput
              onSubmit={handleSubmit}
              className={cn(
                "rounded-2xl border-border bg-card shadow-sm",
                "transition-all duration-200",
                "focus-within:!shadow-lg focus-within:!shadow-primary/10",
                "focus-within:!border-primary/40",
                "focus-within:!ring-2 focus-within:!ring-primary/20"
              )}
            >
              <PromptInputBody>
                <PromptInputTextarea
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="Pergunte sobre partidas, armas, jogadores..."
                  className="!min-h-[64px] !py-3 !text-lg placeholder:!text-lg placeholder:text-muted-foreground/60"
                />
              </PromptInputBody>
              <PromptInputFooter className="justify-between px-4 pb-3">
                <span className="text-sm text-muted-foreground">
                  Pressione <kbd className="rounded border border-border bg-muted px-1.5 py-0.5 font-mono text-xs">Enter</kbd> para enviar
                </span>
                <PromptInputSubmit status={status} onStop={stop} />
              </PromptInputFooter>
            </PromptInput>
          </div>
        </div>
      </main>
    </div>
  );
}
