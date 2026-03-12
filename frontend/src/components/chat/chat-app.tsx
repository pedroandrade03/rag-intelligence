"use client";

import type { UIMessage } from "ai";

import { useChat } from "@ai-sdk/react";
import { Crosshair } from "lucide-react";
import dynamic from "next/dynamic";
import { useCallback, useRef, useState } from "react";

import { ChatComposer } from "@/components/chat/chat-composer";
import { ChatSidebar } from "@/components/chat/chat-sidebar";
import {
  createSession,
  deleteSession,
  listSessionMessages,
  updateSessionTitle,
} from "@/lib/chat-session-api";
import {
  createSessionTitle,
  DEFAULT_CHAT_TITLE,
  type ChatSession,
} from "@/lib/chat-sessions";
import {
  DEFAULT_CHAT_MODEL,
  getChatModel,
  type RagMode,
} from "@/lib/chat-models";

const ChatEmptyState = dynamic(
  () => import("@/components/chat/chat-empty-state").then((mod) => mod.ChatEmptyState),
  {
    loading: () => (
      <div className="flex flex-1 items-center justify-center">
        <Crosshair className="size-5 animate-pulse text-primary/50" />
      </div>
    ),
  }
);

const ChatMessageList = dynamic(
  () =>
    import("@/components/chat/chat-message-list").then(
      (mod) => mod.ChatMessageList
    ),
  {
    loading: () => (
      <div className="flex flex-1 items-center justify-center">
        <Crosshair className="size-5 animate-pulse text-primary/50" />
      </div>
    ),
  }
);

interface ChatAppProps {
  initialActiveChatId: string;
  initialMessages: UIMessage[];
  initialSessions: ChatSession[];
}

export function ChatApp({
  initialActiveChatId,
  initialMessages,
  initialSessions,
}: ChatAppProps) {
  const [input, setInput] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [selectedModel, setSelectedModel] = useState(DEFAULT_CHAT_MODEL.id);
  const [ragMode, setRagMode] = useState<RagMode>("auto");
  const [chatSessions, setChatSessions] = useState(initialSessions);
  const [activeChatId, setActiveChatId] = useState(initialActiveChatId);
  const [isLoadingMessages, setIsLoadingMessages] = useState(false);
  const messageCacheRef = useRef(
    new Map<string, UIMessage[]>([[initialActiveChatId, initialMessages]])
  );

  const currentModel = getChatModel(selectedModel);
  const effectiveRagMode = currentModel.supportsTools ? ragMode : "off";

  const { messages, status, sendMessage, stop } = useChat({
    id: activeChatId,
    messages: messageCacheRef.current.get(activeChatId) ?? [],
    onFinish: ({ messages: nextMessages }) => {
      messageCacheRef.current.set(activeChatId, nextMessages);
    },
  });

  const isGenerating = status === "streaming" || status === "submitted";
  const hasMessages = messages.length > 0;

  const cacheActiveMessages = useCallback(() => {
    messageCacheRef.current.set(activeChatId, messages);
  }, [activeChatId, messages]);

  const ensureCachedMessages = useCallback(async (id: string) => {
    const cached = messageCacheRef.current.get(id);

    if (cached) {
      return cached;
    }

    const nextMessages = await listSessionMessages(id);
    messageCacheRef.current.set(id, nextMessages);
    return nextMessages;
  }, []);

  const renameActiveSession = useCallback(
    async (title: string) => {
      const nextTitle = title.trim() || DEFAULT_CHAT_TITLE;

      setChatSessions((current) =>
        current.map((session) =>
          session.id === activeChatId ? { ...session, title: nextTitle } : session
        )
      );

      try {
        await updateSessionTitle({ id: activeChatId, title: nextTitle });
      } catch (error) {
        console.error("Failed to update session title", error);
      }
    },
    [activeChatId]
  );

  const selectChat = useCallback(
    async (id: string) => {
      if (id === activeChatId || isGenerating) {
        return;
      }

      cacheActiveMessages();

      const shouldLoad = !messageCacheRef.current.has(id);
      if (shouldLoad) {
        setIsLoadingMessages(true);
      }

      try {
        await ensureCachedMessages(id);
        setInput("");
        setActiveChatId(id);
      } catch (error) {
        console.error("Failed to load session messages", error);
      } finally {
        if (shouldLoad) {
          setIsLoadingMessages(false);
        }
      }
    },
    [activeChatId, cacheActiveMessages, ensureCachedMessages, isGenerating]
  );

  const createNewChat = useCallback(async () => {
    if (isGenerating) {
      return;
    }

    cacheActiveMessages();

    try {
      const session = await createSession(DEFAULT_CHAT_TITLE);
      messageCacheRef.current.set(session.id, []);
      setChatSessions((current) => [session, ...current]);
      setInput("");
      setActiveChatId(session.id);
    } catch (error) {
      console.error("Failed to create session", error);
    }
  }, [cacheActiveMessages, isGenerating]);

  const deleteChatById = useCallback(
    async (id: string) => {
      if (isGenerating) {
        return;
      }

      try {
        await deleteSession(id);
      } catch (error) {
        console.error("Failed to delete session", error);
        return;
      }

      messageCacheRef.current.delete(id);

      let remaining: ChatSession[] = [];
      setChatSessions((current) => {
        remaining = current.filter((session) => session.id !== id);
        return remaining;
      });

      if (id !== activeChatId) {
        return;
      }

      if (remaining.length === 0) {
        try {
          const session = await createSession(DEFAULT_CHAT_TITLE);
          messageCacheRef.current.set(session.id, []);
          setChatSessions([session]);
          setInput("");
          setActiveChatId(session.id);
        } catch (error) {
          console.error("Failed to create replacement session", error);
        }
        return;
      }

      const nextActiveId = remaining[0].id;
      const shouldLoad = !messageCacheRef.current.has(nextActiveId);
      if (shouldLoad) {
        setIsLoadingMessages(true);
      }

      try {
        await ensureCachedMessages(nextActiveId);
        setInput("");
        setActiveChatId(nextActiveId);
      } catch (error) {
        console.error("Failed to load replacement session messages", error);
      } finally {
        if (shouldLoad) {
          setIsLoadingMessages(false);
        }
      }
    },
    [activeChatId, ensureCachedMessages, isGenerating]
  );

  const sendPrompt = useCallback(
    (text: string) => {
      const trimmed = text.trim();

      if (!trimmed || isGenerating) {
        return;
      }

      if (messages.length === 0) {
        void renameActiveSession(createSessionTitle(trimmed));
      }

      const userMessage: UIMessage = {
        id: crypto.randomUUID(),
        parts: [{ text: trimmed, type: "text" }],
        role: "user",
      };

      messageCacheRef.current.set(activeChatId, [...messages, userMessage]);
      setInput("");

      void sendMessage(userMessage, {
        body: {
          model: selectedModel,
          ragMode: effectiveRagMode,
          sessionId: activeChatId,
        },
      });
    },
    [
      activeChatId,
      effectiveRagMode,
      isGenerating,
      messages,
      renameActiveSession,
      selectedModel,
      sendMessage,
    ]
  );

  const handleSubmit = useCallback(() => {
    sendPrompt(input);
  }, [input, sendPrompt]);

  const toggleSidebar = useCallback(() => setSidebarOpen((prev) => !prev), []);

  const showConversation = hasMessages || isLoadingMessages;

  return (
    <div className="flex h-dvh bg-background">
      <ChatSidebar
        activeChatId={activeChatId}
        onClose={toggleSidebar}
        onDeleteChat={deleteChatById}
        onNewChat={createNewChat}
        onSelectChat={selectChat}
        open={sidebarOpen}
        sessions={chatSessions}
      />

      <main className="relative flex min-w-0 flex-1 flex-col">
        {!showConversation ? (
          <ChatEmptyState
            currentModel={currentModel}
            effectiveRagMode={effectiveRagMode}
            input={input}
            onInputChange={setInput}
            onRagModeChange={setRagMode}
            onSelectedModelChange={setSelectedModel}
            onStop={stop}
            onSubmit={handleSubmit}
            onSuggestionClick={sendPrompt}
            selectedModel={selectedModel}
            status={status}
          />
        ) : (
          <>
            <ChatMessageList
              isLoading={isLoadingMessages}
              messages={messages}
              status={status}
              supportsReasoning={currentModel.supportsReasoning}
            />

            <div className="border-t border-border/20 bg-background/80 backdrop-blur-sm">
              <div className="mx-auto max-w-3xl px-4 py-4">
                <ChatComposer
                  currentModel={currentModel}
                  effectiveRagMode={effectiveRagMode}
                  input={input}
                  onInputChange={setInput}
                  onRagModeChange={setRagMode}
                  onSelectedModelChange={setSelectedModel}
                  onStop={stop}
                  onSubmit={handleSubmit}
                  selectedModel={selectedModel}
                  status={status}
                />
              </div>
            </div>
          </>
        )}
      </main>
    </div>
  );
}
