"use client";

import type { UIMessage } from "ai";

import { useChat } from "@ai-sdk/react";
import { Crosshair, PanelLeftOpen } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { ChatComposer } from "@/components/chat/chat-composer";
import { ChatEmptyState } from "@/components/chat/chat-empty-state";
import { ChatMessageList } from "@/components/chat/chat-message-list";
import { ChatSidebar } from "@/components/chat/chat-sidebar";
import { Button } from "@/components/ui/button";
import { useChatSessions } from "@/hooks/use-chat-sessions";
import { createSessionTitle } from "@/lib/chat-sessions";
import { DEFAULT_CHAT_MODEL, getChatModel, type RagMode } from "@/lib/chat-models";
import { LiveChatTransport } from "@/lib/live-chat-transport";

export default function Home() {
  const [input, setInput] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const shouldSaveRef = useRef(false);
  const [selectedModel, setSelectedModel] = useState(DEFAULT_CHAT_MODEL.id);
  const [ragMode, setRagMode] = useState<RagMode>("auto");

  const currentModel = getChatModel(selectedModel);
  const effectiveRagMode = currentModel.supportsTools ? ragMode : "off";
  const [transport] = useState(() => new LiveChatTransport());

  const { messages, status, sendMessage, stop, setMessages } = useChat({
    transport,
  });

  const isGenerating = status === "streaming" || status === "submitted";
  const hasMessages = messages.length > 0;

  useEffect(() => {
    transport.setRequestBodyResolver(() => ({
      model: selectedModel,
      ragMode: effectiveRagMode,
    }));
  }, [effectiveRagMode, selectedModel, transport]);

  const replaceMessages = useCallback(
    (nextMessages: UIMessage[]) => {
      setMessages(nextMessages as Parameters<typeof setMessages>[0]);
    },
    [setMessages]
  );

  const {
    activeChatId,
    chatSessions,
    createNewChat,
    deleteChat,
    isLoadingMessages,
    loaded,
    persistMessages,
    selectChat,
    updateSessionTitle,
  } = useChatSessions({
    replaceMessages,
  });

  useEffect(() => {
    if (shouldSaveRef.current && status === "ready" && messages.length > 0 && activeChatId) {
      shouldSaveRef.current = false;
      void persistMessages(activeChatId, messages);
    }
  }, [status, messages, activeChatId, persistMessages]);

  const sendPrompt = useCallback(
    (text: string) => {
      const trimmed = text.trim();

      if (!trimmed || isGenerating) {
        return;
      }

      if (messages.length === 0) {
        void updateSessionTitle(createSessionTitle(trimmed));
      }

      shouldSaveRef.current = true;
      sendMessage({ text: trimmed });
      setInput("");
    },
    [isGenerating, messages.length, sendMessage, updateSessionTitle]
  );

  const handleSubmit = useCallback(() => {
    sendPrompt(input);
  }, [input, sendPrompt]);

  const openSidebar = useCallback(() => setSidebarOpen(true), []);
  const closeSidebar = useCallback(() => setSidebarOpen(false), []);

  const handleNewChat = useCallback(() => {
    void createNewChat();
    setInput("");
  }, [createNewChat]);

  const handleDeleteChat = useCallback(
    (id: string) => {
      void deleteChat(id);
    },
    [deleteChat]
  );

  const showConversation = hasMessages || isLoadingMessages;

  if (!loaded) {
    return (
      <div className="flex h-dvh items-center justify-center bg-background">
        <Crosshair className="size-5 animate-pulse text-primary/50" />
      </div>
    );
  }

  return (
    <div className="flex h-dvh bg-background">
      <ChatSidebar
        activeChatId={activeChatId}
        onClose={closeSidebar}
        onDeleteChat={handleDeleteChat}
        onNewChat={handleNewChat}
        onSelectChat={selectChat}
        open={sidebarOpen}
        sessions={chatSessions}
      />

      <main className="relative flex min-w-0 flex-1 flex-col">
        {!sidebarOpen && (
          <Button
            className="absolute top-3 left-3 z-10 text-muted-foreground/50 hover:text-foreground"
            onClick={openSidebar}
            size="icon-sm"
            variant="ghost"
          >
            <PanelLeftOpen className="size-4" />
          </Button>
        )}

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
