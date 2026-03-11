"use client";

import { memo } from "react";
import { MessageSquarePlus, PanelLeftClose, Trash2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import type { ChatSession } from "@/lib/chat-sessions";
import { cn } from "@/lib/utils";

interface ChatSidebarProps {
  activeChatId: string | null;
  onClose: () => void;
  onDeleteChat: (id: string) => void;
  onNewChat: () => void;
  onSelectChat: (id: string) => void;
  open: boolean;
  sessions: ChatSession[];
}

export const ChatSidebar = memo(function ChatSidebar({
  activeChatId,
  onClose,
  onDeleteChat,
  onNewChat,
  onSelectChat,
  open,
  sessions,
}: ChatSidebarProps) {
  return (
    <aside
      className={cn(
        "flex flex-col border-r border-border/50 bg-card/40 transition-all duration-300 ease-in-out",
        open ? "w-64" : "w-0 overflow-hidden border-r-0"
      )}
    >
      <div className="flex items-center justify-between px-4 py-4">
        <span className="text-sm font-medium tracking-tight text-foreground/60">
          RAG Intelligence
        </span>
        <Button
          className="text-muted-foreground/60 hover:text-foreground"
          onClick={onClose}
          size="icon-sm"
          variant="ghost"
        >
          <PanelLeftClose className="size-4" />
        </Button>
      </div>

      <div className="px-3 pb-2">
        <Button
          className="w-full justify-start gap-2 rounded-lg text-muted-foreground hover:text-foreground"
          onClick={onNewChat}
          variant="ghost"
        >
          <MessageSquarePlus className="size-4" />
          Nova Conversa
        </Button>
      </div>

      <Separator className="opacity-30" />

      <div className="px-3 py-2">
        <p className="px-2 text-[11px] font-medium uppercase tracking-widest text-muted-foreground/50">
          Histórico
        </p>
      </div>

      <ScrollArea className="flex-1 px-3">
        <div className="space-y-0.5 pb-4">
          {sessions.map((session) => (
            <div
              className={cn(
                "group flex cursor-pointer items-center justify-between rounded-lg px-3 py-2 text-sm transition-colors",
                session.id === activeChatId
                  ? "bg-accent/80 text-foreground"
                  : "text-muted-foreground hover:bg-accent/40 hover:text-foreground"
              )}
              key={session.id}
              onClick={() => onSelectChat(session.id)}
            >
              <span className="truncate">{session.title}</span>
              <Button
                className="shrink-0 opacity-0 transition-opacity group-hover:opacity-100"
                onClick={(event) => {
                  event.stopPropagation();
                  onDeleteChat(session.id);
                }}
                size="icon-xs"
                variant="ghost"
              >
                <Trash2 className="size-3" />
              </Button>
            </div>
          ))}
        </div>
      </ScrollArea>
    </aside>
  );
});
