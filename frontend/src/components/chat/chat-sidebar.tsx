"use client";

import { memo } from "react";
import { MessageSquarePlus, PanelLeftClose, PanelLeftOpen, Trash2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import type { ChatSession } from "@/lib/chat-sessions";
import { cn } from "@/lib/utils";

interface ChatSidebarProps {
  activeChatId: string | null;
  isBusy: boolean;
  onClose: () => void;
  onClearChats: () => void;
  onDeleteChat: (id: string) => void;
  onNewChat: () => void;
  onSelectChat: (id: string) => void;
  open: boolean;
  sessions: ChatSession[];
}

export const ChatSidebar = memo(function ChatSidebar({
  activeChatId,
  isBusy,
  onClose,
  onClearChats,
  onDeleteChat,
  onNewChat,
  onSelectChat,
  open,
  sessions,
}: ChatSidebarProps) {
  return (
    <aside
      className={cn(
        "flex flex-col border-r border-border bg-card transition-all duration-300 ease-in-out",
        open ? "w-64" : "w-14"
      )}
      style={{
        backgroundImage:
          "linear-gradient(to top, rgba(31,30,29,0.05), rgba(31,30,29,0.3))",
      }}
    >
      <div className={cn("flex items-center py-4", open ? "justify-between px-4" : "justify-center px-0")}>
        {open && (
          <span className="text-sm font-medium tracking-tight text-foreground/60">
            RAG Intelligence
          </span>
        )}
        <Button
          className="text-muted-foreground/60 hover:text-foreground"
          onClick={onClose}
          size="icon-sm"
          variant="ghost"
        >
          {open ? <PanelLeftClose className="size-4" /> : <PanelLeftOpen className="size-4" />}
        </Button>
      </div>

      <div className={cn(open ? "px-3" : "px-1.5", "pb-2")}>
        <Button
          className={cn(
            "w-full rounded-lg text-muted-foreground hover:text-foreground",
            open ? "justify-start gap-2" : "justify-center px-0"
          )}
          disabled={isBusy}
          onClick={onNewChat}
          variant="ghost"
        >
          <MessageSquarePlus className="size-4 shrink-0" />
          {open && <span>Nova Conversa</span>}
        </Button>
      </div>

      <Separator className="opacity-30" />

      {open && (
        <>
          <div className="px-3 py-2">
            <p className="px-2 text-[11px] font-medium uppercase tracking-widest text-muted-foreground/50">
              Histórico
            </p>
          </div>

          <ScrollArea className="flex-1 px-3">
            <div className="space-y-0.5 overflow-hidden pb-4">
              {sessions.map((session) => (
                <div
                  className={cn(
                    "group flex cursor-pointer items-center gap-1 rounded-lg px-3 py-2 text-sm transition-colors",
                    isBusy && "pointer-events-none opacity-60",
                    session.id === activeChatId
                      ? "bg-accent/80 text-foreground"
                      : "text-muted-foreground hover:bg-accent/40 hover:text-foreground"
                  )}
                  key={session.id}
                  onClick={() => onSelectChat(session.id)}
                >
                  <span className="min-w-0 flex-1 truncate">
                    {session.title}
                  </span>
                  <button
                    className="shrink-0 rounded p-1 text-muted-foreground/40 hover:text-destructive"
                    disabled={isBusy}
                    onClick={(event) => {
                      event.stopPropagation();
                      onDeleteChat(session.id);
                    }}
                    type="button"
                  >
                    <Trash2 className="size-3" />
                  </button>
                </div>
              ))}
            </div>
          </ScrollArea>
        </>
      )}

      <div className={cn("mt-auto border-t border-border/30", open ? "px-3 py-3" : "px-1.5 py-2")}>
        <Button
          aria-label="Excluir conversas"
          className={cn(
            "w-full rounded-lg text-destructive/80 hover:bg-destructive/10 hover:text-destructive",
            open ? "justify-start gap-2" : "justify-center px-0"
          )}
          disabled={isBusy}
          onClick={onClearChats}
          title="Excluir conversas"
          variant="ghost"
        >
          <Trash2 className="size-4 shrink-0" />
          {open && <span>Excluir Conversas</span>}
        </Button>
      </div>
    </aside>
  );
});
