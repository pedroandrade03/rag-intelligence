"use client";

import type { ChatStatus, UIMessage } from "ai";

import { memo } from "react";
import { Bot, User } from "lucide-react";

import {
  Conversation,
  ConversationContent,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation";
import {
  Message,
  MessageContent,
} from "@/components/ai-elements/message";
import {
  MessagePartRenderer,
  sortMessageParts,
  ThinkingIndicator,
} from "@/components/chat/message-part-renderer";
import { cn } from "@/lib/utils";

interface ChatMessageListProps {
  isLoading?: boolean;
  messages: UIMessage[];
  status: ChatStatus;
  supportsReasoning: boolean;
}

export const ChatMessageList = memo(function ChatMessageList({
  isLoading = false,
  messages,
  status,
  supportsReasoning,
}: ChatMessageListProps) {
  if (messages.length === 0 && isLoading) {
    return (
      <Conversation className="flex-1 overflow-hidden">
        <ConversationContent className="mx-auto w-full max-w-3xl space-y-6 px-4 py-6">
          <ThinkingIndicator supportsReasoning={supportsReasoning} />
        </ConversationContent>
        <ConversationScrollButton />
      </Conversation>
    );
  }

  return (
    <Conversation className="flex-1 overflow-hidden">
      <ConversationContent className="mx-auto w-full max-w-3xl space-y-6 px-4 py-6">
        {messages.map((message) => (
          <Message from={message.role} key={message.id}>
            <div
              className={cn(
                "flex gap-3",
                message.role === "user" ? "flex-row-reverse" : "flex-row"
              )}
            >
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

              <MessageContent
                className={cn(
                  "!text-[15px] leading-relaxed",
                  message.role === "user" && "!max-w-[80%]"
                )}
              >
                {sortMessageParts(message.parts).map((part, index) => (
                  <MessagePartRenderer
                    index={index}
                    key={`${message.id}-${index}`}
                    message={message}
                    part={part}
                    status={status}
                  />
                ))}
              </MessageContent>
            </div>
          </Message>
        ))}

        {status === "submitted" && (
          <ThinkingIndicator supportsReasoning={supportsReasoning} />
        )}
      </ConversationContent>
      <ConversationScrollButton />
    </Conversation>
  );
});
