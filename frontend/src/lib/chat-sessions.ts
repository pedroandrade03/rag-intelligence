import type { UIMessage } from "ai";

export const DEFAULT_CHAT_TITLE = "Nova Conversa";

export interface ChatSession {
  id: string;
  title: string;
  createdAt: string;
}

export interface ChatSessionRecord {
  id: string;
  title: string;
  created_at: string;
}

export type ChatSessionMessage = Pick<UIMessage, "id" | "role" | "parts">;

export function createChatSessionId(): string {
  return Date.now().toString();
}

export function createDraftSession(
  title: string = DEFAULT_CHAT_TITLE
): ChatSession {
  return {
    createdAt: new Date().toISOString(),
    id: createChatSessionId(),
    title,
  };
}

export function toChatSession(record: ChatSessionRecord): ChatSession {
  return {
    createdAt: record.created_at,
    id: record.id,
    title: record.title,
  };
}

export function createSessionTitle(
  text: string,
  maxLength: number = 40
): string {
  const trimmed = text.trim();

  if (!trimmed) {
    return DEFAULT_CHAT_TITLE;
  }

  return trimmed.length > maxLength
    ? `${trimmed.slice(0, maxLength)}...`
    : trimmed;
}
