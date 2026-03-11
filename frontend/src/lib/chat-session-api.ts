import type { UIMessage } from "ai";

import {
  createDraftSession,
  type ChatSession,
  type ChatSessionRecord,
  toChatSession,
} from "@/lib/chat-sessions";

async function fetchJson<T>(
  input: RequestInfo | URL,
  init?: RequestInit
): Promise<T> {
  const response = await fetch(input, init);

  if (!response.ok) {
    throw new Error(
      (await response.text()) || `Request failed with status ${response.status}`
    );
  }

  return (await response.json()) as T;
}

export const sessionKeys = {
  all: ["sessions"] as const,
  list: () => ["sessions"] as const,
  messages: (id: string) => ["sessions", id, "messages"] as const,
};

export async function listSessions(): Promise<ChatSession[]> {
  const records = await fetchJson<ChatSessionRecord[]>("/api/sessions");
  return records.map(toChatSession);
}

export async function createSession(title: string): Promise<ChatSession> {
  const session = createDraftSession(title);

  await fetchJson("/api/sessions", {
    body: JSON.stringify({ id: session.id, title: session.title }),
    headers: { "Content-Type": "application/json" },
    method: "POST",
  });

  return session;
}

export async function updateSessionTitle(input: {
  id: string;
  title: string;
}) {
  await fetchJson(`/api/sessions/${input.id}`, {
    body: JSON.stringify({ title: input.title }),
    headers: { "Content-Type": "application/json" },
    method: "PATCH",
  });

  return input;
}

export async function deleteSession(id: string) {
  await fetchJson(`/api/sessions/${id}`, {
    method: "DELETE",
  });

  return id;
}

export async function listSessionMessages(id: string): Promise<UIMessage[]> {
  return fetchJson<UIMessage[]>(`/api/sessions/${id}/messages`);
}

export async function persistSessionMessages(input: {
  id: string;
  messages: UIMessage[];
}) {
  await fetchJson(`/api/sessions/${input.id}/messages`, {
    body: JSON.stringify({ messages: input.messages }),
    headers: { "Content-Type": "application/json" },
    method: "PUT",
  });

  return input.messages;
}
