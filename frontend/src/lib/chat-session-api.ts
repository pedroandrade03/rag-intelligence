import type { UIMessage } from "ai";

import {
  createDraftSession,
  type ChatSession,
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

export async function listSessions(): Promise<ChatSession[]> {
  return fetchJson<ChatSession[]>("/api/sessions");
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
