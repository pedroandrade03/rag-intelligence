import type { UIMessage } from "ai";

import {
  createDraftSession,
  DEFAULT_CHAT_TITLE,
  type ChatSession,
  type ChatSessionRecord,
  toChatSession,
} from "@/lib/chat-sessions";
import { getDb } from "@/lib/db";

interface MessageRow {
  id: string;
  role: UIMessage["role"];
  parts: string;
}

export function listStoredSessions(): ChatSession[] {
  const db = getDb();
  const records = db
    .prepare("SELECT * FROM sessions ORDER BY created_at DESC")
    .all() as ChatSessionRecord[];

  return records.map(toChatSession);
}

export function createStoredSession(
  title: string = DEFAULT_CHAT_TITLE,
  id?: string
): ChatSession {
  const session = id ? { ...createDraftSession(title), id } : createDraftSession(title);
  const db = getDb();

  db.prepare("INSERT INTO sessions (id, title) VALUES (?, ?)").run(
    session.id,
    session.title
  );

  return session;
}

export function updateStoredSessionTitle(id: string, title: string) {
  const db = getDb();
  db.prepare("UPDATE sessions SET title = ? WHERE id = ?").run(title, id);
}

export function deleteStoredSession(id: string) {
  const db = getDb();
  db.prepare("DELETE FROM sessions WHERE id = ?").run(id);
}

export function listStoredSessionMessages(id: string): UIMessage[] {
  const db = getDb();
  const rows = db
    .prepare(
      "SELECT id, role, parts FROM messages WHERE session_id = ? ORDER BY rowid ASC"
    )
    .all(id) as MessageRow[];

  return rows.map((row) => ({
    id: row.id,
    role: row.role,
    parts: JSON.parse(row.parts),
  }));
}

export function upsertStoredSessionMessage(id: string, message: UIMessage) {
  const db = getDb();

  db.prepare(
    `
      INSERT INTO messages (id, session_id, role, parts)
      VALUES (?, ?, ?, ?)
      ON CONFLICT(id) DO UPDATE SET
        role = excluded.role,
        parts = excluded.parts
    `
  ).run(message.id, id, message.role, JSON.stringify(message.parts));
}

export function getInitialChatState(): {
  activeChatId: string;
  initialMessages: UIMessage[];
  sessions: ChatSession[];
} {
  const sessions = listStoredSessions();

  if (sessions.length === 0) {
    const session = createStoredSession();

    return {
      activeChatId: session.id,
      initialMessages: [],
      sessions: [session],
    };
  }

  const activeChatId = sessions[0].id;

  return {
    activeChatId,
    initialMessages: listStoredSessionMessages(activeChatId),
    sessions,
  };
}
