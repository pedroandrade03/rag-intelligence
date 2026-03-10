import { getDb } from "@/lib/db";
import { NextResponse } from "next/server";

interface MessageRow {
  id: string;
  role: string;
  parts: string;
}

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const db = getDb();
  const rows = db
    .prepare(
      "SELECT id, role, parts FROM messages WHERE session_id = ? ORDER BY created_at ASC"
    )
    .all(id) as MessageRow[];

  const messages = rows.map((r) => ({
    id: r.id,
    role: r.role,
    parts: JSON.parse(r.parts),
  }));
  return NextResponse.json(messages);
}

export async function PUT(
  req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const { messages } = await req.json();
  const db = getDb();

  const tx = db.transaction(() => {
    db.prepare("DELETE FROM messages WHERE session_id = ?").run(id);
    const insert = db.prepare(
      "INSERT INTO messages (id, session_id, role, parts) VALUES (?, ?, ?, ?)"
    );
    for (const msg of messages) {
      insert.run(msg.id, id, msg.role, JSON.stringify(msg.parts));
    }
  });
  tx();

  return NextResponse.json({ ok: true });
}
