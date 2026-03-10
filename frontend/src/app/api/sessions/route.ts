import { getDb } from "@/lib/db";
import { NextResponse } from "next/server";

export function GET() {
  const db = getDb();
  const sessions = db
    .prepare("SELECT * FROM sessions ORDER BY created_at DESC")
    .all();
  return NextResponse.json(sessions);
}

export async function POST(req: Request) {
  const { id, title } = await req.json();
  const db = getDb();
  db.prepare("INSERT INTO sessions (id, title) VALUES (?, ?)").run(id, title);
  return NextResponse.json({ id, title }, { status: 201 });
}
