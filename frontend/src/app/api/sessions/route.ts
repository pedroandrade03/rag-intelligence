import {
  createStoredSession,
  listStoredSessions,
  resetStoredSessions,
} from "@/lib/chat-session-store";
import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

export function GET() {
  return NextResponse.json(listStoredSessions());
}

export async function POST(req: Request) {
  const { id, title } = await req.json();
  const session = createStoredSession(title, id);
  return NextResponse.json(session, { status: 201 });
}

export function DELETE() {
  return NextResponse.json(resetStoredSessions());
}
