import {
  deleteStoredSession,
  updateStoredSessionTitle,
} from "@/lib/chat-session-store";
import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

export async function PATCH(
  req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const { title } = await req.json();
  updateStoredSessionTitle(id, title);
  return NextResponse.json({ ok: true });
}

export async function DELETE(
  _req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  deleteStoredSession(id);
  return NextResponse.json({ ok: true });
}
