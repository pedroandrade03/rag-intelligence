import { ChatApp } from "@/components/chat/chat-app";
import { getInitialChatState } from "@/lib/chat-session-store";

export const dynamic = "force-dynamic";

export default function Page() {
  const { activeChatId, initialMessages, sessions } = getInitialChatState();

  return (
    <ChatApp
      initialActiveChatId={activeChatId}
      initialMessages={initialMessages}
      initialSessions={sessions}
    />
  );
}
