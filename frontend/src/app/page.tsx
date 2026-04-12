import { ChatApp } from "@/components/chat/chat-app";
import { getChatRuntimeConfig } from "@/lib/chat-models";
import { getInitialChatState } from "@/lib/chat-session-store";

export const dynamic = "force-dynamic";

export default function Page() {
  const { activeChatId, initialMessages, sessions } = getInitialChatState();
  const { defaultModelId, models } = getChatRuntimeConfig();

  return (
    <ChatApp
      initialActiveChatId={activeChatId}
      initialMessages={initialMessages}
      initialSelectedModel={defaultModelId}
      initialSessions={sessions}
      modelOptions={models}
    />
  );
}
