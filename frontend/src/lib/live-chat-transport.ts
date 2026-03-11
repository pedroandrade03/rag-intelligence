import type { UIMessage } from "ai";

import { DefaultChatTransport } from "ai";

import type { RagMode } from "@/lib/chat-models";

interface ChatRequestBody {
  model: string;
  ragMode: RagMode;
}

export class LiveChatTransport<
  UI_MESSAGE extends UIMessage = UIMessage,
> extends DefaultChatTransport<UI_MESSAGE> {
  private resolveBody: () => ChatRequestBody = () => ({
    model: "",
    ragMode: "auto",
  });

  constructor() {
    super({
      api: "/api/chat",
      body: () => this.resolveBody(),
    });
  }

  setRequestBodyResolver(resolver: () => ChatRequestBody) {
    this.resolveBody = resolver;
  }
}
