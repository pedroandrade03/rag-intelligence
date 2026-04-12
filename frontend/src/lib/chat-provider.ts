import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { createOllama } from "ollama-ai-provider-v2";

export type ChatProviderKind = "ollama" | "openai-compatible";

export interface ChatProviderConfig {
  apiKey: string;
  baseURL: string;
  kind: ChatProviderKind;
}

export function getChatProviderConfig(
  env: Record<string, string | undefined> = process.env
): ChatProviderConfig {
  const kind =
    env.CHAT_PROVIDER?.trim().toLowerCase() === "openai-compatible"
      ? "openai-compatible"
      : "ollama";

  if (kind === "openai-compatible") {
    return {
      apiKey: env.CHAT_PROVIDER_API_KEY?.trim() || "local",
      baseURL: env.CHAT_PROVIDER_BASE_URL?.trim() || "http://127.0.0.1:8080/v1",
      kind,
    };
  }

  return {
    apiKey: "",
    baseURL: env.OLLAMA_BASE_URL?.trim() || "http://localhost:11434",
    kind,
  };
}

export function getChatProvider(
  env: Record<string, string | undefined> = process.env
) {
  const config = getChatProviderConfig(env);

  if (config.kind === "openai-compatible") {
    return {
      config,
      provider: createOpenAICompatible({
        name: "llama-server",
        apiKey: config.apiKey,
        baseURL: config.baseURL,
      }),
    };
  }

  return {
    config,
    provider: createOllama({
      baseURL: `${config.baseURL}/api`,
    }),
  };
}
