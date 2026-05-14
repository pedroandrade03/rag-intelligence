import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { createOllama } from "ollama-ai-provider-v2";

export type ChatProviderKind = "ollama" | "openai-compatible";

export interface ChatProviderConfig {
  apiKey: string;
  baseURL: string;
  kind: ChatProviderKind;
  name: string;
}

export function getChatProviderConfig(
  env: Record<string, string | undefined> = process.env,
  modelId?: string
): ChatProviderConfig {
  const localModelId = env.LOCAL_CHAT_MODEL?.trim() || "gemma4";
  if (modelId === localModelId) {
    return {
      apiKey: env.LOCAL_CHAT_PROVIDER_API_KEY?.trim() || "local",
      baseURL:
        env.LOCAL_CHAT_PROVIDER_BASE_URL?.trim() ||
        env.LLAMA_CPP_BASE_URL?.trim() ||
        "http://127.0.0.1:8080/v1",
      kind: "openai-compatible",
      name: "llama-server",
    };
  }

  const kind =
    env.CHAT_PROVIDER?.trim().toLowerCase() === "openai-compatible"
      ? "openai-compatible"
      : "ollama";

  if (kind === "openai-compatible") {
    return {
      apiKey: env.CHAT_PROVIDER_API_KEY?.trim() || "local",
      baseURL: env.CHAT_PROVIDER_BASE_URL?.trim() || "http://127.0.0.1:8080/v1",
      kind,
      name: env.CHAT_PROVIDER_NAME?.trim() || "openai-compatible",
    };
  }

  return {
    apiKey: "",
    baseURL: env.OLLAMA_BASE_URL?.trim() || "http://localhost:11434",
    kind,
    name: "ollama",
  };
}

export function getChatProvider(
  env: Record<string, string | undefined> = process.env,
  modelId?: string
) {
  const config = getChatProviderConfig(env, modelId);

  if (config.kind === "openai-compatible") {
    return {
      config,
      provider: createOpenAICompatible({
        name: config.name,
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
