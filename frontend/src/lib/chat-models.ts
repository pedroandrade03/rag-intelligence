export type RagMode = "auto" | "always" | "off";

export interface ChatModelOption {
  id: string;
  name: string;
  supportsReasoning: boolean;
  supportsTools: boolean;
}

export interface ChatRuntimeConfig {
  defaultModelId: string;
  models: ChatModelOption[];
}

const BASE_CHAT_MODELS: ChatModelOption[] = [
  {
    id: "qwen2.5:7b-instruct-q4_K_M",
    name: "Qwen 2.5 7B",
    supportsReasoning: false,
    supportsTools: true,
  },
  {
    id: "qwen3:8b",
    name: "Qwen3 8B",
    supportsReasoning: true,
    supportsTools: true,
  },
  {
    id: "deepseek-r1:8b",
    name: "DeepSeek R1 8B",
    supportsReasoning: true,
    supportsTools: false,
  },
];

function parseBoolean(value: string | undefined, fallback: boolean): boolean {
  if (value == null || value.trim() === "") {
    return fallback;
  }

  return ["1", "true", "yes", "on"].includes(value.trim().toLowerCase());
}

export function getChatRuntimeConfig(
  env: Record<string, string | undefined> = process.env
): ChatRuntimeConfig {
  const ollamaFallbackModelId =
    env.CHAT_PROVIDER?.trim().toLowerCase() === "openai-compatible"
      ? ""
      : env.OLLAMA_MODEL?.trim() || "";
  const customModelId =
    env.CHAT_MODEL?.trim() ||
    env.NEXT_PUBLIC_CHAT_MODEL?.trim() ||
    ollamaFallbackModelId ||
    "";
  const defaultModelId = env.CHAT_DEFAULT_MODEL?.trim() || customModelId;

  const models = [...BASE_CHAT_MODELS];
  if (customModelId && !models.some((model) => model.id === customModelId)) {
    models.unshift({
      id: customModelId,
      name: env.CHAT_MODEL_LABEL?.trim() || env.NEXT_PUBLIC_CHAT_MODEL_LABEL?.trim() || customModelId,
      supportsReasoning: parseBoolean(
        env.CHAT_MODEL_SUPPORTS_REASONING ?? env.NEXT_PUBLIC_CHAT_SUPPORTS_REASONING,
        false
      ),
      supportsTools: parseBoolean(
        env.CHAT_MODEL_SUPPORTS_TOOLS ?? env.NEXT_PUBLIC_CHAT_SUPPORTS_TOOLS,
        true
      ),
    });
  }

  return {
    defaultModelId: defaultModelId || models[0]?.id || BASE_CHAT_MODELS[0].id,
    models,
  };
}

export const DEFAULT_CHAT_MODEL = BASE_CHAT_MODELS[0];

export const RAG_MODE_CONFIG: Record<RagMode, { label: string; tip: string }> = {
  auto: { label: "Auto", tip: "Modelo decide quando buscar" },
  always: { label: "RAG", tip: "Sempre buscar no banco" },
  off: { label: "Off", tip: "Sem busca, apenas conhecimento geral" },
};

export const RAG_MODES: RagMode[] = ["auto", "always", "off"];

export function getChatModel(
  models: readonly ChatModelOption[],
  modelId?: string
): ChatModelOption {
  return models.find((model) => model.id === modelId) ?? models[0] ?? DEFAULT_CHAT_MODEL;
}
