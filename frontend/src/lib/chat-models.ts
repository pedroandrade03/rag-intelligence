export type RagMode = "auto" | "always" | "off";

export interface ChatModelOption {
  id: string;
  name: string;
  supportsReasoning: boolean;
  supportsTools: boolean;
}

export const CHAT_MODELS: ChatModelOption[] = [
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

export const DEFAULT_CHAT_MODEL = CHAT_MODELS[0];

export const RAG_MODE_CONFIG: Record<RagMode, { label: string; tip: string }> = {
  auto: { label: "Auto", tip: "Modelo decide quando buscar" },
  always: { label: "RAG", tip: "Sempre buscar no banco" },
  off: { label: "Off", tip: "Sem busca, apenas conhecimento geral" },
};

export const RAG_MODES: RagMode[] = ["auto", "always", "off"];

export function getChatModel(modelId?: string): ChatModelOption {
  return CHAT_MODELS.find((model) => model.id === modelId) ?? DEFAULT_CHAT_MODEL;
}
