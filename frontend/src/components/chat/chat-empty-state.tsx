"use client";

import { motion } from "motion/react";
import type { LucideIcon } from "lucide-react";
import { Crosshair, Sword, Target, Trophy, Zap } from "lucide-react";

import { ChatComposer } from "@/components/chat/chat-composer";
import type { ChatModelOption, RagMode } from "@/lib/chat-models";
import { cn } from "@/lib/utils";

interface Suggestion {
  icon: LucideIcon;
  label: string;
  query: string;
}

const SUGGESTIONS: Suggestion[] = [
  {
    icon: Sword,
    label: "Dano de Armas",
    query: "Quais armas causam mais dano por round em média?",
  },
  {
    icon: Trophy,
    label: "Jogadores",
    query: "Quais são os melhores jogadores de CS:GO?",
  },
  {
    icon: Target,
    label: "Eventos de Round",
    query: "Quais são os eventos mais comuns de fim de round?",
  },
  {
    icon: Zap,
    label: "Economia",
    query: "Como a economia afeta o resultado dos rounds?",
  },
];

interface ChatEmptyStateProps {
  currentModel: ChatModelOption;
  effectiveRagMode: RagMode;
  input: string;
  onInputChange: (value: string) => void;
  onRagModeChange: (mode: RagMode) => void;
  onSelectedModelChange: (modelId: string) => void;
  onStop: () => void;
  onSubmit: () => void;
  onSuggestionClick: (query: string) => void;
  selectedModel: string;
  status: "submitted" | "streaming" | "ready" | "error";
}

export function ChatEmptyState({
  currentModel,
  effectiveRagMode,
  input,
  onInputChange,
  onRagModeChange,
  onSelectedModelChange,
  onStop,
  onSubmit,
  onSuggestionClick,
  selectedModel,
  status,
}: ChatEmptyStateProps) {
  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-6 px-4 pb-16">
      <motion.div
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center gap-3"
        initial={{ opacity: 0, y: 16 }}
        transition={{ duration: 0.6, ease: [0.25, 0.46, 0.45, 0.94] }}
      >
        <Crosshair className="size-7 text-primary/40" />
        <h1 className="font-sans text-5xl font-bold tracking-tight text-foreground/70">
          Analista CS:GO
        </h1>
      </motion.div>

      <motion.div
        animate={{ opacity: 1, y: 0 }}
        className="w-full max-w-[640px]"
        initial={{ opacity: 0, y: 10 }}
        transition={{ delay: 0.1, duration: 0.5, ease: [0.25, 0.46, 0.45, 0.94] }}
      >
        <ChatComposer
          currentModel={currentModel}
          effectiveRagMode={effectiveRagMode}
          input={input}
          onInputChange={onInputChange}
          onRagModeChange={onRagModeChange}
          onSelectedModelChange={onSelectedModelChange}
          onStop={onStop}
          onSubmit={onSubmit}
          selectedModel={selectedModel}
          status={status}
          variant="hero"
        />
      </motion.div>

      <motion.div
        animate={{ opacity: 1 }}
        className="flex flex-wrap justify-center gap-2"
        initial={{ opacity: 0 }}
        transition={{ delay: 0.25, duration: 0.5 }}
      >
        {SUGGESTIONS.map((suggestion) => (
          <button
            className={cn(
              "flex items-center gap-2 rounded-full border border-border bg-popover/60 px-4 py-2",
              "text-sm text-muted-foreground",
              "transition-all duration-200",
              "hover:bg-popover hover:text-foreground hover:border-border"
            )}
            key={suggestion.label}
            onClick={() => onSuggestionClick(suggestion.query)}
            type="button"
          >
            <suggestion.icon className="size-3.5" />
            {suggestion.label}
          </button>
        ))}
      </motion.div>
    </div>
  );
}
