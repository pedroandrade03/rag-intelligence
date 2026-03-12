"use client";

import type { ChatStatus } from "ai";

import { memo } from "react";
import { BrainCircuit, Search, SearchX } from "lucide-react";

import {
  PromptInput,
  PromptInputBody,
  PromptInputFooter,
  PromptInputSubmit,
  PromptInputTextarea,
} from "@/components/ai-elements/prompt-input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  CHAT_MODELS,
  RAG_MODE_CONFIG,
  RAG_MODES,
  type ChatModelOption,
  type RagMode,
} from "@/lib/chat-models";
import { cn } from "@/lib/utils";

type ChatComposerVariant = "hero" | "panel";

interface ChatComposerProps {
  currentModel: ChatModelOption;
  effectiveRagMode: RagMode;
  input: string;
  onInputChange: (value: string) => void;
  onRagModeChange: (mode: RagMode) => void;
  onSelectedModelChange: (modelId: string) => void;
  onStop: () => void;
  onSubmit: () => void;
  selectedModel: string;
  status: ChatStatus;
  variant?: ChatComposerVariant;
}

const composerClasses: Record<ChatComposerVariant, string> = {
  hero: cn(
    "rounded-[20px] border-0 bg-popover backdrop-blur-xl",
    "claude-input-shadow"
  ),
  panel: cn(
    "rounded-[20px] border-0 bg-popover backdrop-blur-xl",
    "claude-input-shadow"
  ),
};

const textareaClasses: Record<ChatComposerVariant, string> = {
  hero: "!min-h-[52px] !py-3.5 !px-4 !text-[15px] placeholder:text-muted-foreground/35",
  panel: "!min-h-[48px] !py-3 !px-4 !text-[15px] placeholder:text-muted-foreground/35",
};

export const ChatComposer = memo(function ChatComposer({
  currentModel,
  effectiveRagMode,
  input,
  onInputChange,
  onRagModeChange,
  onSelectedModelChange,
  onStop,
  onSubmit,
  selectedModel,
  status,
  variant = "panel",
}: ChatComposerProps) {
  return (
    <PromptInput className={composerClasses[variant]} onSubmit={onSubmit}>
      <PromptInputBody>
        <PromptInputTextarea
          className={textareaClasses[variant]}
          onChange={(event) => onInputChange(event.target.value)}
          placeholder="Pergunte sobre partidas, armas, jogadores..."
          value={input}
        />
      </PromptInputBody>

      <PromptInputFooter className="justify-between px-3 pb-2.5">
        <div className="flex items-center gap-1.5">
          <Select onValueChange={onSelectedModelChange} value={selectedModel}>
            <SelectTrigger
              className="h-7 gap-1 rounded-lg border-0 bg-transparent px-2 text-xs text-muted-foreground hover:bg-accent hover:text-foreground"
              size="sm"
            >
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {CHAT_MODELS.map((model) => (
                <SelectItem key={model.id} value={model.id}>
                  <span className="flex items-center gap-1.5">
                    {model.name}
                    {model.supportsReasoning && (
                      <BrainCircuit className="size-3 text-primary/60" />
                    )}
                  </span>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <div
            className={cn(
              "flex items-center rounded-lg",
              !currentModel.supportsTools && "opacity-40"
            )}
          >
            {RAG_MODES.map((mode) => {
              const isActive = effectiveRagMode === mode;

              return (
                <Tooltip key={mode}>
                  <TooltipTrigger asChild>
                    <button
                      className={cn(
                        "flex items-center gap-1 px-2 py-1 text-xs font-medium transition-colors first:rounded-l-[7px] last:rounded-r-[7px]",
                        "disabled:cursor-not-allowed",
                        isActive
                          ? mode === "off"
                            ? "bg-destructive/15 text-destructive"
                            : "bg-primary/15 text-primary"
                          : "text-muted-foreground hover:text-foreground"
                      )}
                      disabled={!currentModel.supportsTools}
                      onClick={() => onRagModeChange(mode)}
                      type="button"
                    >
                      {mode === "always" && <Search className="size-3" />}
                      {mode === "off" && <SearchX className="size-3" />}
                      {RAG_MODE_CONFIG[mode].label}
                    </button>
                  </TooltipTrigger>
                  <TooltipContent className="text-xs" side="top">
                    {!currentModel.supportsTools
                      ? "Modelo não suporta busca"
                      : RAG_MODE_CONFIG[mode].tip}
                  </TooltipContent>
                </Tooltip>
              );
            })}
          </div>
        </div>

        <PromptInputSubmit onStop={onStop} status={status} />
      </PromptInputFooter>
    </PromptInput>
  );
});
