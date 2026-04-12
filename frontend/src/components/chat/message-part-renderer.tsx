"use client";

import type { ChatStatus, UIMessage } from "ai";

import { Bot, BrainCircuit, ChevronDown, Database, Search } from "lucide-react";

import {
  MessageResponse,
} from "@/components/ai-elements/message";
import {
  Reasoning,
  ReasoningContent,
  ReasoningTrigger,
} from "@/components/ai-elements/reasoning";
import {
  Sources,
  SourcesContent,
  SourcesTrigger,
} from "@/components/ai-elements/sources";
import { Badge } from "@/components/ui/badge";

type MessagePart = UIMessage["parts"][number];

interface SearchToolInput {
  query?: string;
  include_semantic?: boolean;
  include_lexical?: boolean;
  model_filter?: string;
}

interface SemanticResult {
  event_type: string | null;
  map: string | null;
  rank: number;
  round: number | string | null;
  score: number | null;
  text: string;
}

interface LexicalResult {
  rank: number;
  score: number;
  model_name: string;
  roc_auc: number | null;
  f1: number | null;
  balanced_accuracy: number | null;
  log_loss_val: number | null;
  brier: number | null;
  feature_importances: Record<string, number> | null;
  text_summary: string;
}

interface SearchToolOutput {
  semantic_results?: SemanticResult[];
  lexical_results?: LexicalResult[];
  results_returned?: number;
  retrieval_ms?: number;
}

function renderThinkingMessage(streaming: boolean, duration?: number) {
  if (streaming || duration === 0) {
    return (
      <span className="flex items-center gap-1.5 text-primary/70">
        <BrainCircuit className="size-3.5 animate-pulse" />
        Pensando...
      </span>
    );
  }

  if (duration === undefined) {
    return <span>Pensou por alguns segundos</span>;
  }

  return <span>Pensou por {duration}s</span>;
}

function renderSearchToolState(
  key: string,
  part: MessagePart
) {
  if (part.type !== "tool-searchKnowledgeBase") {
    return null;
  }

  if (part.state === "input-streaming") {
    return (
      <div
        className="flex items-center gap-2 rounded-xl border border-border/40 bg-muted/30 px-3.5 py-2.5 text-sm text-muted-foreground"
        key={key}
      >
        <Search className="size-4 animate-pulse text-primary/60" />
        <span>Preparando busca...</span>
      </div>
    );
  }

  if (part.state === "input-available") {
    const toolInput = part.input as SearchToolInput;

    return (
      <div
        className="flex items-center gap-2 rounded-xl border border-border/40 bg-muted/30 px-3.5 py-2.5 text-sm text-muted-foreground"
        key={key}
      >
        <Search className="size-4 animate-pulse text-primary/60" />
        <div className="flex flex-col gap-0.5">
          <span>Buscando na base de conhecimento...</span>
          {toolInput.query && (
            <span className="max-w-md truncate text-xs italic text-muted-foreground/60">
              &quot;{toolInput.query}&quot;
              {toolInput.model_filter && ` [${toolInput.model_filter}]`}
            </span>
          )}
        </div>
      </div>
    );
  }

  if (part.state !== "output-available") {
    return null;
  }

  const output = part.output as SearchToolOutput;
  const semanticResults = output.semantic_results ?? [];
  const lexicalResults = output.lexical_results ?? [];
  const count = output.results_returned ?? (semanticResults.length + lexicalResults.length);

  if (count === 0) {
    return (
      <div
        className="flex items-center gap-2 rounded-xl border border-border/40 bg-muted/30 px-3.5 py-2.5 text-sm text-muted-foreground"
        key={key}
      >
        <Database className="size-4 text-muted-foreground/60" />
        <span>Nenhum resultado encontrado</span>
      </div>
    );
  }

  return (
    <Sources className="!text-sm" key={key}>
      <SourcesTrigger count={count}>
        <div className="flex cursor-pointer items-center gap-2 rounded-xl border border-border/40 bg-muted/30 px-3.5 py-2.5 text-sm text-muted-foreground transition-colors hover:bg-muted/50">
          <Database className="size-4 text-primary/60" />
          <span className="font-medium">
            {count} resultados recuperados
            {output.retrieval_ms ? ` em ${output.retrieval_ms}ms` : ""}
          </span>
          <ChevronDown className="ml-auto size-4 transition-transform [[data-state=open]_&]:rotate-180" />
        </div>
      </SourcesTrigger>
      <SourcesContent className="w-full max-w-full">
        <div className="mt-2 space-y-3">
          {semanticResults.map((result: SemanticResult, index: number) => (
            <div
              className="space-y-2.5 rounded-xl border border-border/40 bg-card/60 p-4"
              key={`${key}-sem-${index}`}
            >
              <div className="flex flex-wrap items-center gap-2">
                <Badge className="px-2 py-0.5 text-xs" variant="secondary">
                  pipeline doc
                </Badge>
                {result.score != null && (
                  <span className="ml-auto text-xs text-muted-foreground/60">
                    {(result.score * 100).toFixed(1)}% relevancia
                  </span>
                )}
              </div>
              <p className="line-clamp-3 text-sm leading-relaxed text-muted-foreground">
                {result.text}
              </p>
            </div>
          ))}
          {lexicalResults.map((result: LexicalResult, index: number) => (
            <div
              className="space-y-2.5 rounded-xl border border-border/40 bg-card/60 p-4"
              key={`${key}-lex-${index}`}
            >
              <div className="flex flex-wrap items-center gap-2">
                <Badge className="px-2 py-0.5 text-xs" variant="secondary">
                  ML training
                </Badge>
                <Badge className="px-2 py-0.5 text-xs" variant="outline">
                  {result.model_name}
                </Badge>
                {result.score != null && (
                  <span className="ml-auto text-xs text-muted-foreground/60">
                    score {result.score.toFixed(3)}
                  </span>
                )}
              </div>
              <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-muted-foreground sm:grid-cols-3">
                {result.roc_auc != null && (
                  <span>ROC-AUC: <strong>{result.roc_auc.toFixed(4)}</strong></span>
                )}
                {result.f1 != null && (
                  <span>F1: <strong>{result.f1.toFixed(4)}</strong></span>
                )}
                {result.balanced_accuracy != null && (
                  <span>Bal. Acc: <strong>{result.balanced_accuracy.toFixed(4)}</strong></span>
                )}
                {result.log_loss_val != null && (
                  <span>Log Loss: <strong>{result.log_loss_val.toFixed(4)}</strong></span>
                )}
                {result.brier != null && (
                  <span>Brier: <strong>{result.brier.toFixed(4)}</strong></span>
                )}
              </div>
            </div>
          ))}
        </div>
      </SourcesContent>
    </Sources>
  );
}

export function sortMessageParts(parts: UIMessage["parts"]) {
  return parts.toSorted((left, right) => {
    if (left.type === "reasoning" && right.type !== "reasoning") {
      return -1;
    }

    if (left.type !== "reasoning" && right.type === "reasoning") {
      return 1;
    }

    return 0;
  });
}

interface MessagePartRendererProps {
  index: number;
  message: UIMessage;
  part: MessagePart;
  status: ChatStatus;
}

export function MessagePartRenderer({
  index,
  message,
  part,
  status,
}: MessagePartRendererProps) {
  const key = `${message.id}-${index}`;

  if (part.type === "reasoning") {
    const isReasoningDone = message.parts.some(
      (messagePart, partIndex) => partIndex > index && messagePart.type === "text"
    );
    const isActivelyStreaming =
      status === "streaming" &&
      message.role === "assistant" &&
      !isReasoningDone;

    return (
      <Reasoning isStreaming={isActivelyStreaming} key={key}>
        <ReasoningTrigger getThinkingMessage={renderThinkingMessage} />
        <ReasoningContent>{part.text}</ReasoningContent>
      </Reasoning>
    );
  }

  if (part.type === "text") {
    if (message.role === "user") {
      return <p key={key}>{part.text}</p>;
    }

    const isStreamingAssistantText =
      status === "streaming" && message.role === "assistant";

    return (
      <MessageResponse
        animated={{ animation: "blurIn", duration: 220, easing: "ease-out" }}
        isAnimating={isStreamingAssistantText}
        key={key}
      >
        {part.text}
      </MessageResponse>
    );
  }

  return renderSearchToolState(key, part);
}

export function ThinkingIndicator({
  supportsReasoning,
}: {
  supportsReasoning: boolean;
}) {
  return (
    <div className="flex gap-3">
      <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-muted/80 text-muted-foreground">
        <Bot className="size-3.5" />
      </div>
      <div className="flex items-center gap-1.5 py-3">
        <span className="size-1.5 animate-bounce rounded-full bg-primary/50 [animation-delay:0ms]" />
        <span className="size-1.5 animate-bounce rounded-full bg-primary/50 [animation-delay:150ms]" />
        <span className="size-1.5 animate-bounce rounded-full bg-primary/50 [animation-delay:300ms]" />
        <span className="ml-2 text-sm text-muted-foreground/60">
          {supportsReasoning ? "Iniciando raciocínio..." : "Analisando dados..."}
        </span>
      </div>
    </div>
  );
}
