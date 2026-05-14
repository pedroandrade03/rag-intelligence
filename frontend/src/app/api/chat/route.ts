import {
  convertToModelMessages,
  streamText,
  type UIMessage,
  tool,
  stepCountIs,
} from "ai";
import { z } from "zod";

import {
  getChatModel,
  getChatRuntimeConfig,
  type RagMode,
} from "@/lib/chat-models";
import { getChatProvider } from "@/lib/chat-provider";
import { upsertStoredSessionMessage } from "@/lib/chat-session-store";

const RAG_API_URL = process.env.RAG_API_URL ?? "http://localhost:8000";

const SYSTEM_PROMPT = `Você é um analista especializado no pipeline de dados e modelos de ML do RAG Intelligence — uma plataforma de analytics de CS:GO.

IDIOMA: Responda SEMPRE em Português Brasileiro. Nunca mude para inglês, mesmo que os dados retornados estejam em inglês. Traduza tudo.

REGRA PRINCIPAL: SEMPRE use uma ferramenta antes de responder perguntas sobre o projeto. Você tem acesso a:
- searchPipelineDocs: documentação do pipeline, com filtro real por etapa Bronze/Silver/Gold/ML/arquitetura.
- searchTrainingMetrics: resultados de treinamento ML, métricas, feature importances e comparação de modelos.
- getLatestTrainingRun: último treinamento executado, com run_id, data e métricas dos modelos.

COMPORTAMENTO:
- NÃO mencione a ferramenta, não mostre JSON, não explique como a busca funciona.
- NÃO diga "vou buscar" ou "deixe-me verificar". Apenas busque silenciosamente e apresente os resultados.
- Baseie suas respostas APENAS nos dados retornados pela busca. Nunca invente estatísticas ou use conhecimento próprio.
- Quando o resultado da ferramenta trouxer answer_context, use esse contexto compacto como fonte principal da resposta.
- Só diga que não há dados se todos os resultados retornados estiverem vazios.
- Seja direto, cite números específicos, e organize as informações de forma clara.
- A ÚNICA exceção para não usar a ferramenta é se o usuário fizer uma saudação casual (ex: "oi", "olá") ou pergunta sem relação com o pipeline.

ESTRATÉGIA DE FERRAMENTAS:
- Para "último treinamento", "treinamento mais recente" ou perguntas equivalentes: use getLatestTrainingRun.
- Para perguntas sobre o pipeline (Bronze, Silver, Gold, arquitetura): use searchPipelineDocs. Se a pergunta mencionar uma etapa, preencha pipeline_phase.
- Para perguntas sobre métricas de ML, modelos, feature importances: use searchTrainingMetrics.
- Para perguntas sobre modelos específicos: use model_filter (ex: "logistic_regression", "hist_gradient_boosting").
- Na dúvida sobre documentação, use searchPipelineDocs; na dúvida sobre desempenho de modelos, use searchTrainingMetrics.
- Mantenha top_k entre 3 e 5. Menos resultados = respostas melhores.`;

const SYSTEM_PROMPT_NO_TOOLS = `Você é um analista especializado no pipeline de dados e modelos de ML do RAG Intelligence.

IDIOMA: Responda SEMPRE em Português Brasileiro.

COMPORTAMENTO:
- Responda com base no seu conhecimento geral sobre pipelines de dados, RAG e ML.
- Seja direto, cite números quando possível, e organize as informações de forma clara.
- A busca no banco de dados foi desativada pelo usuário. Use apenas seu conhecimento.`;

type SemanticToolResult = {
  rank?: number;
  score?: number | null;
  text?: string;
  source_file?: string | null;
  metadata?: { pipeline_phase?: string | null; header_path?: string | null };
};

type LexicalToolResult = {
  rank?: number;
  score?: number | null;
  run_id?: string | null;
  created_at?: string | null;
  model_name?: string | null;
  roc_auc?: number | null;
  f1?: number | null;
  balanced_accuracy?: number | null;
  log_loss_val?: number | null;
  brier?: number | null;
  text_summary?: string;
};

function truncateText(value: string | undefined, maxLength = 700): string {
  if (!value) {
    return "";
  }
  return value.length > maxLength ? `${value.slice(0, maxLength).trimEnd()}...` : value;
}

function compactSemanticResults(results: SemanticToolResult[], limit = 3) {
  return results.slice(0, limit).map((result, index) => ({
    rank: result.rank ?? index + 1,
    score: result.score ?? null,
    text: truncateText(result.text),
    source_file: result.source_file ?? null,
    metadata: {
      pipeline_phase: result.metadata?.pipeline_phase ?? null,
      header_path: result.metadata?.header_path ?? null,
    },
  }));
}

function compactLexicalResults(results: LexicalToolResult[], limit = 3) {
  return results.slice(0, limit).map((result, index) => ({
    rank: result.rank ?? index + 1,
    score: result.score ?? null,
    run_id: result.run_id ?? null,
    created_at: result.created_at ?? null,
    model_name: result.model_name ?? null,
    roc_auc: result.roc_auc ?? null,
    f1: result.f1 ?? null,
    balanced_accuracy: result.balanced_accuracy ?? null,
    log_loss_val: result.log_loss_val ?? null,
    brier: result.brier ?? null,
    text_summary: truncateText(result.text_summary, 500),
  }));
}

async function runHybridSearch(body: Record<string, unknown>) {
  const resp = await fetch(`${RAG_API_URL}/search/hybrid`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!resp.ok) {
    return {
      error: `Search API returned ${resp.status}`,
      semantic_results: [],
      lexical_results: [],
      results_returned: 0,
      retrieval_ms: 0,
    };
  }

  const data = await resp.json();
  const semanticResults = data.semantic_results ?? [];
  const lexicalResults = data.lexical_results ?? [];
  return {
    ...data,
    semantic_results: semanticResults,
    lexical_results: lexicalResults,
    results_returned: semanticResults.length + lexicalResults.length,
  };
}

const searchPipelineDocsTool = tool({
  description:
    "Search only the pipeline documentation using semantic retrieval. Use this for questions about Bronze, Silver, Gold, ML training pipeline, architecture, RAG, chunking, pgvector, or system design.",
  inputSchema: z.object({
    query: z.string().describe("The documentation question to search for."),
    top_k: z.number().optional().default(5).describe("Number of doc chunks to retrieve."),
    pipeline_phase: z
      .enum(["bronze", "silver", "gold", "ml-training", "architecture"])
      .optional()
      .describe("Optional exact pipeline phase filter when the question names a phase."),
  }),
  execute: async ({ query, top_k, pipeline_phase }) => {
    const data = await runHybridSearch({
      query,
      embedding_run_id: "pipeline-docs",
      top_k: top_k ?? 5,
      include_semantic: true,
      include_lexical: false,
      pipeline_phase,
    });
    const semanticResults = compactSemanticResults(data.semantic_results ?? []);

    return {
      answer_context: semanticResults.map((result: SemanticToolResult) => ({
        type: "pipeline_doc",
        phase: result.metadata?.pipeline_phase ?? null,
        source: result.source_file ?? null,
        header: result.metadata?.header_path ?? null,
        text: result.text ?? "",
      })),
      semantic_results: semanticResults,
      lexical_results: [],
      results_returned: semanticResults.length,
      retrieval_ms: data.retrieval_ms ?? 0,
      _instruction:
        "IMPORTANTE: Responda em Português Brasileiro usando os trechos de documentação retornados. Se pipeline_phase foi usado, os resultados já estão filtrados para essa etapa.",
    };
  },
});

const searchTrainingMetricsTool = tool({
  description:
    "Search only ML training metadata and metrics using PostgreSQL full-text search. Use this for ROC-AUC, F1, best model, feature importances, model comparison, or training performance questions.",
  inputSchema: z.object({
    query: z.string().describe("The ML metric/performance question to search for."),
    top_k: z.number().optional().default(5).describe("Number of training rows to retrieve."),
    model_filter: z
      .string()
      .optional()
      .describe("Optional model_name filter, e.g. logistic_regression or hist_gradient_boosting."),
  }),
  execute: async ({ query, top_k, model_filter }) => {
    const data = await runHybridSearch({
      query,
      embedding_run_id: "pipeline-docs",
      top_k: top_k ?? 5,
      include_semantic: false,
      include_lexical: true,
      model_filter,
    });
    const lexicalResults = compactLexicalResults(data.lexical_results ?? []);

    return {
      answer_context: lexicalResults.map((result: LexicalToolResult) => ({
        type: "ml_training",
        model_name: result.model_name ?? null,
        roc_auc: result.roc_auc ?? null,
        f1: result.f1 ?? null,
        balanced_accuracy: result.balanced_accuracy ?? null,
        text: result.text_summary ?? "",
      })),
      semantic_results: [],
      lexical_results: lexicalResults,
      results_returned: lexicalResults.length,
      retrieval_ms: data.retrieval_ms ?? 0,
      _instruction:
        "IMPORTANTE: Responda em Português Brasileiro usando somente as métricas retornadas.",
    };
  },
});

const getLatestTrainingRunTool = tool({
  description:
    "Return the latest ML training run from structured metadata. Use for questions like 'qual foi o último treinamento?', 'latest training run', or 'treinamento mais recente'.",
  inputSchema: z.object({
    model_filter: z
      .string()
      .optional()
      .describe(
        "Optional model_name filter, e.g. logistic_regression or hist_gradient_boosting.",
      ),
  }),
  execute: async ({ model_filter }) => {
    const params = new URLSearchParams({ latest: "true" });
    if (model_filter) params.set("model_filter", model_filter);

    const resp = await fetch(`${RAG_API_URL}/metadata/training?${params}`);
    if (!resp.ok) {
      return {
        error: `Training metadata API returned ${resp.status}`,
        latest_training_run: { run_id: null, created_at: null, models: [], count: 0 },
        results_returned: 0,
      };
    }

    const latestTraining = await resp.json();
    const models = compactLexicalResults(latestTraining?.models ?? [], 5);
    return {
      latest_training_run: {
        run_id: latestTraining?.run_id ?? null,
        created_at: latestTraining?.created_at ?? null,
        models,
        count: models.length,
      },
      results_returned: models.length,
      _instruction:
        "IMPORTANTE: Responda em Português Brasileiro. Apresente run_id, data e métricas dos modelos de forma direta, sem mencionar a ferramenta.",
    };
  },
});

export async function POST(req: Request) {
  const {
    messages,
    model,
    ragMode,
    sessionId,
    trigger,
  }: {
    messages: UIMessage[];
    model?: string;
    ragMode?: RagMode;
    sessionId?: string;
    trigger?:
      | "submit-message"
      | "regenerate-message"
      | "submit-user-message"
      | "regenerate-assistant-message";
  } = await req.json();

  const { defaultModelId, models } = getChatRuntimeConfig();
  const defaultModel = process.env.CHAT_DEFAULT_MODEL ?? defaultModelId;
  const selectedModel = getChatModel(models, model ?? defaultModel);
  const modelId = selectedModel.id;
  const chatProvider = getChatProvider(process.env, modelId);
  const mode = ragMode ?? "auto";
  const canUseTools = selectedModel.supportsTools;
  const effectiveMode = canUseTools ? mode : "off";

  const tools =
    effectiveMode === "off"
      ? undefined
      : {
          searchPipelineDocs: searchPipelineDocsTool,
          searchTrainingMetrics: searchTrainingMetricsTool,
          getLatestTrainingRun: getLatestTrainingRunTool,
        };
  const toolChoice =
    effectiveMode === "always"
      ? ("required" as const)
      : effectiveMode === "off"
        ? undefined
        : ("auto" as const);

  const isRegenerateTrigger =
    trigger === "regenerate-message" ||
    trigger === "regenerate-assistant-message";

  if (sessionId && !isRegenerateTrigger) {
    const lastMessage = messages.at(-1);

    if (lastMessage?.role === "user") {
      upsertStoredSessionMessage(sessionId, lastMessage);
    }
  }

  const localChatModelId = process.env.LOCAL_CHAT_MODEL?.trim() || "gemma4";
  const maxOutputTokens = Number.parseInt(process.env.CHAT_MAX_OUTPUT_TOKENS ?? "512", 10);

  const result = streamText({
    model: chatProvider.provider(modelId),
    system: effectiveMode === "off" ? SYSTEM_PROMPT_NO_TOOLS : SYSTEM_PROMPT,
    messages: await convertToModelMessages(messages),
    tools,
    toolChoice,
    ...(chatProvider.config.kind === "ollama" &&
      selectedModel.supportsReasoning && {
        providerOptions: { ollama: { think: true } },
      }),
    ...(modelId === localChatModelId && Number.isFinite(maxOutputTokens) && maxOutputTokens > 0
      ? { maxOutputTokens }
      : {}),
    stopWhen: stepCountIs(3),
  });

  return result.toUIMessageStreamResponse({
    generateMessageId: () => crypto.randomUUID(),
    onFinish: async ({ isAborted, responseMessage }) => {
      if (!sessionId || isAborted) {
        return;
      }

      upsertStoredSessionMessage(sessionId, responseMessage);
    },
    originalMessages: messages,
  });
}
