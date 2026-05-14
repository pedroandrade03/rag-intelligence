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
- searchKnowledgeBase: documentação do pipeline (busca semântica) e resultados de treinamento ML (busca lexical).
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
- Para perguntas sobre o pipeline (Bronze, Silver, Gold, arquitetura): use searchKnowledgeBase com include_semantic=true.
- Para perguntas sobre métricas de ML, modelos, feature importances: use searchKnowledgeBase com include_lexical=true.
- Para perguntas sobre modelos específicos: use model_filter (ex: "logistic_regression", "hist_gradient_boosting").
- Na dúvida, use searchKnowledgeBase com ambos habilitados (padrão).
- Mantenha top_k entre 3 e 5. Menos resultados = respostas melhores.`;

const SYSTEM_PROMPT_NO_TOOLS = `Você é um analista especializado no pipeline de dados e modelos de ML do RAG Intelligence.

IDIOMA: Responda SEMPRE em Português Brasileiro.

COMPORTAMENTO:
- Responda com base no seu conhecimento geral sobre pipelines de dados, RAG e ML.
- Seja direto, cite números quando possível, e organize as informações de forma clara.
- A busca no banco de dados foi desativada pelo usuário. Use apenas seu conhecimento.`;

const searchKnowledgeBaseTool = tool({
  description:
    "Search the knowledge base. Retrieves pipeline documentation (semantic search) and ML training results (lexical search). Always search before answering questions about the pipeline, architecture, or model performance.",
  inputSchema: z.object({
    query: z
      .string()
      .describe(
        "The search query. Be specific - e.g. 'what does the Gold phase do' or 'logistic regression ROC-AUC'.",
      ),
    top_k: z
      .number()
      .optional()
      .default(5)
      .describe("Number of results to retrieve (default: 5)."),
    include_semantic: z
      .boolean()
      .optional()
      .default(true)
      .describe(
        "Include semantic search over pipeline documentation (default: true).",
      ),
    include_lexical: z
      .boolean()
      .optional()
      .default(true)
      .describe(
        "Include lexical search over ML training results (default: true).",
      ),
    model_filter: z
      .string()
      .optional()
      .describe(
        "Optional filter by ML model name (e.g. 'logistic_regression', 'hist_gradient_boosting').",
      ),
  }),
  execute: async ({
    query,
    top_k,
    include_semantic,
    include_lexical,
    model_filter,
  }) => {
    const body: Record<string, unknown> = {
      query,
      embedding_run_id: "pipeline-docs",
      top_k: top_k ?? 5,
      include_semantic: include_semantic ?? true,
      include_lexical: include_lexical ?? true,
    };
    if (model_filter) body.model_filter = model_filter;

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
      };
    }

    const data = await resp.json();
    const semanticResults = data.semantic_results ?? [];
    const lexicalResults = data.lexical_results ?? [];
    const totalCount = semanticResults.length + lexicalResults.length;
    const answerContext = [
      ...semanticResults.slice(0, 5).map(
        (result: {
          text?: string;
          source_file?: string;
          metadata?: { pipeline_phase?: string; header_path?: string };
        }) => ({
          type: "pipeline_doc",
          phase: result.metadata?.pipeline_phase ?? null,
          source: result.source_file ?? null,
          header: result.metadata?.header_path ?? null,
          text: result.text ?? "",
        }),
      ),
      ...lexicalResults.slice(0, 5).map(
        (result: {
          model_name?: string;
          roc_auc?: number;
          f1?: number;
          balanced_accuracy?: number;
          text_summary?: string;
        }) => ({
          type: "ml_training",
          model_name: result.model_name ?? null,
          roc_auc: result.roc_auc ?? null,
          f1: result.f1 ?? null,
          balanced_accuracy: result.balanced_accuracy ?? null,
          text: result.text_summary ?? "",
        }),
      ),
    ];

    return {
      answer_context: answerContext,
      semantic_results: semanticResults,
      lexical_results: lexicalResults,
      results_returned: totalCount,
      retrieval_ms: data.retrieval_ms ?? 0,
      _instruction: `IMPORTANTE: Responda em Português Brasileiro. Os dados acima podem estar em inglês, mas sua resposta DEVE ser em português. Apresente os resultados diretamente, sem mencionar a ferramenta de busca.`,
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
    return {
      latest_training_run: latestTraining,
      results_returned: latestTraining?.count ?? 0,
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
          searchKnowledgeBase: searchKnowledgeBaseTool,
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
