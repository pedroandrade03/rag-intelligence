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

REGRA PRINCIPAL: SEMPRE use a ferramenta de busca antes de responder. Você tem acesso a:
- Documentação do pipeline (busca semântica): como funcionam as camadas Bronze, Silver, Gold, ML Training e a arquitetura geral.
- Resultados de treinamento ML (busca lexical): métricas de modelos (ROC-AUC, F1, etc.), feature importances, segmentos por mapa/metade.

COMPORTAMENTO:
- NÃO mencione a ferramenta, não mostre JSON, não explique como a busca funciona.
- NÃO diga "vou buscar" ou "deixe-me verificar". Apenas busque silenciosamente e apresente os resultados.
- Baseie suas respostas APENAS nos dados retornados pela busca. Nunca invente estatísticas ou use conhecimento próprio.
- Se a busca não retornar resultados, diga brevemente que não há dados disponíveis e sugira uma pergunta alternativa.
- Seja direto, cite números específicos, e organize as informações de forma clara.
- A ÚNICA exceção para não usar a ferramenta é se o usuário fizer uma saudação casual (ex: "oi", "olá") ou pergunta sem relação com o pipeline.

ESTRATÉGIA DE BUSCA:
- Para perguntas sobre o pipeline (Bronze, Silver, Gold, arquitetura): use include_semantic=true.
- Para perguntas sobre métricas de ML, modelos, feature importances: use include_lexical=true.
- Para perguntas sobre modelos específicos: use model_filter (ex: "logistic_regression", "hist_gradient_boosting").
- Na dúvida, deixe ambos habilitados (padrão).
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
        "The search query. Be specific - e.g. 'what does the Gold phase do' or 'logistic regression ROC-AUC'."
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
        "Include semantic search over pipeline documentation (default: true)."
      ),
    include_lexical: z
      .boolean()
      .optional()
      .default(true)
      .describe(
        "Include lexical search over ML training results (default: true)."
      ),
    model_filter: z
      .string()
      .optional()
      .describe(
        "Optional filter by ML model name (e.g. 'logistic_regression', 'hist_gradient_boosting')."
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

    return {
      semantic_results: semanticResults,
      lexical_results: lexicalResults,
      results_returned: totalCount,
      retrieval_ms: data.retrieval_ms ?? 0,
      _instruction: `IMPORTANTE: Responda em Português Brasileiro. Os dados acima podem estar em inglês, mas sua resposta DEVE ser em português. Apresente os resultados diretamente, sem mencionar a ferramenta de busca.`,
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
  const chatProvider = getChatProvider();
  const selectedModel = getChatModel(models, model ?? defaultModel);
  const modelId = selectedModel.id;
  const mode = ragMode ?? "auto";
  const canUseTools = selectedModel.supportsTools;
  const effectiveMode = canUseTools ? mode : "off";

  const tools =
    effectiveMode === "off"
      ? undefined
      : { searchKnowledgeBase: searchKnowledgeBaseTool };
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

  const result = streamText({
    model: chatProvider.provider(modelId),
    system: effectiveMode === "off" ? SYSTEM_PROMPT_NO_TOOLS : SYSTEM_PROMPT,
    messages: await convertToModelMessages(messages),
    tools,
    toolChoice,
    ...(chatProvider.config.kind === "ollama" && selectedModel.supportsReasoning && {
      providerOptions: { ollama: { think: true } },
    }),
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
