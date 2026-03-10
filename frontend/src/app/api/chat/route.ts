import {
  convertToModelMessages,
  streamText,
  tool,
  UIMessage,
  stepCountIs,
} from "ai";
import { createOllama } from "ollama-ai-provider-v2";
import { z } from "zod";

const RAG_API_URL = process.env.RAG_API_URL ?? "http://localhost:8000";
const EMBEDDING_RUN_ID = process.env.EMBEDDING_RUN_ID ?? "20260306T025119Z";
const OLLAMA_BASE_URL = process.env.OLLAMA_BASE_URL ?? "http://localhost:11434";
const OLLAMA_MODEL =
  process.env.OLLAMA_MODEL ?? "qwen2.5:7b-instruct-q4_K_M";

const ollama = createOllama({ baseURL: `${OLLAMA_BASE_URL}/api` });

const SYSTEM_PROMPT = `Você é um analista profissional de partidas de CS:GO / Counter-Strike.

IDIOMA: Responda SEMPRE em Português Brasileiro. Nunca mude para inglês, mesmo que os dados retornados estejam em inglês. Traduza tudo.

REGRA PRINCIPAL: SEMPRE use a ferramenta de busca antes de responder qualquer pergunta relacionada a CS:GO. Isso inclui perguntas sobre mapas, armas, jogadores, economia, estratégias, partidas, rounds, dano, kills — QUALQUER tema de CS:GO. Nunca responda com conhecimento geral. Sempre busque primeiro.

COMPORTAMENTO:
- NÃO mencione a ferramenta, não mostre JSON, não explique como a busca funciona.
- NÃO diga "vou buscar" ou "deixe-me verificar". Apenas busque silenciosamente e apresente os resultados.
- Baseie suas respostas APENAS nos dados retornados pela busca. Nunca invente estatísticas ou use conhecimento próprio.
- Se a busca não retornar resultados, diga brevemente que não há dados disponíveis e sugira uma pergunta alternativa.
- Seja direto, cite números específicos, e organize as informações de forma clara.
- A ÚNICA exceção para não usar a ferramenta é se o usuário fizer uma saudação casual (ex: "oi", "olá") ou pergunta que não tem relação com CS:GO.`;

export async function POST(req: Request) {
  const { messages }: { messages: UIMessage[] } = await req.json();

  const result = streamText({
    model: ollama(OLLAMA_MODEL),
    system: SYSTEM_PROMPT,
    messages: await convertToModelMessages(messages),
    tools: {
      searchMatchData: tool({
        description:
          "Search the CS:GO match event database. Use this to find information about kills, damages, weapon stats, player performance, economy, round outcomes, and any match-related data. Returns relevant match event documents ranked by similarity.",
        inputSchema: z.object({
          query: z
            .string()
            .describe(
              "The search query describing what match data to find. Be specific - e.g. 'AK-47 headshot kills on dust2' rather than just 'weapons'."
            ),
          top_k: z
            .number()
            .optional()
            .default(10)
            .describe("Number of results to retrieve (default: 10)"),
          event_type: z
            .string()
            .optional()
            .describe(
              "Optional filter by event type (e.g. 'kill', 'damage', 'round_end')"
            ),
          map_name: z
            .string()
            .optional()
            .describe(
              "Optional filter by map name (e.g. 'de_dust2', 'de_mirage')"
            ),
        }),
        execute: async ({ query, top_k, event_type, map_name }) => {
          const body: Record<string, unknown> = {
            query,
            embedding_run_id: EMBEDDING_RUN_ID,
            top_k: top_k ?? 10,
          };
          if (event_type) body.event_type = event_type;
          if (map_name) body.map_name = map_name;

          const resp = await fetch(`${RAG_API_URL}/search`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
          });

          if (!resp.ok) {
            return { error: `Search API returned ${resp.status}`, results: [] };
          }

          const data = await resp.json();
          return {
            results: data.results ?? [],
            results_returned: data.results_returned ?? (data.results?.length ?? 0),
            retrieval_ms: data.retrieval_ms ?? 0,
            _instruction: `IMPORTANTE: Responda em Português Brasileiro. Os dados acima podem estar em inglês, mas sua resposta DEVE ser em português. Apresente os resultados diretamente, sem mencionar a ferramenta de busca.`,
          };
        },
      }),
    },
    stopWhen: stepCountIs(3),
  });

  return result.toUIMessageStreamResponse();
}
