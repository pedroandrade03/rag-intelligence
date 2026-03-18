# Frontend

Interface web do projeto `rag-intelligence`, construída com Next.js 16, React 19 e AI SDK.

## Objetivo

Fornecer uma UI de chat para consultar dados de partidas de CS:GO, com:

- streaming de respostas
- histórico local de sessões
- seleção de modelo
- modos de uso do RAG (`auto`, `always`, `off`)

## Stack

- Next.js 16 + App Router
- React 19
- AI SDK: `ai` + `@ai-sdk/react`
- Ollama via `ollama-ai-provider-v2`
- SQLite local via `better-sqlite3`
- Tailwind CSS v4 + componentes UI locais
- `react-grab` em desenvolvimento para debug visual

## Arquitetura

### Server/client split

- `src/app/page.tsx` é um Server Component dinâmico.
- O bootstrap inicial das sessões e mensagens vem do servidor, não do cliente.
- `src/components/chat/chat-app.tsx` é a ilha cliente principal e usa `useChat` como fonte de verdade do chat ativo.

### Persistência local

- O banco fica em `frontend/data/chat.db`.
- `src/lib/db.ts` cria o schema automaticamente.
- `src/lib/chat-session-store.ts` concentra leitura e escrita de sessões e mensagens.

### Rotas internas

- `src/app/api/chat/route.ts`
  Faz o bridge entre a UI e o modelo. Persiste a mensagem do usuário, chama `streamText(...)` com Ollama e salva a resposta final do assistente.
- `src/app/api/sessions/route.ts`
  Lista e cria sessões.
- `src/app/api/sessions/[id]/route.ts`
  Renomeia e remove sessões.
- `src/app/api/sessions/[id]/messages/route.ts`
  Carrega as mensagens de uma sessão.

### Fluxo de resposta

1. A página carrega com a sessão mais recente já renderizada pelo servidor.
2. O usuário envia a mensagem pela UI.
3. A rota `/api/chat` decide modelo e modo RAG.
4. Quando o RAG está habilitado, a tool `searchMatchData` consulta `POST {RAG_API_URL}/search`.
5. O Ollama gera a resposta e o AI SDK streama os chunks para o cliente.
6. A resposta final é persistida no SQLite.

## Variáveis de ambiente

- `RAG_API_URL`
  URL da API FastAPI usada pela busca semântica.
  Default: `http://localhost:8000`
- `EMBEDDING_RUN_ID`
  Run de embeddings consultado pelo endpoint de busca.
- `OLLAMA_BASE_URL`
  URL base do Ollama.
  Default: `http://localhost:11434`
- `OLLAMA_MODEL`
  Modelo default do chat quando nenhum modelo é escolhido pelo usuário.

## Desenvolvimento

Instalar dependências:

```bash
npm install
```

Rodar em modo desenvolvimento:

```bash
npm run dev
```

Build de produção:

```bash
npm run build
```

Subir versão buildada:

```bash
npm run start
```

Lint:

```bash
npm run lint
```

## Comandos via Makefile

Da raiz do repositório:

```bash
make frontend
make frontend-build
make ci-frontend
make db-reset
```

## Observações

- O frontend não usa mais TanStack Query para o fluxo de chat. O estado vivo do chat fica no `useChat`.
- Componentes mais pesados do chat são carregados com `next/dynamic` para reduzir o custo inicial do bundle.
- `react-grab` é carregado apenas em `development` por `src/app/dev-tools.tsx`.
