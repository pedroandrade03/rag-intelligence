# RAG Intelligence — Roteiro de Estudo para Apresentação

> Objetivo: folha de estudo rápida para explicar o projeto e responder perguntas do professor sobre arquitetura, RAG, configuração, busca lexical e busca densa/semântica.

---

## 1. Resposta curta: usamos DuckDB na busca lexical?

**Não. O projeto não usa DuckDB para lexical search.**

A busca lexical é feita no **PostgreSQL/TimescaleDB**, usando **Full-Text Search nativo do PostgreSQL**:

- tabela: `training_runs`
- coluna gerada: `search_vector tsvector GENERATED ALWAYS AS (to_tsvector('english', search_text)) STORED`
- índice: `GIN (search_vector)`
- query: `websearch_to_tsquery('english', query)`
- ranking: `ts_rank(search_vector, websearch_to_tsquery(...))`

O DuckDB também não aparece como dependência do backend. O armazenamento principal é:

- **MinIO**: Data Lake Bronze/Silver/Gold.
- **PostgreSQL/TimescaleDB + pgvector**: metadados, full-text search e vetores.
- **MLflow**: tracking de experimentos.
- **SQLite/better-sqlite3**: só no frontend para salvar sessões de chat, não para busca RAG.

Resposta oral sugerida:

> Não usamos DuckDB. Para a busca lexical usamos o Full-Text Search do PostgreSQL em cima da tabela `training_runs`. O PostgreSQL gera um `tsvector` a partir do texto de busca dos resultados de treino, indexa com GIN e rankeia com `ts_rank`. Para busca densa usamos pgvector no mesmo banco PostgreSQL/TimescaleDB.

---

## 2. Pitch de 1 minuto do projeto

O **RAG Intelligence** é uma plataforma de analytics de CS:GO com arquitetura de dados e IA. O dataset vem do Kaggle com eventos de partidas, como dano, kills, granadas e metadados de rounds.

O pipeline segue arquitetura **Medallion**:

1. **Bronze**: baixa e guarda o dataset bruto no MinIO.
2. **Silver**: limpa, normaliza colunas e remove dados inválidos/duplicados.
3. **Gold**: padroniza tudo em um schema único de eventos.
4. **ML Training**: treina modelos para prever o vencedor do próximo round e registra métricas.
5. **RAG/Search**: permite perguntar sobre a arquitetura/pipeline e sobre resultados de modelos.

A aplicação tem:

- backend **FastAPI**;
- frontend **Next.js** com chat;
- **LlamaIndex** para embedding e retrieval semântico;
- **Ollama** como LLM/embedding local;
- **PostgreSQL + pgvector** para busca vetorial;
- **PostgreSQL Full-Text Search** para busca lexical;
- **MLflow** para experiment tracking.

---

## 3. Como o RAG está feito

Existem duas formas relacionadas de RAG/busca no projeto.

### 3.1 Chat principal do frontend

O fluxo mais importante para demonstração é o chat Next.js:

1. Usuário pergunta no chat.
2. A rota `frontend/src/app/api/chat/route.ts` usa AI SDK com tool calling.
3. A ferramenta `searchKnowledgeBase` chama o backend em `POST /search/hybrid`.
4. O backend retorna dois tipos de resultado:
   - `semantic_results`: busca semântica/densa em docs do pipeline.
   - `lexical_results`: busca lexical em métricas de treinamento ML.
5. O LLM do chat sintetiza a resposta em Português Brasileiro usando os resultados retornados.

Resumo oral:

> No chat, o RAG é implementado como tool calling. O modelo é instruído a buscar antes de responder. A tool chama `/search/hybrid`, que combina busca semântica sobre documentação do pipeline com busca lexical sobre resultados de treinamento. Depois o modelo sintetiza uma resposta em português com base nesses dados.

### 3.2 Endpoint backend `/rag/query`

Também existe um endpoint backend puro de RAG:

- arquivo: `src/rag_intelligence/api/routes/rag.py`
- endpoint: `POST /rag/query`
- implementação: `src/rag_intelligence/rag.py`

Esse endpoint monta um `VectorStoreIndex` do LlamaIndex sobre o pgvector e usa `index.as_query_engine(...)` com:

- `similarity_top_k`
- filtros de metadata
- prompt QA customizado
- LLM vindo do `ProviderRegistry`
- modo streaming opcional via SSE

Resumo oral:

> Além do chat com tool calling, o backend tem `/rag/query`, que usa o QueryEngine do LlamaIndex: recupera nós do pgvector e gera uma resposta com LLM, com suporte a streaming SSE.

---

## 4. Busca densa / semântica

### O que ela busca?

Atualmente a busca semântica principal do chat busca a **documentação do pipeline** em `docs/pipeline/*.md`.

O script `embed-docs` lê esses Markdown, divide em chunks com `MarkdownNodeParser`, gera embeddings e salva no pgvector com `embedding_run_id = pipeline-docs`.

### Como indexa?

Arquivo principal: `src/rag_intelligence/embed_docs_cli.py`

Fluxo:

1. Lê arquivos Markdown em `docs/pipeline`.
2. Cria objetos `Document` do LlamaIndex com metadata:
   - `embedding_run_id: pipeline-docs`
   - `document_tier: pipeline_doc`
   - `pipeline_phase`
   - `source_file`
3. Remove embeddings antigos do run `pipeline-docs` para ser idempotente.
4. Usa `MarkdownNodeParser` para chunking.
5. Usa `IngestionPipeline(transformations=[parser, embed_model], vector_store=vector_store)`.
6. Salva os nós/vetores no PostgreSQL + pgvector.

### Como consulta?

Arquivo principal: `src/rag_intelligence/retrieval.py`

Fluxo:

1. Carrega configurações (`AppSettings`).
2. Verifica se a tabela pgvector existe.
3. Obtém o modelo de embedding via `ProviderRegistry`.
4. Cria o `PGVectorStore`.
5. Cria `VectorStoreIndex.from_vector_store(...)`.
6. Executa `index.as_retriever(similarity_top_k=top_k, filters=...)`.
7. Chama `retriever.retrieve(query)`.
8. Retorna resultados com rank, score, texto e metadata.

### Onde os vetores ficam?

Arquivo principal: `src/rag_intelligence/db.py`

O projeto usa `llama_index.vector_stores.postgres.PGVectorStore` com:

- banco: PostgreSQL/TimescaleDB
- tabela base configurável por `PG_TABLE_NAME`, padrão `rag_embeddings`
- tabela real de dados: `data_rag_embeddings`
- dimensão: `PG_EMBED_DIM`, padrão `768`
- metadata em JSONB (`use_jsonb=True`)
- índice HNSW com distância de cosseno:
  - `hnsw_m = 16`
  - `hnsw_ef_construction = 64`
  - `hnsw_ef_search = 40`
  - `hnsw_dist_method = vector_cosine_ops`

Resposta oral sugerida:

> A busca densa é feita com embeddings. O LlamaIndex transforma a query em embedding usando o mesmo provider configurado, consulta o pgvector no PostgreSQL e retorna os chunks mais similares por distância de cosseno. A configuração padrão usa `nomic-embed-text` via Ollama, 768 dimensões, e índice HNSW no pgvector.

---

## 5. Busca lexical

### O que ela busca?

A busca lexical consulta **resultados de treinamento ML**, não os documentos do pipeline. Ela responde perguntas como:

- Qual modelo teve melhor ROC-AUC?
- Quais métricas da logistic regression?
- Qual feature foi mais importante?
- Compare logistic regression, histogram gradient boosting e baseline.

### Como os dados entram na tabela?

Arquivos principais:

- `src/rag_intelligence/train_cli.py`
- `src/rag_intelligence/training_metadata.py`

Depois que o treino termina, o CLI:

1. treina o modelo;
2. loga métricas no MLflow;
3. chama `ensure_training_schema()`;
4. chama `store_training_result(...)`;
5. salva métricas na tabela `training_runs`.

A tabela contém:

- `run_id`
- `model_name`
- `experiment`
- métricas: `roc_auc`, `f1`, `balanced_accuracy`, `log_loss_val`, `brier`
- `feature_importances` em JSONB
- métricas segmentadas por mapa/metade em JSONB
- `params` em JSONB
- `search_text`
- `search_vector` gerado automaticamente

### Como busca?

Arquivo principal: `src/rag_intelligence/lexical_retrieval.py`

A query principal faz:

```sql
SELECT ..., ts_rank(search_vector, websearch_to_tsquery('english', %s)) AS rank_score
FROM training_runs
WHERE search_vector @@ websearch_to_tsquery('english', %s)
ORDER BY rank_score DESC, created_at DESC, model_name ASC
LIMIT %s;
```

Se não encontrar resultados, existe um fallback heurístico:

- busca linhas recentes;
- normaliza tokens da pergunta;
- remove stop words;
- detecta intenção de performance, como “melhor modelo”;
- escolhe métrica preferida, como ROC-AUC, F1, balanced accuracy, log loss ou Brier;
- ordena os modelos pela métrica correta.

Resposta oral sugerida:

> A lexical search usa Full-Text Search do PostgreSQL. Na hora do treino, criamos um texto pesquisável com nome do modelo, métricas, features e segmentos. O PostgreSQL transforma isso em `tsvector`, indexa com GIN e a busca usa `websearch_to_tsquery` com ranking `ts_rank`. Se a busca textual não retornar nada, temos um fallback que interpreta intenção de performance e ordena pelos scores dos modelos.

---

## 6. Busca híbrida

Arquivo principal: `src/rag_intelligence/api/routes/search.py`

Endpoint: `POST /search/hybrid`

Body principal:

```json
{
  "query": "pergunta do usuário",
  "embedding_run_id": "pipeline-docs",
  "top_k": 5,
  "include_semantic": true,
  "include_lexical": true,
  "model_filter": null
}
```

O endpoint faz duas buscas independentes:

1. **Semântica**: chama `search_events(...)` em `retrieval.py`.
2. **Lexical**: chama `lexical_search(...)` em `lexical_retrieval.py`.

Ele retorna:

```json
{
  "semantic_results": [...],
  "lexical_results": [...],
  "retrieval_ms": 123
}
```

Importante: ele não faz reranking combinado sofisticado. Ele devolve as duas listas separadas, e o LLM/frontend usa os dois blocos como contexto.

Resposta oral sugerida:

> A busca híbrida combina duas estratégias complementares. A semântica encontra trechos relevantes da documentação mesmo quando as palavras não batem exatamente. A lexical é melhor para métricas, nomes de modelos e termos exatos. O endpoint retorna os dois conjuntos separados para o chat sintetizar a resposta.

---

## 7. Configurações importantes

Arquivo: `.env.example` e `src/rag_intelligence/settings.py`

### Banco e vetores

- `PG_HOST=localhost`
- `PG_PORT=54330`
- `PG_USER=raguser`
- `PG_PASSWORD=ragpassword`
- `PG_DATABASE=ragdb`
- `PG_TABLE_NAME=rag_embeddings`
- `PG_EMBED_DIM=768`

### Ollama / modelos

- `OLLAMA_BASE_URL=http://localhost:11434`
- `DEFAULT_LLM=ollama/qwen2.5`
- `DEFAULT_EMBED_MODEL=ollama/nomic-embed-text`
- `OLLAMA_EMBED_BATCH_SIZE=32`

### RAG

- `DEFAULT_EMBEDDING_RUN_ID`
- para docs do pipeline, o run usado pelo chat é `pipeline-docs`.

### MinIO

- `MINIO_ENDPOINT=localhost:9000`
- buckets Bronze/Silver/Gold e prefixos de dataset/run.

### MLflow

- `MLFLOW_TRACKING_URI=http://localhost:5000`
- `MLFLOW_EXPERIMENT_NAME=csgo_round_next_winner`

---

## 8. Principais arquivos para citar

### Backend RAG/search

- `src/rag_intelligence/api/routes/search.py`
  - expõe `/search` e `/search/hybrid`.
- `src/rag_intelligence/api/routes/rag.py`
  - expõe `/rag/query` e o alias `/query` com streaming opcional.
- `src/rag_intelligence/api/routes/metadata.py`
  - expõe `/metadata` para metadados do serviço, runs e linhagem.
- `src/rag_intelligence/retrieval.py`
  - busca semântica/densa com LlamaIndex retriever.
- `src/rag_intelligence/rag.py`
  - QueryEngine do LlamaIndex para RAG com síntese.
- `src/rag_intelligence/db.py`
  - conexão PostgreSQL, `PGVectorStore`, índices de metadata e HNSW.
- `src/rag_intelligence/providers.py`
  - registry de LLMs/embeddings: Ollama, OpenAI, Anthropic, Voyage.
- `src/rag_intelligence/settings.py`
  - configurações por variáveis de ambiente.

### Busca lexical / ML

- `src/rag_intelligence/training_metadata.py`
  - cria tabela `training_runs`, `tsvector`, índice GIN, e grava métricas.
- `src/rag_intelligence/lexical_retrieval.py`
  - executa Full-Text Search e fallback heurístico.
- `src/rag_intelligence/train_cli.py`
  - treina modelos, loga MLflow e grava metadados lexicais.
- `src/rag_intelligence/round_winner.py`
  - construção de features e treino do preditor de vencedor do próximo round.

### Embeddings/documentos

- `src/rag_intelligence/embed_docs_cli.py`
  - embedda os Markdown de `docs/pipeline` com run `pipeline-docs`.
- `src/rag_intelligence/embeddings.py`
  - pipeline mais antigo/geral para embedar documentos JSONL vindos do Gold.
- `src/rag_intelligence/documents.py`
  - geração de documentos agregados a partir do Gold; mantido para compatibilidade, mas o projeto atual usa docs do pipeline para semântico e training metadata para lexical.

### Frontend

- `frontend/src/app/api/chat/route.ts`
  - tool calling do chat, chama `/search/hybrid`.
- `frontend/src/lib/chat-models.ts`
  - modelos e capacidades de tool/reasoning.
- `frontend/src/components/chat/*`
  - interface do chat.

### Infra

- `docker-compose.yml`
  - serviços: MinIO, TimescaleDB/PostgreSQL, Ollama, API, frontend, MLflow, observability.
- `Makefile`
  - comandos padronizados de setup, pipeline, treino, busca e demo.
- `pyproject.toml`
  - dependências e scripts CLI.

---

## 9. Perguntas prováveis do professor e respostas

### “Vocês usam DuckDB?”

Não. Usamos PostgreSQL/TimescaleDB. A parte vetorial usa pgvector e a parte lexical usa Full-Text Search do PostgreSQL. DuckDB não faz parte da arquitetura de busca.

### “Qual é a diferença entre busca lexical e densa?”

Busca lexical procura correspondência textual e termos exatos. No projeto ela usa `tsvector`, `websearch_to_tsquery` e `ts_rank` no PostgreSQL para métricas de modelos.

Busca densa transforma textos e perguntas em vetores de embedding e compara similaridade semântica. No projeto ela usa LlamaIndex + pgvector + embeddings do Ollama.

### “Por que usar as duas?”

Porque elas resolvem problemas diferentes. A semântica é boa para perguntas abertas sobre arquitetura e pipeline, mesmo sem termos exatos. A lexical é melhor para métricas, nomes de modelos, valores e termos técnicos específicos.

### “Onde estão os embeddings?”

No PostgreSQL/TimescaleDB com pgvector. A tabela configurada é `rag_embeddings`, e o LlamaIndex cria/usa a tabela de dados `data_rag_embeddings`.

### “Qual modelo de embedding?”

Padrão: `ollama/nomic-embed-text`, com 768 dimensões. Também há suporte a OpenAI `text-embedding-3-small` e Voyage `voyage-3` via `ProviderRegistry`.

### “Qual LLM?”

Padrão local: Ollama com Qwen 2.5. O projeto também tem abstração para OpenAI GPT-4o e Claude Sonnet se houver API keys.

### “O que o `/search/hybrid` faz?”

Ele executa busca semântica em documentos do pipeline e busca lexical em resultados de treinamento. Retorna dois arrays separados: `semantic_results` e `lexical_results`.

### “O que o `/query` e o `/rag/query` fazem?”

Eles fazem RAG completo no backend usando LlamaIndex QueryEngine: retrieval vetorial + prompt QA + LLM, com streaming SSE opcional. `/query` é o alias direto para atender ao checklist da apresentação; `/rag/query` é o endpoint original namespaced.

### “O que o `/metadata` faz?”

Ele expõe metadados do serviço e da governança. Sem parâmetros retorna configuração resumida; com `stage` retorna o último run daquele estágio; com `stage`, `run_id` e `lineage=true` retorna a cadeia de linhagem upstream.

### “Como garantem governança?”

Com arquitetura Medallion no MinIO e registro de runs/metadados. Cada etapa tem run id, source run id, contadores de qualidade e artefatos. Isso permite rastrear a origem dos dados desde Bronze até Gold, Documents, Embeddings e ML Training.

### “Por que pgvector em vez de Milvus?”

Porque simplifica a infraestrutura local: um único PostgreSQL/TimescaleDB serve para dados relacionais, metadados, FTS lexical e vetores. Para um projeto acadêmico/local, reduz complexidade operacional mantendo capacidade de busca vetorial.

### “A busca híbrida faz reranking?”

Não há reranking combinado sofisticado. O backend executa as buscas separadamente e retorna os resultados. A síntese do chat usa ambos como contexto.

### “Qual é o maior risco técnico?”

A qualidade do RAG depende da qualidade dos documentos indexados e das métricas salvas. Se os docs ou metadados estiverem incompletos, o LLM não deve inventar; ele deve dizer que não há dados disponíveis.

---

## 10. Comandos úteis para demonstração

### Subir stack local

```bash
make start
```

### Subir com Docker Compose

```bash
docker compose up -d
```

### Rodar pipeline local com Ollama do host

```bash
make pipeline-local-ollama
```

### Embedar docs do pipeline

```bash
make embed-docs
```

### Treinar modelos

```bash
make train-logreg
make train-histgbt
make train-baseline
```

### Abrir interfaces

- Frontend: `http://localhost:3002`
- API docs: `http://localhost:8000/docs`
- MinIO: `http://localhost:9001`
- MLflow: `http://localhost:5000`

---

## 11. Roteiro oral de 3 a 5 minutos

1. **Contexto**
   - “Escolhemos CS:GO analytics com dataset público do Kaggle.”
   - “O objetivo é consultar dados e resultados de ML via linguagem natural.”

2. **Dados e governança**
   - “Seguimos Medallion: Bronze bruto, Silver limpo, Gold curado.”
   - “MinIO armazena os artefatos e PostgreSQL registra metadados e vetores.”

3. **IA/RAG**
   - “Usamos LlamaIndex para embeddings e retrieval.”
   - “Ollama roda localmente LLM e embeddings.”
   - “O chat usa tool calling para buscar antes de responder.”

4. **Busca semântica**
   - “Docs do pipeline são embedados com `nomic-embed-text` e salvos em pgvector.”
   - “A pergunta vira embedding e buscamos por similaridade de cosseno.”

5. **Busca lexical**
   - “Resultados de ML ficam na tabela `training_runs`.”
   - “Usamos Full-Text Search do PostgreSQL, não DuckDB.”
   - “A busca ranqueia com `ts_rank` e tem fallback por métricas.”

6. **Aplicação**
   - “FastAPI expõe `/search`, `/search/hybrid` e `/rag/query`.”
   - “Next.js fornece o chat com streaming e fontes.”

7. **Fechamento**
   - “A principal decisão arquitetural foi centralizar busca vetorial e lexical no PostgreSQL para simplificar a infraestrutura local.”

---

## 12. Frase final para defender a arquitetura

> O projeto implementa uma arquitetura RAG local e governada: MinIO cuida do data lake em camadas, PostgreSQL/TimescaleDB centraliza metadados, busca lexical e vetorial com pgvector, LlamaIndex faz embedding/retrieval, Ollama permite inferência local, e o frontend usa tool calling para garantir que as respostas sejam fundamentadas nos dados recuperados.
