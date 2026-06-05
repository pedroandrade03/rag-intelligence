# Backend Architecture — `src/rag_intelligence`

Mapa de arquitetura do backend Python: endpoints HTTP, CLIs batch, módulos internos e integrações externas.  
Entry point da API: `rag_intelligence.api.main:app` (porta padrão **8000**).

---

## 1. Visão geral — camadas e pacotes

```mermaid
flowchart TB
    subgraph clients [Clientes]
        FE[Next.js frontend<br/>RAG_API_URL]
        CLI[CLIs batch<br/>pyproject scripts]
        SW[Swagger / curl / integrações]
    end

    subgraph api [api/ — FastAPI]
        MAIN[main.py<br/>create_app + lifespan]
        MW[middleware.py<br/>RequestID + CORS]
        H[routes/health.py]
        M[routes/metadata.py]
        S[routes/search.py]
        R[routes/rag.py]
        DEPS[deps.py<br/>Settings + Registry]
    end

    subgraph core [Domínio / serviços]
        RET[retrieval.py<br/>search_events]
        LEX[lexical_retrieval.py]
        RAG[rag.py<br/>rag_query / stream]
        PROV[providers.py<br/>ProviderRegistry]
        DB[db.py<br/>PGVectorStore]
        META[metadata.py<br/>dataset_runs]
        TRAIN[training_metadata.py]
    end

    subgraph pipeline [Pipeline batch — CLIs]
        BR[cli.py → ingest.py]
        SV[silver.py + silver_cli]
        GD[gold.py + gold_cli]
        DOC[documents.py + document_cli]
        EMB[embeddings*.py + embedding_cli]
        EDOC[embed_docs_cli]
        TRN[train_cli + round_winner]
        AUD[run_audit_cli]
        SRC[search_cli]
    end

    subgraph external [Infra externa]
        PG[(TimescaleDB / PostgreSQL<br/>dataset_runs + pgvector + training_runs)]
        MINIO[(MinIO<br/>bronze / silver / gold)]
        OLL[Ollama<br/>embeddings + LLM fallback]
        LLM[llama.cpp / OpenAI / Anthropic]
        MLF[MLflow<br/>experimentos ML]
        KAG[Kaggle API]
        OTEL[OTEL Collector<br/>opcional]
    end

    FE --> api
    CLI --> pipeline
    SW --> api

    MAIN --> MW --> H & M & S & R
    H & M & S & R --> DEPS
    S --> RET & LEX
    R --> RAG
    RET & RAG --> PROV & DB
    LEX --> PG
    M --> META & LEX
    PROV --> OLL & LLM

    pipeline --> MINIO & PG & OLL & MLF & KAG
    EMB & EDOC --> DB
    BR & SV & GD & DOC & EMB --> META

    MAIN -.-> OTEL
```

---

## 2. Superfície HTTP — endpoints por use case

| Use case | Método | Path | Handler | Módulo de domínio |
|----------|--------|------|---------|-------------------|
| **Ops / health** | `GET` | `/health` | `health()` | — |
| **Governança** | `GET` | `/metadata` | `metadata()` | `metadata.py` |
| **Governança ML** | `GET` | `/metadata/training` | `training_metadata()` | `lexical_retrieval.py` |
| **Busca vetorial** | `POST` | `/search` | `search()` | `retrieval.py` |
| **Busca híbrida** | `POST` | `/search/hybrid` | `hybrid_search()` | `retrieval.py` + `lexical_retrieval.py` |
| **RAG server-side** | `POST` | `/rag/query` | `query()` | `rag.py` |
| **RAG server-side (alias)** | `POST` | `/query` | `query()` | `rag.py` |

Documentação interativa: `http://localhost:8000/docs`

---

## 3. Bootstrap da aplicação (todas as requisições)

```mermaid
sequenceDiagram
    autonumber
    participant Client
    participant MW as RequestIDMiddleware
    participant CORS as CORSMiddleware
    participant Route as routes/*
    participant Deps as deps.py
    participant App as app.state

    Client->>MW: HTTP request
    MW->>MW: bind X-Request-ID (structlog)
    MW->>CORS: forward
    CORS->>Route: matched route

    Note over App: lifespan (startup)
    App->>App: AppSettings.from_env()
    App->>App: ProviderRegistry(settings)
    opt OTEL_ENABLED=true
        App->>App: telemetry.setup_telemetry()
        App->>App: instrument_fastapi(app)
    end

    Route->>Deps: Depends(get_settings / get_registry)
    Deps->>App: read settings + registry
    Deps-->>Route: SettingsDep, RegistryDep
```

**Arquivos:** `api/main.py`, `api/middleware.py`, `api/deps.py`, `settings.py`, `telemetry.py`

---

## 4. Use case — Ops & health check

**Objetivo:** verificar se o processo FastAPI está vivo (load balancer, Docker, CI).

```mermaid
sequenceDiagram
    autonumber
    participant Client
    participant API as GET /health
    participant MW as middleware

    Client->>MW: GET /health
    MW->>API: health()
    API-->>Client: 200 {"status": "ok"}
```

| Campo | Valor |
|-------|-------|
| Path | `GET /health` |
| Auth | nenhuma |
| Dependências externas | nenhuma |

---

## 5. Use case — Governança de dados & lineage

**Objetivo:** consultar catálogo `dataset_runs`, metadados de serviço e último treinamento ML.

### 5.1 `GET /metadata` — modos de consulta

| Query params | Comportamento |
|--------------|---------------|
| _(vazio)_ | Metadados do serviço (stages, tabela vetorial, modelos default) |
| `stage=bronze\|silver\|gold\|documents\|embeddings` | Último run da stage |
| `stage` + `run_id` | Run específico |
| `stage` + `run_id` + `lineage=true` | Cadeia upstream completa |

```mermaid
sequenceDiagram
    autonumber
    participant Client
    participant API as GET /metadata
    participant META as metadata.py
    participant PG as PostgreSQL<br/>dataset_runs

    Client->>API: GET /metadata?stage=gold&run_id=...
    API->>API: MetadataSettings.from_app(settings)
    alt lineage=true
        API->>META: get_run_lineage(stage, run_id)
        META->>PG: SELECT chain by source_run_id
        PG-->>META: RunLineageReport
        META-->>API: report.to_dict()
    else run_id present
        API->>META: get_run(stage, run_id)
    else stage only
        API->>META: get_latest_run(stage)
    end
    META->>PG: query
    PG-->>API: RunRecord
    API-->>Client: 200 JSON
```

### 5.2 `GET /metadata/training`

**Objetivo:** expor o último run de treinamento para tools do chat e demos.

```mermaid
sequenceDiagram
    autonumber
    participant Client
    participant API as GET /metadata/training
    participant LEX as lexical_retrieval.py
    participant PG as PostgreSQL<br/>training_runs

    Client->>API: ?latest=true&model_filter=logistic_regression
    API->>LEX: get_latest_training_run(model_filter)
    LEX->>PG: SELECT latest run_id + model rows
    PG-->>LEX: rows
    LEX-->>API: List[LexicalSearchResult]
    API-->>Client: {run_id, created_at, models[], count}
```

**CLI equivalente (sem HTTP):** `run-audit --stage embeddings --run-id <id>` → `metadata.get_run_lineage()`

**Arquivos:** `api/routes/metadata.py`, `metadata.py`, `lexical_retrieval.py`, `run_audit_cli.py`

---

## 6. Use case — Busca semântica (vetorial)

**Objetivo:** recuperar chunks/documentos por similaridade coseno no pgvector, **sem** síntese LLM.

**Endpoint:** `POST /search`

### Request body (principais campos)

| Campo | Obrigatório | Descrição |
|-------|-------------|-----------|
| `query` | sim | Texto da busca |
| `embedding_run_id` | sim* | Run de embeddings (* ou `DEFAULT_EMBEDDING_RUN_ID`) |
| `top_k` | não (5) | Quantidade de resultados |
| `event_type`, `map_name`, `file_name`, `round_number` | não | Filtros metadata CS:GO |
| `pipeline_phase` | não | Filtro docs do pipeline (`bronze`, `silver`, …) |

```mermaid
sequenceDiagram
    autonumber
    participant Client
    participant API as POST /search
    participant RET as retrieval.search_events
    participant REG as ProviderRegistry
    participant OLL as Ollama / OpenAI / Voyage
    participant VS as PGVectorStore
    participant PG as PostgreSQL<br/>data_rag_embeddings

    Client->>API: SearchBody JSON
    API->>API: resolve embedding_run_id
    API->>RET: SearchRequest(...)
    RET->>RET: pgvector_data_table_exists()
    RET->>REG: get_embed_model()
    REG->>OLL: embed query vector
    OLL-->>REG: vector[768]
    RET->>VS: create_vector_store(perform_setup=false)
    RET->>RET: VectorStoreIndex.as_retriever(filters)
    RET->>PG: similarity search + metadata filters
    PG-->>RET: NodeWithScore[]
    RET->>RET: build_search_result + sanitize_metadata
    RET-->>API: SearchResponse
    API-->>Client: 200 JSON (results, retrieval_ms)
```

**CLI equivalente:** `semantic-search --query "..." --embedding-run-id pipeline-docs`

**Arquivos:** `api/routes/search.py`, `retrieval.py`, `providers.py`, `db.py`, `search_cli.py`

---

## 7. Use case — Busca híbrida (docs + métricas ML)

**Objetivo:** combinar busca semântica (documentação / eventos embeddados) com busca lexical FTS em `training_runs`.  
**Principal integração do frontend** via tools do chat.

**Endpoint:** `POST /search/hybrid`

| Flag | Efeito |
|------|--------|
| `include_semantic=true` | Chama `search_events()` com `document_tier=pipeline_doc` |
| `include_lexical=true` | Chama `lexical_search()` em `training_runs` |
| `pipeline_phase` | Filtro semântico por fase do pipeline |
| `model_filter` | Filtro lexical por `model_name` |

```mermaid
sequenceDiagram
    autonumber
    participant FE as Next.js /api/chat<br/>tools
    participant API as POST /search/hybrid
    participant RET as retrieval.search_events
    participant LEX as lexical_search
    participant OLL as Ollama embeddings
    participant PGV as pgvector table
    participant PGT as training_runs FTS

    FE->>API: {query, include_semantic, include_lexical, ...}

    par include_semantic
        API->>RET: SearchRequest(document_tier=pipeline_doc)
        RET->>OLL: embed query
        RET->>PGV: vector retrieval
        PGV-->>API: semantic_results[]
    and include_lexical
        API->>LEX: lexical_search(query, model_filter)
        LEX->>PGT: websearch_to_tsquery + ts_rank
        PGT-->>API: lexical_results[]
    end

    API-->>FE: {semantic_results, lexical_results, retrieval_ms}
```

### Integração frontend → backend (tools)

```mermaid
flowchart LR
    subgraph next [frontend/src/app/api/chat/route.ts]
        T1[searchPipelineDocs]
        T2[searchTrainingMetrics]
        T3[getLatestTrainingRun]
    end

    T1 -->|POST /search/hybrid<br/>semantic only| HYB[/search/hybrid]
    T2 -->|POST /search/hybrid<br/>lexical only| HYB
    T3 -->|GET /metadata/training| TRN[/metadata/training]

    HYB --> API[FastAPI rag-api :8000]
    TRN --> API
```

**Arquivos:** `api/routes/search.py`, `retrieval.py`, `lexical_retrieval.py`, `training_metadata.py`

---

## 8. Use case — RAG server-side (retrieval + LLM)

**Objetivo:** endpoint alternativo que faz retrieval **e** gera resposta com LlamaIndex QueryEngine no backend.  
O chat principal do frontend usa AI SDK + tools (`/search/hybrid`); este endpoint é útil para integrações diretas e Swagger.

**Endpoints:** `POST /rag/query` e `POST /query` (mesmo handler)

| Campo | Descrição |
|-------|-----------|
| `stream=false` | JSON sync: `{answer, sources, retrieval_ms, generation_ms}` |
| `stream=true` | SSE: `sources` → `chunk`* → `done` |
| `llm_key` | Override LLM (`llama-cpp/gemma4`, `ollama/...`, `gpt-4o`, …) |

```mermaid
sequenceDiagram
    autonumber
    participant Client
    participant API as POST /rag/query
    participant RAG as rag.py
    participant IDX as VectorStoreIndex
    participant REG as ProviderRegistry
    participant OLL as Ollama / llama.cpp / cloud LLM
    participant PG as pgvector

    Client->>API: RAGBody {query, embedding_run_id, stream}

    alt stream=false
        API->>RAG: rag_query(RAGRequest)
        RAG->>IDX: as_query_engine(filters, QA_PROMPT, llm)
        IDX->>PG: retrieve context
        IDX->>REG: get_llm(llm_key)
        REG->>OLL: complete(prompt + context)
        OLL-->>RAG: answer text
        RAG-->>API: RAGResponse
        API-->>Client: 200 JSON
    else stream=true
        API->>RAG: rag_query_stream (async generator)
        RAG->>IDX: aquery + async_response_gen
        RAG-->>Client: SSE event:sources
        loop tokens
            RAG-->>Client: SSE event:chunk
        end
        RAG-->>Client: SSE event:done
    end
```

**Arquivos:** `api/routes/rag.py`, `rag.py`, `retrieval.py`, `providers.py`, `db.py`

---

## 9. Use case — Pipeline batch (CLIs, sem HTTP)

Jobs executados via Docker Compose profile `jobs` ou scripts `pyproject.toml`.  
Cada stage registra lineage em `dataset_runs`.

```mermaid
flowchart LR
    KAG[Kaggle] --> BR[bronze-import<br/>cli.py + ingest.py]
    BR --> MINIO_B[(MinIO bronze)]
    MINIO_B --> SV[silver-transform<br/>silver.py]
    SV --> MINIO_S[(MinIO silver)]
    MINIO_S --> GD[gold-transform<br/>gold.py]
    GD --> MINIO_G[(MinIO gold)]
    MINIO_G --> DOC[document-build<br/>documents.py]
    DOC --> MINIO_D[(MinIO documents JSONL)]
    MINIO_D --> EMB[embedding-ingest<br/>embeddings_pipeline.py]
    EMB --> PGV[(pgvector)]
    EMB --> OLL[Ollama embeddings]

    BR & SV & GD & DOC & EMB --> META[(dataset_runs)]

    REPO[docs/pipeline/*.md] --> EDOC[embed-docs<br/>embed_docs_cli.py]
    EDOC --> PGV
    EDOC --> META
```

### Sequência — ingestão Bronze

```mermaid
sequenceDiagram
    autonumber
    participant CLI as bronze-import
    participant ING as ingest.run_import
    participant KAG as Kaggle API
    participant MINIO as MinIO bronze
    participant META as metadata.register_run

    CLI->>ING: Settings.from_env()
    ING->>KAG: download dataset ZIP
    ING->>ING: extract CSV/PNG
    ING->>MINIO: upload raw/ + extracted/
    ING-->>CLI: uploaded_objects[]
    CLI->>META: stage=bronze, run_id, quality_summary
```

### Sequência — embedding ingest (eventos CS:GO)

```mermaid
sequenceDiagram
    autonumber
    participant CLI as embedding-ingest
    participant PIPE as embeddings_pipeline
    participant MINIO as MinIO documents/
    participant REG as ProviderRegistry
    participant OLL as Ollama
    participant DB as db.py
    participant PG as pgvector
    participant META as metadata

    CLI->>PIPE: run_embedding_ingest(EmbeddingSettings)
    PIPE->>MINIO: read manifest.json + JSONL parts
    PIPE->>DB: ensure_pgvector_storage_contract()
    PIPE->>REG: get_embed_model()
    loop batches
        PIPE->>OLL: embed documents
        PIPE->>PG: upsert vectors + metadata
    end
    PIPE->>MINIO: write embeddings/manifest + quality_report
    CLI->>META: stage=embeddings
```

| Script CLI | Entry point | Módulo principal | Saída |
|------------|-------------|------------------|-------|
| `bronze-import` | `cli.py` | `ingest.py` | MinIO `bronze/...` |
| `silver-transform` | `silver_cli.py` | `silver.py` | MinIO `silver/.../round_meta_context.csv` |
| `gold-transform` | `gold_cli.py` | `gold.py` | MinIO `gold/.../round_context.csv` |
| `document-build` | `document_cli.py` | `documents.py` | MinIO `.../documents/*.jsonl` |
| `embedding-ingest` | `embedding_cli.py` | `embeddings_pipeline.py` | pgvector + MinIO reports |
| `embed-docs` | `embed_docs_cli.py` | — | pgvector run `pipeline-docs` |
| `semantic-search` | `search_cli.py` | `retrieval.py` | stdout JSON |
| `run-audit` | `run_audit_cli.py` | `metadata.py` | stdout text/json |

---

## 10. Use case — Treinamento ML (batch)

**Objetivo:** treinar modelos de previsão de vencedor do próximo round a partir da Gold, registrar em MLflow e indexar métricas para busca lexical.

```mermaid
sequenceDiagram
    autonumber
    participant CLI as train-logreg / train-histgbt / train-baseline
    participant TRN as train_cli.main
    participant MINIO as MinIO gold
    participant RW as round_winner.py
    participant MLF as MLflow
    participant PG as training_runs

    CLI->>TRN: TrainSettings.from_env()
    TRN->>MINIO: get round_context.csv (Gold)
    TRN->>RW: build_supervised_frame + train_next_round_winner
    RW->>MLF: log_training_to_mlflow(metrics, artifacts)
    RW->>PG: upsert training_runs + search_vector
    TRN-->>CLI: exit 0
```

| Script | Modelo |
|--------|--------|
| `train-logreg` | Regressão logística |
| `train-histgbt` | Histogram Gradient Boosting |
| `train-baseline` | Baseline |

**Consumido depois por:** `POST /search/hybrid` (lexical) e `GET /metadata/training`

**Arquivos:** `train_cli.py`, `round_winner.py`, `training_metadata.py`, `lexical_retrieval.py`

---

## 11. Mapa de dependências externas

```mermaid
flowchart TB
    subgraph api_layer [Camada HTTP]
        FAST[FastAPI routes]
    end

    subgraph stores [Persistência]
        PG_META[(dataset_runs)]
        PG_VEC[(data_rag_embeddings<br/>HNSW + BTREE metadata)]
        PG_TRN[(training_runs<br/>FTS search_vector)]
    end

    subgraph object [Object storage]
        MINIO[(MinIO S3)]
    end

    subgraph ai [Provedores IA]
        OLL_EMB[Ollama nomic-embed-text]
        OLL_LLM[Ollama qwen2.5]
        LLAMA[llama.cpp OpenAI-compatible]
        OPENAI[OpenAI API]
        ANTH[Anthropic API]
        VOY[Voyage API]
    end

    FAST --> PG_META & PG_VEC & PG_TRN
    FAST --> OLL_EMB & OLL_LLM & LLAMA & OPENAI & ANTH

    pipeline_jobs[CLIs batch] --> MINIO & PG_META & PG_VEC & PG_TRN
    pipeline_jobs --> OLL_EMB
    pipeline_jobs --> MLF[MLflow :5000]
    pipeline_jobs --> KAG[Kaggle]
```

Variáveis principais: `.env` / `AppSettings` (`settings.py`), `MetadataSettings` (`metadata.py`), configs por job (`config.py`).

---

## 12. Índice de arquivos `src/rag_intelligence/`

| Path | Responsabilidade |
|------|------------------|
| `api/main.py` | App factory, routers, lifespan, OTEL |
| `api/deps.py` | Injeção `SettingsDep`, `RegistryDep` |
| `api/middleware.py` | `X-Request-ID`, structlog context |
| `api/routes/health.py` | `GET /health` |
| `api/routes/metadata.py` | `GET /metadata`, `GET /metadata/training` |
| `api/routes/search.py` | `POST /search`, `POST /search/hybrid` |
| `api/routes/rag.py` | `POST /rag/query`, `POST /query` |
| `retrieval.py` | Core semântico — `search_events()` |
| `lexical_retrieval.py` | FTS em `training_runs` |
| `rag.py` | Síntese LlamaIndex — sync + SSE stream |
| `providers.py` | LLM + embedding providers com fallback Ollama |
| `db.py` | PGVectorStore, índices metadata, contrato PB07 |
| `metadata.py` | Catálogo `dataset_runs`, lineage audit |
| `settings.py` | `AppSettings` (API + retrieval) |
| `config.py` | Settings por job (Bronze, Embed, Train, …) |
| `ingest.py` | Download Kaggle → MinIO Bronze |
| `silver.py` / `gold.py` / `documents.py` | Transformações Medallion |
| `embeddings_pipeline.py` | IngestionPipeline → pgvector |
| `embed_docs_cli.py` | Embeddar `docs/pipeline/*.md` |
| `train_cli.py` / `round_winner.py` | Treino ML + MLflow |
| `search_cli.py` | CLI de busca semântica |
| `run_audit_cli.py` | Auditoria de lineage |
| `telemetry.py` | OpenTelemetry (Jaeger/Prometheus) |
| `logging.py` | structlog setup |
| `minio_utils.py` | Helpers MinIO |

---

## 13. Referência rápida — fluxo ponta a ponta (chat demo)

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant FE as Next.js UI
    participant Chat as POST /api/chat
    participant API as FastAPI :8000
    participant PG as PostgreSQL
    participant OLL as Ollama

    User->>FE: pergunta sobre pipeline
    FE->>Chat: useChat message
    Chat->>API: POST /search/hybrid (tool)
    API->>OLL: embed query
    API->>PG: pgvector search (pipeline-docs)
    PG-->>Chat: semantic_results
    Chat->>OLL: streamText (síntese PT-BR)
    OLL-->>FE: SSE tokens
    FE-->>User: resposta + fontes
```

---

*Gerado a partir do código em `src/rag_intelligence/` — alinhar com `http://localhost:8000/docs` ao evoluir rotas.*
