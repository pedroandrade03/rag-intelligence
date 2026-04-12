# Arquitetura do RAG Intelligence

O RAG Intelligence e uma plataforma de analytics de partidas de CS:GO construida sobre RAG (Retrieval-Augmented Generation). O sistema transforma dados brutos de partidas em conhecimento consultavel via chat.

## Camadas do sistema

### Camada de Dados (Medallion Architecture)
Pipeline de dados em estagios: Bronze (ingestao bruta) -> Silver (limpeza) -> Gold (curacao) -> ML Training (predicao). Cada estagio transforma e refina os dados, com rastreabilidade completa via catalogo `dataset_runs` no PostgreSQL.

### Camada de IA
- **LlamaIndex**: Gera embeddings via IngestionPipeline e faz busca vetorial via VectorStoreIndex.
- **Ollama**: Roda localmente modelos de LLM (Qwen 2.5 7B, ~12GB RAM) e embedding (nomic-embed-text, 768 dimensoes, ~300MB RAM).
- **AI SDK**: Sintese de resposta no frontend via streaming SSE.

### Camada de Aplicacao
- **Backend**: FastAPI com endpoints /search (busca vetorial), /search/hybrid (busca hibrida semantica + lexical), e /rag/query (RAG completo com sintese LLM).
- **Frontend**: Next.js 16 com chat interativo, selecao de modelo e modo RAG configuravel.

### Camada de Observabilidade
OpenTelemetry exportando traces para Jaeger, metricas para Prometheus, e dashboards em Grafana. Logging estruturado via structlog com trace_id/span_id automaticos.

## Retrieval Hibrido

O sistema usa dois tipos de retrieval:

### Semantico (pgvector)
Documentos de pipeline (Markdown) sao chunked, embedados via nomic-embed-text, e armazenados no pgvector. Busca por similaridade coseno para perguntas sobre o pipeline e arquitetura.

### Lexical (PostgreSQL FTS)
Metadados de treinamento ML (metricas, feature importances, segmentos) sao armazenados com tsvector no PostgreSQL. Busca full-text para perguntas sobre resultados de treinamento.

## Infraestrutura

Tudo orquestrado via Docker Compose:
- **MinIO**: Data lake S3-compatible (portas 9000/9001)
- **TimescaleDB**: PostgreSQL com pgvector (porta 54330)
- **Ollama**: Inferencia local (porta 11434)
- **MLflow**: Experiment tracking (porta 5000, perfil mlflow)
- **Observability stack**: Jaeger + Prometheus + Grafana (perfil observability)

## ProviderRegistry

Fabrica lazy-loading que abstrai provedores de LLM e embedding. Suporta Ollama (padrao/fallback), OpenAI e Anthropic para LLMs; Ollama, OpenAI e Voyage para embeddings. Fallback automatico para Ollama se provedor falhar.

## Governanca de Dados

Rastreabilidade completa via tabela `dataset_runs`. Cada run registra: run_id, stage, source_run_id, contadores, quality_summary JSONB. CLI `run-audit` reconstroi cadeia de linhagem e verifica integridade.
