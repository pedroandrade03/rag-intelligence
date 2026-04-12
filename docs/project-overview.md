# RAG Intelligence — Visao Completa do Projeto

## 1. Contexto Academico

Este projeto e o Trabalho Final da disciplina de Inteligencia Artificial do curso de Engenharia de Software da FACENS. As instrucoes completas estao no documento **AI_Project.pdf** (o slide deck "Trabalho Final" do professor). A proposta foi: construir uma **Plataforma RAG Enterprise com Governanca de Dados (local)** para uma empresa ficticia, usando um dataset publico como base.

Conforme definido no AI_Project.pdf, o projeto foi gerenciado com metodologia Scrum ao longo de 12 sprints, com entregaveis obrigatorios que incluem: Docker Compose funcional, Makefile com comandos padronizados, API documentada, RAG funcionando ponta a ponta, documentacao arquitetural, demonstracao final, sprint backlog documentado e justificativa das decisoes tecnicas.

Os criterios de avaliacao definidos no AI_Project.pdf sao: Arquitetura (20%), Governanca de dados (15%), Qualidade do RAG (20%), Infraestrutura (15%), Documentacao (15%) e Apresentacao final (15%).

### Stack de referencia do professor (slide "Tema / Objetivo" do AI_Project.pdf)
- Governanca com Arquitetura Medallion (Bronze / Silver / Gold)
- Armazenamento em Data Lake (MinIO)
- Banco relacional (PostgreSQL)
- Banco vetorial (Milvus)
- Experiment tracking (MLflow)
- Inferencia com LLM local (Ollama)
- API (FastAPI)
- Interface (Gradio ou frontend simples)
- Docker Compose + Makefile
- Scrum

### O que o grupo escolheu (conforme "Cenario do Projeto" no AI_Project.pdf, cada grupo escolhe um dominio)
- **Dominio**: E-sports — partidas competitivas de CS:GO
- **Dataset**: `csgo-matchmaking-damage` do Kaggle (milhoes de eventos de dano, kills, granadas e metadados de rounds de partidas reais de matchmaking)
- **Empresa ficticia**: Uma plataforma de analytics de CS:GO que permite consultar estatisticas de partidas por linguagem natural

---

## 2. O que e o RAG Intelligence

RAG Intelligence e uma plataforma completa de analytics de partidas de CS:GO construida sobre RAG (Retrieval-Augmented Generation). O sistema pega dados brutos de partidas — milhoes de eventos individuais de dano, kills, granadas e metadados de rounds — transforma esses dados em conhecimento estruturado em linguagem natural, e permite consulta-los via chat.

O usuario final abre o chat, digita uma pergunta como "Qual arma causa mais dano em Dust2?" e recebe uma resposta fundamentada em dados reais. O sistema busca os documentos mais relevantes no banco vetorial, entrega ao LLM como contexto, e o LLM sintetiza a resposta usando apenas os dados encontrados — sem alucinacao.

---

## 3. Arquitetura Geral

O sistema e organizado em cinco camadas, seguindo a arquitetura de referencia do professor com algumas decisoes tecnicas proprias (justificadas na secao 14):

### Camada de Dados (Medallion Architecture)
Pipeline de dados em 5 estagios: Bronze -> Silver -> Gold -> Documents -> Embeddings. Cada estagio transforma e refina os dados, com rastreabilidade completa entre eles via catalogo `dataset_runs` no PostgreSQL.

### Camada de IA (LlamaIndex + Ollama)
LlamaIndex e usado para gerar embeddings (via `IngestionPipeline`) e fazer busca vetorial (via `VectorStoreIndex`). Ollama roda localmente os modelos de LLM (Qwen 2.5 7B) e embedding (nomic-embed-text). A sintese de resposta e feita pelo AI SDK no frontend, nao pelo LlamaIndex.

### Camada de Aplicacao
API FastAPI no backend com endpoints de busca semantica e RAG. Frontend Next.js 16 com chat interativo, selecao de modelo e modo RAG configuravel. O professor sugeriu Gradio ou frontend simples — o grupo optou por um frontend mais completo com Next.js (justificado na secao 14).

### Camada de Observabilidade
OpenTelemetry exportando traces e metricas para Jaeger (traces), Prometheus (metricas) e Grafana (dashboards). Logging estruturado via structlog. Isso vai alem do pedido pelo professor — foi implementado para facilitar debugging do pipeline RAG.

### Camada de Infraestrutura
Tudo orquestrado via Docker Compose. MinIO como data lake, TimescaleDB com pgvector para armazenamento relacional + vetorial, Ollama para inferencia local. Makefile com todos os comandos padronizados.

---

## 4. Pipeline de Dados — A Medallion Architecture

A Medallion Architecture foi um requisito do professor (Sprint 3 — "Governanca e Medallion" no AI_Project.pdf). E um padrao de engenharia de dados que organiza dados em camadas de qualidade crescente. O projeto implementa isso com 5 estagios, cada um com seu proprio CLI, container Docker e registro de metadados.

### 4.1 Bronze — Ingestao Bruta

**O que faz**: Baixa o dataset do Kaggle como ZIP, extrai os CSVs e imagens, e armazena tudo no MinIO sem nenhuma transformacao.

**Por que existe a Bronze**: A camada Bronze preserva o dado exatamente como veio da fonte. Se algo der errado nas transformacoes posteriores, sempre e possivel voltar ao dado original. E o principio de "dados imutaveis na origem" — fundamental para governanca de dados.

**Comando**: `bronze-import` ou `make bronze`

**Saida**: Arquivos em `bronze/<dataset_prefix>/<run_id>/raw/` (ZIP original) e `bronze/<dataset_prefix>/<run_id>/extracted/` (CSVs extraidos).

**Dados contidos nos CSVs**:
- `damage.csv`: Cada evento de dano — quem atirou, quem recebeu, arma, dano HP, dano de armadura, posicoes XY, hitbox, lado (CT/T), tick, segundos
- `kills.csv`: Cada kill registrada
- `grenades.csv`: Cada granada lancada — tipo, posicao de lancamento e impacto
- `meta.csv`: Metadados de rounds — mapa, vencedor, economia, jogadores vivos

### 4.2 Silver — Limpeza e Normalizacao

**O que faz**: Le os CSVs da Bronze e aplica:
- Normalizacao de nomes de colunas (lowercase, remocao de caracteres especiais, deduplicacao de nomes)
- Validacao de campos numericos (rejeita valores negativos, NaN, invalidos)
- Remocao de linhas duplicadas
- Filtragem de linhas completamente nulas

**Por que existe a Silver**: Dados brutos vem com inconsistencias — colunas com nomes diferentes entre arquivos, valores faltantes, duplicatas. A Silver garante que todos os dados downstream tem formato consistente e confiavel. Sem essa camada, cada transformacao posterior teria que lidar com essas inconsistencias individualmente.

**Metricas de qualidade**: Cada execucao registra quantas linhas leu, quantas saiu, quantas duplicatas removeu, quantas linhas invalidas encontrou. Isso fica no `quality_report.json` e no catalogo `dataset_runs` do PostgreSQL.

**Comando**: `silver-transform` ou `make silver`

**Saida**: CSVs limpos em `silver/<dataset_prefix>/<run_id>/cleaned/`

### 4.3 Gold — Curacao e Padronizacao de Schema

**O que faz**: Le os CSVs da Silver e os transforma em um unico arquivo `events.csv` com schema padronizado de 40 colunas. O processo inclui:

- **Inferencia de tipo de evento**: A partir do nome do arquivo e das colunas disponiveis, classifica cada linha como `damage`, `kill`, `grenade`, `round_meta` ou `event`
- **Projecao para schema fixo**: Todas as linhas sao mapeadas para as mesmas 40 colunas Gold, independente do CSV de origem
- **Validacao de campos obrigatorios**: Cada tipo de evento tem seus campos minimos — kills precisam de arma, damage precisa de hp_dmg ou arm_dmg, round_meta precisa de mapa
- **Descarte de dados inuteis**: Linhas de `map_layout` sao removidas (dados de layout de mapa nao sao uteis para analise de combate)

**Por que existe a Gold**: Sem a Gold, o sistema teria que lidar com schemas diferentes para cada tipo de arquivo. A Gold cria um "contrato de dados" — qualquer coisa downstream pode contar com exatamente essas 40 colunas, nesses tipos, com essas validacoes ja aplicadas. E a camada que define os "dados prontos para consumo" da Medallion Architecture.

**Campos Gold (40 colunas)**:
- Base: `file`, `round`, `map`, `weapon`, `hp_dmg`, `arm_dmg`, `att_pos_x`, `att_pos_y`, `vic_pos_x`, `vic_pos_y`
- Extras: `event_type`, `source_file`, `tick`, `seconds`, `att_team`, `vic_team`, `att_side`, `vic_side`, `wp_type`, `nade`, `hitbox`, `bomb_site`, `is_bomb_planted`, `att_id`, `vic_id`, `att_rank`, `vic_rank`, `winner_team`, `winner_side`, `round_type`, `ct_eq_val`, `t_eq_val`, `ct_alive`, `t_alive`, `nade_land_x`, `nade_land_y`, `avg_match_rank`, `start_seconds`, `end_seconds`

**Comando**: `gold-transform` ou `make gold`

**Saida**: `gold/<dataset_prefix>/<run_id>/curated/events.csv`

### 4.4 Documents — Agregacao Estatistica em Linguagem Natural

**O que faz**: Le o `events.csv` da Gold e gera documentos de texto em portugues. NAO gera um documento por evento — isso geraria milhoes de documentos quase identicos. Em vez disso, agrega os dados em 5 niveis hierarquicos de perfis estatisticos.

**Por que agregar em vez de embeddar cada evento**: Se cada evento fosse um documento, uma busca por "AK-47 em Dust2" retornaria milhares de documentos praticamente identicos dizendo "jogador X deu 27 de dano com AK-47". O LLM nao conseguiria sintetizar nada util disso. Ao agregar, cada documento e um resumo estatistico rico — "A AK-47 em Dust2 tem dano medio de 27, taxa de headshot de 20%, e representa 39% dos eventos de dano no mapa". Isso produz respostas RAG de qualidade muito superior.

#### Os 5 niveis de documentos:

**Tier 1 — Perfil Arma-Mapa** `(map, weapon, event_type)`
Agrega todos os eventos de uma arma especifica em um mapa especifico. Exemplo:
> "Perfil de Arma: ak47 em de_dust2 (eventos de dano). Com base em 1.000 instancias de dano registradas: Dano HP medio por acerto: 27.0 | Maximo: 111. Taxa de headshot: 20.0% dos acertos. Hitboxes mais atingidas: head (20.0%), chest (40.0%). Distribuicao por lado atacante: T 60.0%, CT 40.0%."

**Tier 2 — Visao Geral do Mapa** `(map)`
Agrega todos os eventos de um mapa. Inclui contadores por tipo de evento, top armas por dano e por kills, taxa de vitoria CT vs T, duracao media de round.
> "Visao Geral do Mapa: de_mirage. Total de eventos: 1.280 (dano: 1.000, kills: 200, granadas: 50, rounds: 30). Top 5 armas por dano: ak47 (39.1%), m4a4 (23.4%)..."

**Tier 3 — Zonas de Combate (Hotspots)** `(map, grid_x, grid_y)`
Divide cada mapa em zonas de 500x500 unidades do CS:GO e agrega eventos por zona. Permite perguntas espaciais como "onde acontecem mais kills em Dust2?"
> "Zona de Combate: de_dust2, setor x:[1000,1500] y:[-500,0]. Eventos de dano: 500. Eventos de kill: 80. Armas mais usadas: ak47 (35.0%), m4a4 (26.0%)."

**Tier 4 — Perfil de Tipo de Round** `(map, round_type)`
Agrega por tipo de round (eco, full-buy, force-buy) em cada mapa. Inclui taxa de vitoria por lado e economia media.
> "Perfil de Round: tipo 'eco' em de_nuke. Total de rounds: 120. Taxa de vitoria: CT 87.5%, T 12.5%. Equipamento medio: CT R$21.000, T R$1.900."

**Tier 5 — Perfil Global de Arma** `(weapon)`
Agrega todos os eventos de uma arma em todos os mapas. Visao global do desempenho da arma.
> "Perfil Global de Arma: ak47 (todos os mapas). Total de acertos: 50.000. Dano HP medio: 24.0 | Maximo: 111. Taxa de headshot: 18%. Mapas com mais eventos: de_dust2 (20.000), de_inferno (15.000)."

**Formato de saida**: Arquivos JSONL particionados. Cada linha e um documento com `doc_id`, `text` (texto em portugues) e `metadata` (tier, mapa, arma, tipo de evento, contadores, IDs de linhagem).

**Comando**: `document-build` ou `make documents`

### 4.5 Embeddings — Vetorizacao e Armazenamento

**O que faz**: Le os documentos JSONL, gera embeddings vetoriais para cada texto usando o modelo `nomic-embed-text` (768 dimensoes) via Ollama, e armazena os vetores no PostgreSQL com pgvector.

**Como funciona tecnicamente**:
1. Le o `manifest.json` que lista todos os parts JSONL
2. Converte cada documento em um objeto `Document` do LlamaIndex
3. Usa o `IngestionPipeline` do LlamaIndex com o modelo de embedding como unica transformacao
4. O pipeline gera um vetor de 768 dimensoes para cada texto
5. Os vetores sao persistidos na tabela pgvector com toda a metadata JSONB

**Por que LlamaIndex aqui**: O LlamaIndex fornece uma abstracao que desacopla o modelo de embedding do armazenamento. Trocar de `nomic-embed-text` (Ollama local) para `text-embedding-3-small` (OpenAI) e so mudar uma variavel de ambiente. O pipeline e o mesmo.

**Otimizacoes implementadas**:
- **Processamento paralelo**: Batches de embedding rodam em paralelo via `ThreadPoolExecutor`, com escrita serializada no pgvector para evitar conflitos
- **Resume**: Se uma execucao falhar no meio, pode ser retomada de onde parou sem reprocessar o que ja foi indexado (`EMBEDDING_RESUME=true`)
- **Fallback de NaN**: Se o Ollama retornar NaN em algum embedding (bug conhecido), o sistema automaticamente divide o batch ao meio e retenta recursivamente
- **Validacao de dimensao**: Antes de indexar, valida que o embedding tem exatamente 768 dimensoes

**Comando**: `embedding-ingest` ou `make embeddings`

---

## 5. O Fluxo RAG — Da Pergunta a Resposta

Este e o fluxo ponta a ponta que o AI_Project.pdf lista como entregavel obrigatorio 4 ("RAG funcionando ponta a ponta"). Quando o usuario faz uma pergunta no chat:

### 5.1 Frontend (Next.js + AI SDK)

1. O usuario digita: "Qual arma causa mais dano em Dust2?"
2. O frontend envia a mensagem para a rota `/api/chat` do Next.js
3. A rota usa o `streamText()` do AI SDK com o modelo Ollama selecionado
4. O modelo recebe um system prompt em portugues: "Voce e um analista profissional de CS:GO. SEMPRE busque no banco antes de responder qualquer pergunta sobre CS:GO."
5. O modelo decide usar a ferramenta `searchMatchData` com parametros extraidos da pergunta: `query: "arma mais dano dust2", map_name: "de_dust2", event_type: "damage"`

### 5.2 Busca Vetorial (Backend FastAPI)

6. O frontend executa a tool call fazendo POST no endpoint `/search` do backend FastAPI
7. O backend gera um embedding da query usando o mesmo modelo (`nomic-embed-text`)
8. Faz uma busca de similaridade coseno no pgvector, filtrando por metadata (`map = de_dust2`, `event_type = damage`)
9. Retorna os top-K documentos mais similares com score de relevancia e tempo de execucao

### 5.3 Sintese (LLM)

10. Os documentos retornados sao injetados de volta na conversa como resultado da tool
11. O LLM le os perfis estatisticos e sintetiza uma resposta em portugues
12. A resposta e enviada em streaming (SSE) para o frontend
13. O frontend renderiza a resposta com as fontes colapsaveis mostrando quais documentos foram usados

### 5.4 O que o usuario ve

- Enquanto busca: "Buscando no banco de dados..." com spinner
- Depois da busca: "X documentos recuperados em Yms" com badges de tipo de evento e mapa
- Resposta: Texto gerado pelo LLM, fundamentado nos dados, em portugues
- Fontes: Lista colapsavel com score de relevancia, tipo de evento e preview do documento

---

## 6. A API FastAPI

Corresponde ao Sprint 7 — "API (AC2)" do AI_Project.pdf, que pede endpoints `/query` e `/metadata` com validacao de input/output. O backend expoe tres endpoints:

### GET /health
Health check simples. Retorna `{"status": "ok"}`.

### POST /search — Busca Vetorial Pura
Faz busca de similaridade no pgvector e retorna os documentos mais relevantes SEM sintese de LLM.

**Parametros**:
- `query` (obrigatorio): Texto da busca
- `embedding_run_id` (opcional): Qual run de embeddings consultar
- `top_k` (padrao: 5): Quantos resultados retornar
- `event_type` (opcional): Filtro por tipo de evento (damage, kill, grenade, round_meta)
- `map_name` (opcional): Filtro por mapa
- `file_name` (opcional): Filtro por arquivo fonte
- `round_number` (opcional): Filtro por numero do round

**Resposta**: Lista de documentos com rank, score de relevancia, doc_id, texto completo, metadata e tempo de execucao em milissegundos.

### POST /rag/query — RAG Completo
Faz busca vetorial + sintese de LLM. Suporta streaming via SSE.

**Parametros adicionais**:
- `stream` (padrao: true): Se true, retorna Server-Sent Events
- `llm_key` (opcional): Qual LLM usar (ex: "ollama/qwen2.5", "gpt-4o", "claude-sonnet")

**Resposta**: Em modo JSON, retorna `query`, `answer`, `sources`, `retrieval_ms`, `generation_ms`. Em modo stream, retorna SSE com chunks de texto.

---

## 7. O Frontend Next.js

O AI_Project.pdf sugere Gradio ou frontend simples (Sprint 8 — "Interface"). O grupo optou por um frontend mais completo com Next.js 16 porque: (1) permite streaming real via SSE com melhor UX, (2) o AI SDK da Vercel tem integracao nativa com tool calling que o Gradio nao tem, (3) permite persistencia de sessoes de chat, e (4) demonstra capacidade de engenharia de software full-stack.

### Tecnologias
- **Next.js 16** com App Router e React 19
- **AI SDK** (Vercel) para streaming de texto e integracao com tools
- **ollama-ai-provider-v2** para comunicacao direta com Ollama
- **better-sqlite3** para persistencia de sessoes de chat
- **Tailwind CSS v4** + Radix UI para interface
- **Streamdown** para renderizacao de Markdown com suporte a codigo, math e mermaid

### Interface do Chat
- **Sidebar esquerda**: Lista de sessoes de chat, criacao de nova conversa, exclusao
- **Area principal**: Mensagens com Markdown, blocos de "pensamento" (para modelos com reasoning), e visualizacao de tool calls em tempo real
- **Compositor**: Textarea com selecao de modelo e toggle de modo RAG

### Modelos Disponiveis no Chat
- **Qwen 2.5 7B** — Modelo padrao, suporta tools (RAG funciona), sem reasoning visivel
- **Qwen3 8B** — Suporta tools E reasoning (mostra o processo de pensamento do modelo)
- **DeepSeek R1 8B** — Suporta reasoning, mas NAO suporta tools (RAG fica desabilitado)

### Modos RAG
- **Auto**: O modelo decide se precisa buscar no banco (toolChoice: "auto")
- **RAG**: Sempre busca no banco antes de responder (toolChoice: "required")
- **Off**: Sem busca, apenas conhecimento geral do modelo (sem tools)

### Persistencia de Sessoes
Sessoes de chat sao salvas em SQLite local com WAL mode. Cada sessao tem ID, titulo e timestamp. Mensagens sao salvas com role (user/assistant), parts (JSON com texto, reasoning e tool calls) e timestamp. O usuario pode fechar o navegador e voltar com todo o historico intacto.

---

## 8. ProviderRegistry — Abstracao de Provedores

O `ProviderRegistry` e uma fabrica lazy-loading que abstrai a criacao de instancias de LLM e embedding. Permite trocar de provedor mudando apenas variaveis de ambiente, sem alterar codigo.

### Provedores de LLM suportados
- **Ollama** (padrao e fallback): `ollama/qwen2.5:7b-instruct-q4_K_M` — Roda localmente, sem API key
- **OpenAI** (se `OPENAI_API_KEY` configurada): `gpt-4o`
- **Anthropic** (se `ANTHROPIC_API_KEY` configurada): `claude-sonnet`

### Provedores de Embedding suportados
- **Ollama** (padrao e fallback): `nomic-embed-text` — 768 dimensoes, roda localmente
- **OpenAI** (se key configurada): `text-embedding-3-small` — 768 dimensoes
- **Voyage** (se `VOYAGE_API_KEY` configurada): `voyage-3`

### Ollama como provedor padrao
O projeto foi desenhado para funcionar completamente offline, conforme o AI_Project.pdf pede ("Inferencia com LLM local (Ollama)"). O Qwen 2.5 7B consome cerca de 12GB de RAM e o nomic-embed-text cerca de 300MB. Qualquer integrante do grupo pode rodar o sistema inteiro na propria maquina sem gastar nada com APIs.

### Mecanismo de fallback
Se o `ProviderRegistry` tentar instanciar um provedor e falhar (API key invalida, rede fora, etc.), ele automaticamente cai para o Ollama local e loga um warning. O sistema nunca para por causa de um provedor indisponivel.

---

## 9. Infraestrutura

### Docker Compose
Corresponde ao entregavel obrigatorio 1 do AI_Project.pdf ("Docker Compose funcional"). Tudo roda via Docker Compose com tres perfis:

**Servicos principais** (sempre ativos):
- **MinIO**: Object storage compativel com S3. Armazena todos os dados do data lake (Bronze, Silver, Gold, Documents). Portas 9000 (API) e 9001 (console web).
- **TimescaleDB**: PostgreSQL com extensoes pgvector e TimescaleDB. Armazena embeddings, metadata de runs e catalogo de dados. Porta 54330.
- **Ollama**: Servidor de inferencia local. Roda LLMs e modelos de embedding. Porta 11434.
- **RAG API**: FastAPI servindo os endpoints de busca e RAG. Porta 8000.

**Jobs de dados** (perfil `jobs`, rodam sob demanda):
- `bronze-importer`, `silver-transformer`, `gold-transformer`, `document-builder`, `embedding-ingestor`
- Cada job e um container efemero que roda a transformacao e sai

**Frontend** (sempre ativo):
- **Next.js**: Frontend containerizado em modo standalone, porta 3002, depende do rag-api. Dockerfile multi-stage com Node 24.

**Observabilidade** (perfil `observability`, opcional):
- **OTEL Collector**: Recebe traces e metricas via OpenTelemetry Protocol (porta 4318)
- **Jaeger**: Visualizacao de traces distribuidos (porta 16686)
- **Prometheus**: Coleta e armazenamento de metricas (porta 9090)
- **Grafana**: Dashboards (porta 3000)

**MLflow** (perfil `mlflow`, opcional):
- **mlflow-db-init**: Job efemero que cria o database `mlflow` no TimescaleDB se nao existir
- **MLflow server**: Imagem oficial v2.22.4 + psycopg2, porta 5000, backend no PostgreSQL, artefatos em volume Docker

### Makefile
Corresponde ao entregavel obrigatorio 2 do AI_Project.pdf ("Makefile com comandos padronizados") e Sprint 10 ("Automacao"). Todos os comandos do projeto sao acessiveis via `make`:

- `make up` / `make down` — Sobe/desce os servicos
- `make bronze` / `make silver` / `make gold` / `make documents` / `make embeddings` — Pipeline de dados completo
- `make api` / `make frontend` — Desenvolvimento local
- `make ci` / `make ci-frontend` / `make ci-all` — CI local (lint + types + tests)
- `make otel-up` / `make otel-down` — Observabilidade
- `make ollama-pull` — Baixa modelos do Ollama
- Smoke tests: `make documents-smoke`, `make embeddings-smoke` — Execucoes rapidas com poucos registros para validacao

---

## 10. Observabilidade — OTEL, Jaeger, Prometheus, Grafana

### Como funciona
O backend FastAPI e instrumentado com OpenTelemetry. Cada request HTTP gera um trace com spans para: recebimento da request, geracao do embedding da query, busca no pgvector, chamada ao LLM (quando aplicavel), envio da resposta.

Logs estruturados via structlog incluem automaticamente `trace_id` e `span_id` em cada linha, permitindo correlacionar logs com traces.

### Pipeline de telemetria
1. Aplicacao FastAPI exporta traces via OTLP HTTP
2. OTEL Collector recebe, processa e distribui
3. Jaeger armazena e visualiza traces
4. Connector spanmetrics converte traces em metricas
5. Prometheus coleta metricas
6. Grafana exibe dashboards

### Auto-deteccao de formato de log
O sistema detecta automaticamente se esta rodando em terminal ou container. Em terminal, usa logs coloridos legiveis. Em container (Docker/CI), usa JSON estruturado para ser parseado por ferramentas.

---

## 11. Rastreabilidade e Auditoria (Governanca de Dados)

Governanca de dados e 15% da nota segundo os criterios de avaliacao do AI_Project.pdf. O sistema implementa rastreabilidade completa do pipeline.

### O catalogo dataset_runs

Toda execucao de cada estagio do pipeline registra uma entrada na tabela `dataset_runs` do PostgreSQL com:
- `run_id`: Identificador unico da execucao (timestamp ISO)
- `stage`: Qual estagio (bronze, silver, gold, documents, embeddings)
- `source_run_id`: De qual run do estagio anterior essa execucao depende
- `dataset_prefix`: Prefixo do dataset
- `files_processed`, `rows_read`, `rows_output`: Contadores
- `quality_summary`: JSONB com metricas detalhadas
- Caminhos dos artefatos gerados (events_key, manifest_key, quality_report_key)

### Cadeia de linhagem

Cada estagio aponta para o run do estagio anterior via `source_run_id`:
```
embeddings:run5 -> documents:run4 -> gold:run3 -> silver:run2 -> bronze:run1
```

Isso permite reconstruir toda a linhagem de qualquer embedding ate o dado bruto original do Kaggle.

### CLI de auditoria

O comando `run-audit` reconstroi a cadeia de linhagem e verifica integridade:
```bash
run-audit --stage embeddings --run-id 20260308T180500Z --format text
```

Verifica: se cada run na cadeia existe, se os `dataset_prefix` sao consistentes, se existem gaps ou ciclos, se os artefatos estao referenciados.

### Quality Reports

Cada estagio gera um `quality_report.json` no MinIO com metricas:
- **Silver**: Linhas lidas, saidas, duplicatas removidas, invalidas encontradas
- **Gold**: Linhas por tipo de evento, rejeitadas por validacao
- **Documents**: Documentos gerados por tier, contadores por tipo de evento
- **Embeddings**: Documentos indexados, modelo usado, configuracao de batch, progresso de resume

---

## 12. CI/CD

### Backend (Python)
- **Ruff**: Linter e formatter. Garante estilo consistente
- **Pyright**: Type checker estatico. Todas as funcoes tem type hints
- **Pytest**: Testes unitarios e de integracao

### Frontend (Next.js)
- **ESLint**: Linting para TypeScript/React
- **next build**: Compilacao de producao

### Comandos
```bash
make ci           # Backend: lint + types + tests
make ci-frontend  # Frontend: lint + build
make ci-all       # Tudo junto
```

CI roda automaticamente via GitHub Actions em cada push.

---

## 13. O que esta implementado vs. o que falta

### Implementado (cobrindo Sprints 1-3, 5-8, 10 do AI_Project.pdf)
- Pipeline Medallion completo: Bronze -> Silver -> Gold -> Documents -> Embeddings
- Banco vetorial com pgvector + metadata JSONB
- RAG funcionando ponta a ponta (busca vetorial + sintese LLM + streaming)
- API FastAPI com `/search` e `/rag/query`
- Frontend Next.js com chat interativo, selecao de modelo, modo RAG
- Docker Compose com todos os servicos
- Makefile com comandos padronizados
- Rastreabilidade completa (catalogo dataset_runs, CLI de auditoria, quality reports)
- Observabilidade (OTEL -> Jaeger + Prometheus + Grafana)
- MLflow server containerizado (infraestrutura pronta, faltam scripts de treinamento)
- Frontend containerizado (Dockerfile multi-stage, standalone mode)
- CI automatizado (Ruff + Pyright + Pytest + ESLint + next build)
- Persistencia de sessoes de chat em SQLite

### Parcialmente implementado (Sprints 4 e 9 do AI_Project.pdf)
- **MLflow server**: A infraestrutura do MLflow ja esta no docker-compose (`--profile mlflow`). O servidor roda na porta 5000, usa o mesmo TimescaleDB como backend (database `mlflow` separada) e armazena artefatos em volume Docker. O que falta sao os scripts de treinamento e o registro de experimentos.
- **Treinamento de modelos ML**: Definicao de problema ML sobre os dados (classificacao, regressao ou serie temporal), selecao de modelos (Linear, Logistica, LSTM, XGBoost, TCN conforme sugerido no AI_Project.pdf), scripts de treinamento, registro de experimentos no MLflow. Esses resultados de treinamento futuramente tambem seriam transformados em documentos para enriquecer o RAG.

---

## 14. Decisoes Tecnicas e Justificativas

O entregavel obrigatorio 8 do AI_Project.pdf pede "justificativa das decisoes tecnicas". Estas sao as principais divergencias da stack de referencia sugerida no AI_Project.pdf e por que foram feitas:

### pgvector em vez de Milvus
O AI_Project.pdf lista Milvus como banco vetorial de referencia (slide "Arquitetura Exemplo"). O grupo optou por pgvector (extensao do PostgreSQL) porque: (1) elimina um servico inteiro do docker-compose — o PostgreSQL que ja era necessario para metadados e catalogo passa a servir tambem como banco vetorial, (2) simplifica operacoes — um unico banco para queries relacionais e vetoriais, (3) metadata JSONB permite filtros combinados (ex: buscar por similaridade E filtrar por mapa e tipo de evento na mesma query), (4) TimescaleDB adiciona otimizacoes para o tipo de dado que temos. O trade-off e que Milvus escala melhor para bilhoes de vetores, mas o volume do projeto (milhares de documentos) nao justifica essa complexidade.

### Next.js em vez de Gradio
O AI_Project.pdf sugere "Gradio ou frontend simples" (Sprint 8). O grupo optou por Next.js 16 porque: (1) o AI SDK da Vercel permite streaming real de respostas via SSE com feedback visual em tempo real (spinner de busca, contagem de documentos, rendering incremental da resposta), (2) tool calling nativo — o modelo pode decidir quando buscar e o frontend renderiza isso como widget interativo com fontes colapsaveis, (3) persistencia de sessoes — o usuario nao perde o historico ao recarregar, (4) multiplos modelos selecionaveis na interface, (5) demonstra capacidade de engenharia full-stack. Gradio seria mais rapido de implementar mas resultaria em UX significativamente inferior para demonstracao.

### Dimensao fixa de 768 para embeddings
Tanto o `nomic-embed-text` (Ollama) quanto o `text-embedding-3-small` (OpenAI, com dim=768) geram vetores de mesma dimensao. Isso permite trocar de provedor de embedding sem reconstruir a tabela ou reindexar.

### 5 estagios no pipeline em vez de 3
A Medallion classica tem 3 camadas (Bronze/Silver/Gold). O projeto adicionou Documents e Embeddings como estagios formais do pipeline porque: (1) a geracao de documentos de texto a partir da Gold e uma transformacao substancial (agregacao em 5 tiers, templates em portugues), (2) a geracao de embeddings e um processo longo e custoso que precisa de resume, (3) cada estagio adicional tem seu proprio quality report e registro no catalogo, (4) isso permite reprocessar embeddings sem regerar documentos, ou regerar documentos sem reprocessar a Gold.

### Agregacao em 5 tiers em vez de 1 documento por evento
Com milhoes de eventos, embeddar cada um individualmente geraria documentos quase identicos e tornaria a busca vetorial inutil (tudo teria score similar). A agregacao hierarquica produz documentos semanticamente ricos e distintos, o que melhora drasticamente a qualidade da busca e consequentemente a qualidade das respostas do RAG.

### AI SDK para sintese no frontend, LlamaIndex para embedding/busca no backend
A divisao de responsabilidades: LlamaIndex e superior para pipeline de embedding e busca vetorial (IngestionPipeline, VectorStoreIndex). AI SDK e superior para streaming, tool calling e integracao com React. Cada ferramenta e usada onde e mais forte. O LLM nao e chamado pelo LlamaIndex — a sintese acontece no frontend via AI SDK + Ollama.

### Sem Redis, sem worker separado, sem message broker
O professor nao pediu essas tecnologias e elas adicionariam complexidade sem beneficio para o escopo do projeto. O pipeline de dados roda como jobs efemeros. A API e sincrona (com streaming). Nao ha necessidade de filas ou cache distribuido.

---

## 15. Stack Tecnologica Completa

### Backend
- **Python 3.13+** com type hints completos
- **FastAPI** para API HTTP com streaming SSE
- **LlamaIndex** para IngestionPipeline (embeddings) e VectorStoreIndex (busca)
- **MinIO SDK** para operacoes no data lake
- **Psycopg2 + SQLAlchemy** para PostgreSQL
- **structlog** para logging estruturado
- **OpenTelemetry** para traces e metricas
- **Pydantic** para validacao de configuracao

### Frontend
- **Next.js 16** com App Router e React 19
- **AI SDK** (Vercel) para streaming, tools e integracao com LLM
- **ollama-ai-provider-v2** para comunicacao com Ollama
- **better-sqlite3** para persistencia local
- **Tailwind CSS v4** + Radix UI para interface
- **Streamdown** para renderizacao de Markdown

### Infraestrutura
- **Docker Compose** para orquestracao
- **MinIO** como data lake (S3-compatible)
- **TimescaleDB** com pgvector para banco relacional + vetorial
- **Ollama** para inferencia local de LLM e embeddings
- **Jaeger + Prometheus + Grafana** para observabilidade

### Modelos de IA (Locais via Ollama)
- **Qwen 2.5 7B Instruct** (Q4_K_M) — LLM principal, ~12GB RAM, suporta tool calling
- **Qwen3 8B** — LLM com reasoning + tool calling
- **DeepSeek R1 8B** — LLM com reasoning (sem tool support)
- **nomic-embed-text** — Modelo de embedding, 768 dimensoes, ~300MB RAM
