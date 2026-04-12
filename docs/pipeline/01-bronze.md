# Bronze — Ingestao Bruta

A camada Bronze e o primeiro estagio da Medallion Architecture. Ela baixa o dataset `csgo-matchmaking-damage` do Kaggle como ZIP, extrai os CSVs e imagens, e armazena tudo no MinIO sem nenhuma transformacao.

## Por que existe a Bronze

A camada Bronze preserva o dado exatamente como veio da fonte. Se algo der errado nas transformacoes posteriores, sempre e possivel voltar ao dado original. E o principio de "dados imutaveis na origem" — fundamental para governanca de dados.

## Dados contidos nos CSVs

- **damage.csv**: Cada evento de dano — quem atirou, quem recebeu, arma, dano HP, dano de armadura, posicoes XY, hitbox, lado (CT/T), tick, segundos.
- **kills.csv**: Cada kill registrada — arma, posicoes, lados, tick.
- **grenades.csv**: Cada granada lancada — tipo, posicao de lancamento e impacto.
- **meta.csv**: Metadados de rounds — mapa, vencedor, economia, jogadores vivos.

## Saida

Arquivos em `bronze/<dataset_prefix>/<run_id>/raw/` (ZIP original) e `bronze/<dataset_prefix>/<run_id>/extracted/` (CSVs extraidos).

## Comando

```bash
bronze-import   # CLI direto
make bronze     # Via Makefile
```

## Rastreabilidade

Cada execucao registra uma entrada na tabela `dataset_runs` do PostgreSQL com run_id, stage="bronze", contadores de arquivos processados e linhas lidas.
