# Silver — Limpeza e Normalizacao

A camada Silver e o segundo estagio da Medallion Architecture. Le os CSVs da Bronze e aplica transformacoes de qualidade para garantir dados consistentes downstream.

## Transformacoes aplicadas

- **Normalizacao de colunas**: Lowercase, remocao de caracteres especiais, deduplicacao de nomes de colunas.
- **Validacao numerica**: Rejeita valores negativos, NaN e invalidos em campos numericos.
- **Deduplicacao**: Remove linhas completamente duplicadas.
- **Filtragem de nulos**: Remove linhas onde todos os campos sao nulos.

## Por que existe a Silver

Dados brutos vem com inconsistencias — colunas com nomes diferentes entre arquivos, valores faltantes, duplicatas. A Silver garante que todos os dados downstream tem formato consistente e confiavel. Sem essa camada, cada transformacao posterior teria que lidar com essas inconsistencias individualmente.

## Metricas de qualidade

Cada execucao registra quantas linhas leu, quantas saiu, quantas duplicatas removeu, quantas linhas invalidas encontrou. Isso fica no `quality_report.json` no MinIO e no catalogo `dataset_runs` do PostgreSQL.

## Saida

CSVs limpos em `silver/<dataset_prefix>/<run_id>/cleaned/`.

## Comando

```bash
silver-transform  # CLI direto
make silver       # Via Makefile
```
