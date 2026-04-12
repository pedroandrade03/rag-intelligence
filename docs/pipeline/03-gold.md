# Gold — Curacao e Padronizacao de Schema

A camada Gold e o terceiro estagio da Medallion Architecture. Le os CSVs limpos da Silver e os transforma em um unico arquivo `events.csv` com schema padronizado de 40 colunas.

## Transformacoes aplicadas

- **Inferencia de tipo de evento**: A partir do nome do arquivo e das colunas disponiveis, classifica cada linha como `damage`, `kill`, `grenade`, `round_meta` ou `event`.
- **Projecao para schema fixo**: Todas as linhas sao mapeadas para as mesmas 40 colunas Gold, independente do CSV de origem.
- **Validacao de campos obrigatorios**: Cada tipo de evento tem seus campos minimos — kills precisam de arma, damage precisa de hp_dmg ou arm_dmg, round_meta precisa de mapa.
- **Descarte de dados inuteis**: Linhas de `map_layout` sao removidas (dados de layout de mapa nao sao uteis para analise de combate).

## Por que existe a Gold

Sem a Gold, o sistema teria que lidar com schemas diferentes para cada tipo de arquivo. A Gold cria um "contrato de dados" — qualquer coisa downstream pode contar com exatamente essas 40 colunas, nesses tipos, com essas validacoes ja aplicadas. E a camada que define os "dados prontos para consumo" da Medallion Architecture.

## Campos Gold (40 colunas)

### Base
`file`, `round`, `map`, `weapon`, `hp_dmg`, `arm_dmg`, `att_pos_x`, `att_pos_y`, `vic_pos_x`, `vic_pos_y`

### Extras
`event_type`, `source_file`, `tick`, `seconds`, `att_team`, `vic_team`, `att_side`, `vic_side`, `wp_type`, `nade`, `hitbox`, `bomb_site`, `is_bomb_planted`, `att_id`, `vic_id`, `att_rank`, `vic_rank`, `winner_team`, `winner_side`, `round_type`, `ct_eq_val`, `t_eq_val`, `ct_alive`, `t_alive`, `nade_land_x`, `nade_land_y`, `avg_match_rank`, `start_seconds`, `end_seconds`

## Saida

`gold/<dataset_prefix>/<run_id>/curated/events.csv`

## Comando

```bash
gold-transform  # CLI direto
make gold       # Via Makefile
```
