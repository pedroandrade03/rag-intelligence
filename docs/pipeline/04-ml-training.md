# ML Training — Predicao de Vencedor do Proximo Round

O estagio de treinamento de ML usa os dados curados da camada Gold para treinar modelos de classificacao que predizem qual lado (CT ou T) vencera o proximo round, dado o contexto do round atual.

## Modelos treinados

### Baseline (repeat current winner)
Predicao naive: repete o vencedor do round atual como predicao do proximo. Serve como baseline para comparacao — qualquer modelo util deve superar essa heuristica.

### Logistic Regression
Regressao logistica com class_weight="balanced" e max_iter=2000. Modelo linear que serve como referencia interpretavel. Combinado com preprocessamento via ColumnTransformer que aplica StandardScaler para features numericas e OneHotEncoder para categoricas.

### Histogram Gradient Boosting
HistGradientBoostingClassifier com learning_rate=0.05, max_depth=6, max_iter=350. Modelo ensemble nao-linear que tipicamente supera a regressao logistica em dados tabulares.

## Features utilizadas

### Numericas
Economia (ct_eq_val, t_eq_val, eq_diff), contexto do round (round_number, overtime_flag), combate (total_hp_dmg, total_arm_dmg, kills_round, bomb_planted_any, ct_alive_end, t_alive_end, alive_diff_end), e features temporais (lags de 1-3 rounds e medias moveis).

### Categoricas
Mapa (map), tipo de round (round_type), metade (half), e lado vencedor atual (winner_side_norm).

## Validacao

Split por grupo (GroupShuffleSplit por arquivo de partida) para evitar data leakage entre rounds da mesma partida. Metricas: ROC-AUC, F1, balanced accuracy, log loss, Brier score. Metricas segmentadas por mapa e por metade (H1/H2).

## Feature Importances

Importancias calculadas via permutation importance (sklearn.inspection.permutation_importance) em X_test/y_test para consistencia entre modelos. Armazenadas na tabela training_runs do PostgreSQL para consulta via busca lexical.

## Experiment Tracking

Resultados logados no MLflow com metricas, parametros e artefatos do modelo. Cada modelo gera um run separado no MLflow.

## Comandos

```bash
train-logreg    # Treina regressao logistica
train-histgbt   # Treina histogram gradient boosting
train-baseline  # Calcula baseline
make train-logreg / make train-histgbt / make train-baseline  # Via Makefile
```
