# Model Card — Recomendação User-Item (TC02)

> Última atualização: 2026-07-31  
> Domínio da plataforma: `recommendation`  
> Modelo Registry: `tc02_recommender`

---

## 1. Identificação

| Campo | Valor |
|-------|-------|
| Nome do modelo | `tc02_recommender` |
| Domínio | `recommendation` |
| Tipo de problema | Recomendação user-item / ranking top-K |
| Modelo principal | `torch_embedding` |
| Arquitetura | Embeddings de usuário e item + MLP scorer em PyTorch |
| Baselines suportados | `popularity`, `nmf_sklearn` |
| Dataset de desenvolvimento | MovieLens `ml-latest-small` |
| Pipeline reprodutível | DVC: `preprocess -> feature_eng -> train -> evaluate` |
| Experimento MLflow | `tc02_recommendation_dvc` no DVC; `tc02_recommendation` nos exemplos de demo/API |
| Serving | FastAPI: `POST /v1/domains/recommendation/predict` |
| Fonte de verdade do serving | PostgreSQL `deployed_models` |
| Registry | MLflow Model Registry como side-effect do treino/promote |

---

## 2. Uso Pretendido

O modelo ranqueia itens para um `user_id` com base no histórico de interações do usuário e do catálogo observado no treino.

Uso esperado:

- Recomendar uma lista top-K de itens para usuários conhecidos.
- Apoiar uma experiência de vitrine, sugestão de produtos ou recomendação de conteúdo.
- Demonstrar o ciclo MLOps completo do TC02: treino, registro, promote, inferência e rollback.

Consumidores previstos:

- API síncrona via `/v1/domains/recommendation/predict`.
- Serviços downstream que precisem de `recommended_items`.
- Equipe de engenharia/ML durante validação do pipeline.

Fora de escopo:

- Recomendação causal ou uplift de conversão.
- Otimização direta de receita, margem, estoque ou diversidade.
- Recomendação para catálogo dinâmico em tempo real.
- Recomendação com dados textuais, imagens, sessão de navegação ou contexto temporal avançado.

---

## 3. Dados

### Fonte

O desenvolvimento usa MovieLens `ml-latest-small`, tratado como proxy acadêmico para um catálogo de e-commerce.

Arquivo esperado:

```text
data/recommendation/raw/ratings.csv
```

Colunas usadas:

| Coluna original | Coluna interna | Uso |
|-----------------|----------------|-----|
| `userId` | `user_id` | Identificador do usuário |
| `movieId` | `item_id` | Identificador do item |
| `rating` | `rating` | Preferência explícita |
| `timestamp` | `timestamp` | Ordenação temporal por usuário |

### Preparação

O estágio `preprocess` normaliza nomes, tipos e remove nulos.

O estágio `feature_eng`:

- Ordena interações por `user_id` e `timestamp`.
- Define interações positivas com `rating >= min_positive_rating`.
- Usa a última interação positiva de cada usuário como verdade de teste.
- Mantém as demais interações como treino.
- Gera:
  - `data/recommendation/features/train.parquet`
  - `data/recommendation/features/test_truth.json`

Configuração atual relevante em `params.yaml`:

| Parâmetro | Valor |
|-----------|-------|
| `random_state` | `42` |
| `top_k` | `10` |
| `min_interactions` | `10000` |
| `train_models` | `popularity`, `nmf`, `torch_embedding` |
| `embedding_dim` | `32` |
| `hidden_dim` | `128` |
| `dropout` | `0.1` |
| `use_item_bias` | `true` |
| `n_epochs` | `10` |
| `batch_size` | `512` |
| `lr` | `0.01` |
| `n_negatives` | `4` |
| `validation_fraction` | `0.1` |
| `early_stopping_patience` | `3` |
| `min_positive_rating` | `4.0` |
| `champion_model` | `torch_embedding` |

---

## 4. Modelo

### Baselines

| Modelo | Descrição |
|--------|-----------|
| `popularity` | Recomenda itens mais frequentes no treino, excluindo itens já vistos quando possível. |
| `nmf_sklearn` | Matriz usuário-item com NMF (`sklearn.decomposition.NMF`) e ranking por score reconstruído. |

### Modelo principal — `torch_embedding`

O modelo principal usa embeddings de usuário e item em PyTorch.

Arquitetura atual:

```text
user_id -> user_embedding
item_id -> item_embedding
[user_emb, item_emb, user_emb * item_emb]
        -> Linear(3 * embedding_dim, hidden_dim)
        -> ReLU
        -> Dropout
        -> Linear(hidden_dim, 1)
        -> score/logit
        + item_bias opcional
```

Treino:

- Loss: `BCEWithLogitsLoss`.
- Otimizador: `Adam`.
- Positivos: pares com `rating >= min_positive_rating`.
- Negativos:
  - amostragem negativa por usuário (`n_negatives`);
  - interações explícitas abaixo do threshold entram como negativos.
- Validação interna: fração dos pares positivos (`validation_fraction`).
- Early stopping por `val_loss`.
- Semente fixa (`random_state=42`) para reprodutibilidade.

Artefatos:

```text
models/recommendation/torch_embedding.pt
models/recommendation/torch_embedding.meta.json
models/recommendation/champion_manifest.json
```

O `meta.json` persiste mapas usuário/item, hiperparâmetros, itens populares e resumo de treino. O manifest aponta o prefixo do artefato usado pela engine de inferência.

---

## 5. Avaliação

O problema é avaliado como ranking top-K, não como classificação binária simples.

Métricas implementadas:

| Métrica | Interpretação |
|---------|---------------|
| `hit_rate` | Percentual de usuários em que pelo menos um item relevante apareceu no top-K. |
| `precision_at_k` | Fração do top-K que é relevante. |
| `recall_at_k` | Fração dos itens relevantes recuperada no top-K. |
| `ndcg_at_k` | Qualidade da posição dos itens relevantes no ranking. |
| `map_at_k` | Média da precisão acumulada nas posições relevantes. |

Seleção de campeão:

- O vencedor métrico é o maior `ndcg_at_k`.
- `champion_model` em `params.yaml` pode forçar um modelo para produção acadêmica/operacional.
- A configuração atual define `champion_model=torch_embedding`, mesmo quando outro baseline vence
  metricamente, porque o requisito central do TC02 é servir uma rede neural PyTorch.

### Resultado Local Atual

Métricas lidas dos artefatos locais atuais:

```text
reports/recommendation/metrics.json
models/recommendation/champion_manifest.json
```

| Campo | Valor |
|-------|-------|
| Campeão | `torch_embedding` |
| Vencedor métrico | `nmf_sklearn` |
| Seleção | `champion_model` configurado para produção acadêmica/operacional |
| K no manifest local | `10` |
| `hit_rate` | `0.0558` |
| `precision_at_k` | `0.0056` |
| `recall_at_k` | `0.0558` |
| `ndcg_at_k` | `0.0301` |
| `map_at_k` | `0.0223` |

Comparativo offline atual:

| Modelo | Hit Rate@K | Precision@K | Recall@K | NDCG@K | MAP@K |
|--------|------------|-------------|----------|--------|-------|
| `popularity` | `0.0279` | `0.0028` | `0.0279` | `0.0138` | `0.0095` |
| `nmf_sklearn` | `0.0920` | `0.0092` | `0.0920` | `0.0420` | `0.0272` |
| `torch_embedding` | `0.0558` | `0.0056` | `0.0558` | `0.0301` | `0.0223` |

Observação de governança: `nmf_sklearn` é o melhor baseline offline neste snapshot, mas
`torch_embedding` permanece como campeão para demonstrar e operar o modelo neural exigido
pelo desafio. Em produção real, a decisão deveria ser revista com métricas online e critérios
de negócio.

Resumo do treino no artefato local:

| Campo | Valor |
|-------|-------|
| `best_epoch` | `1` |
| `best_val_loss` | `0.3325` |
| Interações positivas usadas | `47971` |
| Interações negativas explícitas | `52256` |
| Interações de treino | `43174` |
| Interações de validação | `4797` |

Observação: antes de uma entrega final, regenere estes números com:

```bash
make tc02-repro
cat reports/recommendation/metrics.json | python3 -m json.tool
```

---

## 6. Serving e Contrato de API

Endpoint:

```text
POST /v1/domains/recommendation/predict
```

Payload:

```json
{
  "user_id": 1,
  "top_k": 5
}
```

Resposta esperada:

```json
{
  "domain": "recommendation",
  "recommended_items": [1272, 1209, 1250, 3741, 246],
  "pipeline_run_id": 1
}
```

Fluxo de serving:

1. API lê o deployment ativo em `deployed_models`.
2. Carrega o manifest do `pipeline_run`.
3. Resolve o prefixo dos artefatos no volume compartilhado.
4. Carrega `TorchEmbeddingRecommender`.
5. Gera top-K e grava auditoria em `predictions`.

Cold-start:

- Usuários desconhecidos usam fallback de popularidade.
- Itens novos não são recomendados até entrarem em novo treino.

---

## 7. MLOps, Governança e Promoção

### Pipeline

DVC:

```text
preprocess -> feature_eng -> train -> evaluate
```

Airflow/API:

- DAG canônica: `ml_training_dispatch`.
- Ramo recommendation: Airflow chama `worker_recommendation`.
- Rotas:
  - `POST /v1/domains/recommendation/admin/train/sync`
  - `POST /v1/domains/recommendation/admin/train/trigger`
  - `GET /v1/domains/recommendation/admin/runs`
  - `POST /v1/domains/recommendation/admin/promote`
  - `POST /v1/domains/recommendation/admin/rollback`

### Tracking

MLflow registra:

- parâmetros do run;
- métricas top-K;
- manifest do campeão;
- artefatos `.pt` e `.meta.json`;
- modelo PyTorch em `pytorch_model`, quando compatível com a versão do servidor.

### Promoção

O promote:

- exige run `recommendation` concluído e ativo;
- grava deployment ativo em `deployed_models`;
- arquiva deployment anterior;
- tenta sincronizar `tc02_recommender` no MLflow Registry como side-effect.

A fonte de verdade do `/predict` é `deployed_models`, não o Registry.

---

## 8. Limitações

| Limitação | Impacto |
|-----------|---------|
| MovieLens não é e-commerce real | Métricas e comportamento podem não transferir para catálogo de produtos. |
| Sem metadados de item | O modelo não usa gênero, preço, categoria, texto, imagem ou disponibilidade. |
| Cold-start parcial | Usuários desconhecidos caem em popularidade; itens novos não aparecem. |
| Split temporal simplificado | Usa última interação positiva por usuário, não simula janelas reais de produção. |
| Popularity bias | Itens populares tendem a dominar, principalmente para usuários com pouco histórico. |
| Métricas offline | `hit_rate`/`ndcg` não medem conversão real, satisfação ou receita. |
| Sem diversidade/novidade | O ranking otimiza relevância histórica, não diversidade ou descoberta. |
| Sem filtro de regras de negócio | Não considera estoque, margem, compliance ou disponibilidade. |

---

## 9. Riscos e Vieses

Riscos principais:

- Reforço de popularidade: itens já populares recebem mais exposição.
- Baixa cobertura de cauda longa.
- Recomendações pouco personalizadas para usuários com poucas interações.
- Generalização limitada para catálogo real.
- Otimização offline desalinhada de objetivos de negócio.

Mitigações recomendadas:

- Monitorar cobertura de catálogo e concentração de recomendações.
- Acompanhar métricas por coorte de usuários novos vs recorrentes.
- Introduzir regras de diversidade e filtros de elegibilidade.
- Validar online com A/B test antes de decisões comerciais.
- Calibrar objetivos com métricas de negócio, como CTR, conversão, receita e retenção.

---

## 10. Monitoramento Recomendado

### Offline

- `hit_rate@K`, `ndcg@K`, `map@K` por retreino.
- Cobertura de catálogo: percentual de itens recomendados ao menos uma vez.
- Concentração: participação dos top-N itens nas recomendações.
- Percentual de usuários atendidos por fallback de popularidade.

### Online

Quando houver ambiente real:

- CTR por posição.
- Conversão pós-clique.
- Receita por sessão.
- Taxa de rejeição/ocultação.
- Latência p95 de `/predict`.
- Erros 4xx/5xx por endpoint.

---

## 11. Reprodutibilidade

Executar pipeline DVC:

```bash
make tc02-repro
```

Ver métricas:

```bash
cat reports/recommendation/metrics.json | python3 -m json.tool
dvc metrics show
```

Treino rápido via API:

```bash
curl -s -X POST "$API/v1/domains/recommendation/admin/train/sync" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"train_models":["torch_embedding"],"n_epochs":1,"top_k":5,"mlflow_experiment":"tc02_recommendation"}' \
  | python3 -m json.tool
```

Promote:

```bash
curl -s -X POST "$API/v1/domains/recommendation/admin/promote" \
  -H "Authorization: Bearer $TOKEN" \
  | python3 -m json.tool
```

Predict:

```bash
curl -s -X POST "$API/v1/domains/recommendation/predict" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"user_id":1,"top_k":5}' \
  | python3 -m json.tool
```

---

## 12. Aprovação Para Uso

Critérios mínimos antes de usar em um cenário real:

- `make tc02-repro` concluído sem erro.
- Métricas top-K registradas e revisadas.
- Smoke de `/predict` com deployment ativo.
- MLflow com artefatos e parâmetros do run.
- Rollback testado.
- Definição de métricas online e plano de A/B test.
- Revisão de privacidade para IDs de usuário e logs de predição.

Status atual: adequado para demonstração acadêmica e validação de arquitetura MLOps; não aprovado para decisão comercial sem validação online.
