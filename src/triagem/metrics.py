"""Métricas Prometheus da API, expostas em /metrics.

Cada combinação de valores de rótulo vira uma série no Prometheus.

As métricas vivem na memória do processo: com mais de um worker do uvicorn, cada um teria a sua
contagem. Por isso a API roda com um worker só.
"""

from prometheus_client import Counter, Gauge, Histogram, Info

REQUISICOES = Counter(
    "triagem_requisicoes",
    "Requisições HTTP atendidas, por rota, método e status.",
    ["endpoint", "metodo", "status"],
)

LATENCIA_HTTP = Histogram(
    "triagem_requisicao_segundos",
    "Tempo total da requisição HTTP dentro da API (validação + inferência + serialização).",
    ["endpoint"],
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.04, 0.06, 0.08, 0.1, 0.15, 0.25, 0.5, 1, 2.5),
)

FAIXAS_INFERENCIA = (
    0.0002,
    0.0005,
    0.001,
    0.0025,
    0.005,
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    0.075,
    0.1,
    0.25,
    0.5,
    1,
)

LATENCIA_INFERENCIA = Histogram(
    "triagem_inferencia_segundos",
    "Tempo só do modelo (tokenização + inferência), por backend.",
    ["backend"],
    buckets=FAIXAS_INFERENCIA,
)

PREDICOES = Counter(
    "triagem_predicoes",
    "Predições por classe prevista: a distribuição ao longo do tempo é um sinal simples de drift.",
    ["backend", "classe"],
)

# Drift de entrada: se o tamanho dos textos muda muito em relação ao treino (mediana ~1.200
# caracteres), o modelo está vendo dados diferentes dos que aprendeu.
TAMANHO_TEXTO = Histogram(
    "triagem_texto_caracteres",
    "Tamanho do texto recebido em /predict, em caracteres.",
    buckets=(100, 250, 500, 750, 1000, 1250, 1500, 2000, 2500, 3000, 4000, 6000, 10000, 20000),
)

MODELO_CARREGADO = Gauge("triagem_modelo_carregado", "1 se há modelo carregado e pronto, 0 se não.")

MODELO = Info("triagem_modelo", "Backend e versão do modelo servido.")
