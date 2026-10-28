"""Configuração do projeto, lida do ambiente (prefixo `TRIAGEM_`) e do `.env`.

API, container e DAG mudam caminhos, backend e regras do gate por variável de ambiente,
sem tocar no código.
As constantes no fim do módulo mantêm a forma de uso antiga (`from triagem.config import SEED`).
"""

import os
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# src/triagem/config.py
RAIZ = Path(__file__).resolve().parents[2]
PONTEIRO_DADOS = RAIZ / "data" / "raw.dvc"

Backend = Literal["tfidf-logreg", "bert-onnx", "bert-onnx-int8"]


class Configuracao(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TRIAGEM_", env_file=RAIZ / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    dir_raw: Path = RAIZ / "data" / "raw"
    dir_modelos: Path = RAIZ / "models"
    seed: int = 42
    frac_validacao: float = 0.10

    # Backend servido pela API (padrão)
    model_backend: Backend = "bert-onnx-int8"
    # Threads do ONNX Runtime por inferência
    onnx_threads: int = 4

    # Condições para modelo candidato se tornar produção:
    #   A) F1 ≥ F1_atual + margem   e  p95 ≤ p95_max_ms
    #   B) F1 ≥ F1_atual − margem   e  p95 ≤ p95_atual × (1 − latencia_ganho_min)
    gate_f1_margem: float = 0.005
    gate_latencia_ganho_min: float = 0.10
    gate_p95_max_ms: float = 5.0


class ConfiguracaoMLflow(BaseSettings):
    """Credenciais do MLflow (DagsHub). Sem `tracking_uri`, nada é enviado."""

    model_config = SettingsConfigDict(
        env_prefix="MLFLOW_", env_file=RAIZ / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    tracking_uri: str | None = None
    tracking_username: str | None = None
    tracking_password: str | None = None
    # Produção é o ponteiro do DVC no Git
    experiment_name: str = "triagem-laudos"


def aplicar_credenciais_mlflow() -> bool:
    """Copia as credenciais do `.env` para `os.environ` e diz se o rastreamento está configurado."""
    cfg = ConfiguracaoMLflow()
    for nome, valor in {
        "MLFLOW_TRACKING_URI": cfg.tracking_uri,
        "MLFLOW_TRACKING_USERNAME": cfg.tracking_username,
        "MLFLOW_TRACKING_PASSWORD": cfg.tracking_password,
    }.items():
        if valor and nome not in os.environ:
            os.environ[nome] = valor
    return bool(cfg.tracking_uri)


CONFIG = Configuracao()

# Forma de uso do Bloco 0, mantida para os módulos que importam constantes
DIR_RAW = CONFIG.dir_raw
DIR_MODELOS = CONFIG.dir_modelos
SEED = CONFIG.seed
FRAC_VALIDACAO = CONFIG.frac_validacao
