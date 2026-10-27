"""Métricas de avaliação, medição de latência e o gate do retreino.

F1-macro é a métrica principal (as 5 classes pesam igual) e o acerto por resumo complementa.
Como o dataset é multirrótulo, prever *uma* das condições de um resumo não é erro, mas o
F1 por linha conta como erro nas outras linhas do mesmo resumo.

O gate compara o candidato com o modelo em produção **na mesma execução e nas mesmas condições**.
Como a latência varia 5–10% entre rodadas e até 2× entre máquinas,
comparar com um número gravado em outra hora não serve.
"""

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from itertools import cycle, islice

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from triagem.config import CONFIG, Configuracao


def acerto_por_resumo(textos_norm: pd.Series, previsto, conjuntos: pd.Series) -> float:
    """Fração dos resumos únicos cuja classe prevista está entre as condições do resumo."""
    por_resumo = pd.DataFrame({"texto": np.asarray(textos_norm), "previsto": np.asarray(previsto)})
    por_resumo = por_resumo.drop_duplicates("texto")
    acertos = [p in conjuntos[t] for t, p in zip(por_resumo["texto"], por_resumo["previsto"])]
    return float(np.mean(acertos))


def calcular_metricas(
    real, previsto, textos_norm: pd.Series | None = None, conjuntos: pd.Series | None = None
) -> dict[str, float]:
    """F1-macro, acurácia e acerto por resumo."""
    metricas = {
        "f1_macro": float(f1_score(real, previsto, average="macro")),
        "acuracia": float(accuracy_score(real, previsto)),
    }
    if textos_norm is not None and conjuntos is not None:
        metricas["acerto_por_resumo"] = acerto_por_resumo(textos_norm, previsto, conjuntos)
    return metricas


def _em_ciclo(textos: list[str], total: int) -> list[str]:
    """Repete os textos se houver menos que o necessário (ex.: amostra pequena dos testes)."""
    return list(islice(cycle(textos), total))


def _percentis(tempos_ms: list[float]) -> dict[str, float]:
    p50, p95, p99 = np.percentile(tempos_ms, [50, 95, 99])
    return {"p50_ms": float(p50), "p95_ms": float(p95), "p99_ms": float(p99)}


def medir_latencia(
    prever_um: Callable[[str], object], textos: list[str], n: int = 300, warmup: int = 30
) -> dict[str, float]:
    """Latência de 1 texto por chamada, em percentis."""
    textos = _em_ciclo(textos, warmup + n)
    for texto in textos[:warmup]:
        prever_um(texto)
    tempos = []
    for texto in textos[warmup : warmup + n]:
        inicio = time.perf_counter()
        prever_um(texto)
        tempos.append((time.perf_counter() - inicio) * 1000)
    return _percentis(tempos)


def medir_latencia_lado_a_lado(
    prever_a: Callable[[str], object],
    prever_b: Callable[[str], object],
    textos: list[str],
    n: int = 300,
    warmup: int = 30,
) -> tuple[dict[str, float], dict[str, float]]:
    """Mede dois modelos intercalando as chamadas: a variação do momento afeta os
    dois igualmente, tornando a comparação relativa confiável."""
    textos = _em_ciclo(textos, warmup + n)
    for texto in textos[:warmup]:
        prever_a(texto)
        prever_b(texto)
    tempos_a, tempos_b = [], []
    for texto in textos[warmup : warmup + n]:
        for prever, tempos in ((prever_a, tempos_a), (prever_b, tempos_b)):
            inicio = time.perf_counter()
            prever(texto)
            tempos.append((time.perf_counter() - inicio) * 1000)
    return _percentis(tempos_a), _percentis(tempos_b)


@dataclass(frozen=True)
class Desempenho:
    f1_macro: float
    p95_ms: float


@dataclass(frozen=True)
class DecisaoGate:
    aprovado: bool
    criterio: str  # "sem_modelo_atual" | "A" | "B" | "reprovado"
    motivo: str
    candidato: Desempenho
    atual: Desempenho | None

    def como_dict(self) -> dict:
        return asdict(self)


def decidir(
    candidato: Desempenho, atual: Desempenho | None, cfg: Configuracao = CONFIG
) -> DecisaoGate:
    """Aplica as regras do gate. Sobe se A **ou** B; sem modelo atual, sobe sempre.

    A) Melhor que o atual (F1 ≥ atual + margem), desde que dentro do limite absoluto de latência;
    B) Igual ou melhor (F1 ≥ atual − margem), desde que mais rápido (p95 abaixo do atual por pelo
       menos o ganho mínimo).
    """
    if atual is None:
        return DecisaoGate(True, "sem_modelo_atual", "não há modelo em produção", candidato, None)

    margem = cfg.gate_f1_margem
    teto_b = atual.p95_ms * (1 - cfg.gate_latencia_ganho_min)
    criterio_a = (
        candidato.f1_macro >= atual.f1_macro + margem and candidato.p95_ms <= cfg.gate_p95_max_ms
    )
    criterio_b = candidato.f1_macro >= atual.f1_macro - margem and candidato.p95_ms <= teto_b

    if criterio_a:
        motivo = (
            f"A: F1 {candidato.f1_macro:.4f} ≥ {atual.f1_macro + margem:.4f} e "
            f"p95 {candidato.p95_ms:.2f} ms ≤ limite {cfg.gate_p95_max_ms:.2f} ms"
        )
        return DecisaoGate(True, "A", motivo, candidato, atual)
    if criterio_b:
        motivo = (
            f"B: F1 {candidato.f1_macro:.4f} ≥ {atual.f1_macro - margem:.4f} e "
            f"p95 {candidato.p95_ms:.2f} ms ≤ {teto_b:.2f} ms ({cfg.gate_latencia_ganho_min:.0%} "
            "mais rápido)"
        )
        return DecisaoGate(True, "B", motivo, candidato, atual)
    motivo = (
        f"nem A nem B: F1 {candidato.f1_macro:.4f} × atual {atual.f1_macro:.4f}; "
        f"p95 {candidato.p95_ms:.2f} ms × atual {atual.p95_ms:.2f} ms"
    )
    return DecisaoGate(False, "reprovado", motivo, candidato, atual)
