import pytest

from triagem.config import Configuracao
from triagem.evaluate import Desempenho, decidir, medir_latencia, medir_latencia_lado_a_lado

CFG = Configuracao(gate_f1_margem=0.005, gate_latencia_ganho_min=0.10, gate_p95_max_ms=5.0)
ATUAL = Desempenho(f1_macro=0.660, p95_ms=1.0)


def test_sem_modelo_atual_o_candidato_sobe():
    decisao = decidir(Desempenho(0.50, 100.0), None, CFG)
    assert decisao.aprovado and decisao.criterio == "sem_modelo_atual"


@pytest.mark.parametrize(
    ("candidato", "criterio"),
    [
        (
            Desempenho(0.666, 4.9),
            "A",
        ),  # melhor (≥ +0,005), mais lento, mas dentro do limite de 5 ms
        (Desempenho(0.656, 0.89), "B"),  # igual (dentro da margem) e ≥ 10% mais rápido
        (Desempenho(0.700, 0.50), "A"),  # melhor e mais rápido: A tem precedência
    ],
)
def test_criterios_que_aprovam(candidato, criterio):
    decisao = decidir(candidato, ATUAL, CFG)
    assert decisao.aprovado and decisao.criterio == criterio


@pytest.mark.parametrize(
    "candidato",
    [
        Desempenho(0.666, 5.1),  # melhor, mas acima do limite absoluto de latência
        Desempenho(0.660, 1.0),  # idêntico ao atual: nem melhor nem mais rápido
        Desempenho(0.663, 0.95),  # ganho de F1 abaixo da margem e só 5% mais rápido
        Desempenho(0.650, 0.50),  # muito mais rápido, mas pior além da margem
    ],
)
def test_criterios_que_reprovam(candidato):
    decisao = decidir(candidato, ATUAL, CFG)
    assert not decisao.aprovado and decisao.criterio == "reprovado"


def test_medicoes_funcionam_com_poucos_textos():
    # a amostra dos testes tem menos textos que warmup + n: a medição repete os textos em ciclo
    lat = medir_latencia(lambda t: t.upper(), ["a", "b"], n=20, warmup=5)
    assert lat["p50_ms"] <= lat["p95_ms"] <= lat["p99_ms"]
    lat_a, lat_b = medir_latencia_lado_a_lado(len, str.upper, ["a", "b", "c"], n=10, warmup=2)
    assert set(lat_a) == set(lat_b) == {"p50_ms", "p95_ms", "p99_ms"}
