import pandas as pd
import pytest

from triagem.evaluate import acerto_por_resumo, calcular_metricas

CONJUNTOS = pd.Series({"a": frozenset({1, 5}), "b": frozenset({2}), "c": frozenset({3})})


def test_acerto_por_resumo_aceita_qualquer_condicao_do_resumo():
    # "a" aparece duas vezes (multirrótulo); prever 5 acerta, mesmo a linha sendo do rótulo 1
    textos = pd.Series(["a", "a", "b", "c"])
    previsto = [5, 5, 2, 1]
    assert acerto_por_resumo(textos, previsto, CONJUNTOS) == pytest.approx(2 / 3)


def test_calcular_metricas_inclui_acerto_so_com_conjuntos():
    real, previsto = [1, 2, 3], [1, 2, 1]
    assert set(calcular_metricas(real, previsto)) == {"f1_macro", "acuracia"}
    com_conjuntos = calcular_metricas(real, previsto, pd.Series(["a", "b", "c"]), CONJUNTOS)
    assert com_conjuntos["acerto_por_resumo"] == pytest.approx(2 / 3)
