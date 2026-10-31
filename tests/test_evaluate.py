import pandas as pd
import pytest

from triagem.evaluate import acerto_por_resumo, calcular_metricas, metricas_no_teste

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


def test_metricas_no_teste_separa_textos_vistos_e_junta_os_rotulos():
    treino = pd.DataFrame({"medical_abstract": ["resumo a", "resumo b"], "condition_label": [1, 2]})

    teste = pd.DataFrame({"medical_abstract": ["RESUMO  a", "resumo c"], "condition_label": [3, 4]})
    previsto = [1, 4]

    m = metricas_no_teste(teste, previsto, treino)

    assert m["total"]["linhas"] == 2
    assert m["texto_novo"]["linhas"] == 1 and m["texto_novo"]["acuracia"] == 1.0
    assert m["texto_visto_no_treino"]["linhas"] == 1
    assert m["texto_visto_no_treino"]["acuracia"] == 0.0  # erra a linha...
    assert (
        m["texto_visto_no_treino"]["acerto_por_resumo"] == 1.0
    )  # ...mas acerta uma condição do resumo
    assert m["total"]["acerto_por_resumo"] == 1.0
