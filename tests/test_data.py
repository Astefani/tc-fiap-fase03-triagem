import pandas as pd
import pytest

from triagem.config import DIR_RAW
from triagem.data import (
    COLUNA_ROTULO,
    COLUNA_TEXTO,
    ErroSchema,
    carregar_treino,
    conjuntos_de_rotulos,
    normalizar_texto,
    separar_validacao,
    validar_schema,
)


def test_amostra_tem_schema_valido(amostra):
    assert validar_schema(amostra) is amostra
    assert set(amostra[COLUNA_ROTULO]) == {1, 2, 3, 4, 5}


def test_schema_rejeita_coluna_ausente(amostra):
    with pytest.raises(ErroSchema, match="colunas ausentes"):
        validar_schema(amostra.drop(columns=COLUNA_TEXTO))


@pytest.mark.parametrize("texto", ["", "   ", None])
def test_schema_rejeita_texto_vazio(amostra, texto):
    dados = amostra.copy()
    dados.loc[dados.index[0], COLUNA_TEXTO] = texto
    with pytest.raises(ErroSchema, match="texto nulo ou vazio"):
        validar_schema(dados)


@pytest.mark.parametrize("rotulo", [0, 6])
def test_schema_rejeita_rotulo_fora_do_intervalo(amostra, rotulo):
    dados = amostra.copy()
    dados.loc[dados.index[0], COLUNA_ROTULO] = rotulo
    with pytest.raises(ErroSchema, match="rótulo fora"):
        validar_schema(dados)


def test_normalizar_texto_ignora_caixa_e_espacos():
    textos = pd.Series(["Resumo  Sobre\tCancer", "resumo sobre cancer"])
    assert normalizar_texto(textos).nunique() == 1


def test_split_nunca_divide_o_mesmo_resumo(multirrotulo):
    treino, validacao = separar_validacao(multirrotulo, frac=0.2)
    lado_treino = set(normalizar_texto(treino[COLUNA_TEXTO]))
    lado_validacao = set(normalizar_texto(validacao[COLUNA_TEXTO]))
    assert lado_treino.isdisjoint(lado_validacao)
    assert len(treino) + len(validacao) == len(multirrotulo)


def test_split_e_deterministico(amostra):
    _, primeira = separar_validacao(amostra, seed=42)
    _, segunda = separar_validacao(amostra, seed=42)
    _, outra_seed = separar_validacao(amostra, seed=7)
    assert primeira.index.equals(segunda.index)
    assert not primeira.index.equals(outra_seed.index)


def test_split_preserva_tamanho_e_classes(amostra):
    _, validacao = separar_validacao(amostra, frac=0.1)
    assert 0.05 <= len(validacao) / len(amostra) <= 0.15
    assert set(validacao[COLUNA_ROTULO]) == {1, 2, 3, 4, 5}


def test_conjuntos_de_rotulos_juntam_as_condicoes_do_resumo(multirrotulo):
    conjuntos = conjuntos_de_rotulos(multirrotulo)
    assert len(conjuntos) == 40
    assert conjuntos["resumo 0 sobre a condicao"] == frozenset({1, 2})
    assert conjuntos["resumo 39 sobre a condicao"] == frozenset({5})


@pytest.mark.skipif(
    not (DIR_RAW / "medical_tc_train.csv").exists(), reason="dados reais ausentes (rodar dvc pull)"
)
def test_split_real_e_o_mesmo_dos_notebooks_do_bert():
    # Os resultados do BERT foram medidos neste split.
    treino, validacao = separar_validacao(carregar_treino())
    assert (len(treino), len(validacao)) == (10395, 1155)
    assert normalizar_texto(validacao[COLUNA_TEXTO]).nunique() == 947
