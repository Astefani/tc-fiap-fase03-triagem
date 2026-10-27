"""Leitura, validação de schema e separação treino/validação do dataset Medical Abstracts.

O dataset é multirrótulo: um resumo que trata de N condições aparece N vezes, uma linha
por rótulo. Por isso a validação é separada **agrupando por texto** — o mesmo resumo
nunca fica dos dois lados — e o conjunto de condições de cada resumo fica disponível para a métrica
de acerto por resumo.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from triagem.config import DIR_RAW, FRAC_VALIDACAO, SEED

COLUNA_TEXTO = "medical_abstract"
COLUNA_ROTULO = "condition_label"
ROTULOS_VALIDOS = (1, 2, 3, 4, 5)


class ErroSchema(ValueError):
    """Os dados não têm o formato esperado."""


def validar_schema(dados: pd.DataFrame) -> pd.DataFrame:
    """Confere colunas, texto presente e rótulo em 1 a 5; devolve os próprios dados se ok."""
    faltando = {COLUNA_TEXTO, COLUNA_ROTULO} - set(dados.columns)
    if faltando:
        raise ErroSchema(f"colunas ausentes: {sorted(faltando)}")

    problemas = []
    texto_vazio = dados[COLUNA_TEXTO].fillna("").astype(str).str.strip() == ""
    if texto_vazio.any():
        problemas.append(f"{int(texto_vazio.sum())} linha(s) com texto nulo ou vazio")
    rotulo_invalido = ~dados[COLUNA_ROTULO].isin(ROTULOS_VALIDOS)
    if rotulo_invalido.any():
        problemas.append(
            f"{int(rotulo_invalido.sum())} linha(s) com rótulo fora de {ROTULOS_VALIDOS}"
        )
    if problemas:
        raise ErroSchema("; ".join(problemas))
    return dados


def carregar_rotulos(dir_raw: Path = DIR_RAW) -> dict[int, str]:
    """Mapa id → nome das 5 condições."""
    rotulos = pd.read_csv(dir_raw / "medical_tc_labels.csv")
    return dict(zip(rotulos["condition_label"], rotulos["condition_name"]))


def carregar_treino(dir_raw: Path = DIR_RAW) -> pd.DataFrame:
    """Treino oficial do dataset, validado."""
    return validar_schema(pd.read_csv(dir_raw / "medical_tc_train.csv"))


def carregar_teste(dir_raw: Path = DIR_RAW) -> pd.DataFrame:
    """Teste oficial do dataset, validado. Só para a avaliação final."""
    return validar_schema(pd.read_csv(dir_raw / "medical_tc_test.csv"))


def normalizar_texto(textos: pd.Series) -> pd.Series:
    """Caixa baixa e espaços colapsados para identificar o mesmo resumo em várias linhas."""
    return textos.str.lower().str.split().str.join(" ")


def separar_validacao(
    dados: pd.DataFrame, frac: float = FRAC_VALIDACAO, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa treino e validação agrupando por texto e estratificando por classe.

    Usa a primeira dobra de um `StratifiedGroupKFold` com `round(1 / frac)` dobras: cada resumo cai
    inteiro de um lado só e a proporção das classes é preservada de forma aproximada.
    Determinístico para a mesma seed. TF-IDF e BERT compartilham este split.
    """
    grupos = normalizar_texto(dados[COLUNA_TEXTO]).factorize()[0]
    divisor = StratifiedGroupKFold(n_splits=round(1 / frac), shuffle=True, random_state=seed)
    idx_treino, idx_validacao = next(divisor.split(dados, dados[COLUNA_ROTULO], groups=grupos))
    return dados.iloc[idx_treino], dados.iloc[idx_validacao]


def conjuntos_de_rotulos(dados: pd.DataFrame) -> pd.Series:
    """Para cada resumo (texto normalizado), o conjunto de condições em que ele aparece."""
    chave = normalizar_texto(dados[COLUNA_TEXTO])
    return dados[COLUNA_ROTULO].groupby(chave).agg(frozenset)
