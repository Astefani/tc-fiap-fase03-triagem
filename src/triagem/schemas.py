"""Contratos de entrada e saída da API. Entrada fora do contrato vira 422, sem chegar ao modelo.

Limites do texto definidos a partir da EDA:
Os resumos têm de 170 a 4.000 caracteres (mediana 1.210).
O mínimo barra entrada vazia ou sem conteúdo clínico.
O máximo (5× o maior resumo) barra payloads abusivos.
O BERT trunca em 256 tokens, mas a tokenização e o TF-IDF percorrem o texto inteiro.
"""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

TEXTO_MIN = 20
TEXTO_MAX = 20_000

EXEMPLO_TEXTO = (
    "Long-term outcome of patients with malignant melanoma treated with adjuvant chemotherapy. "
    "We reviewed 120 patients with stage III disease followed for a median of 8 years."
)


class EntradaPredicao(BaseModel):
    # extra="forbid": um campo com nome errado ({"text": ...}) vira 422 em vez de ser ignorado
    model_config = ConfigDict(extra="forbid")

    texto: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=TEXTO_MIN, max_length=TEXTO_MAX),
        Field(description="Texto do laudo (em inglês).", examples=[EXEMPLO_TEXTO]),
    ]


class SaidaPredicao(BaseModel):
    classe_id: int = Field(description="Rótulo do dataset, de 1 a 5.", examples=[1])
    classe: str = Field(examples=["neoplasms"])
    probabilidades: dict[str, float] = Field(description="Nome da classe → probabilidade.")
    backend: str = Field(examples=["bert-onnx-int8"])
    versao_modelo: str = Field(description="nome@data (commit) do artefato servido.")


class Status(BaseModel):
    status: str
    backend: str | None = None
    versao_modelo: str | None = None
