"""Inferência com a mesma interface para todos os backends.

A API pede um preditor a `carregar_preditor` e chama `prever(texto)`, sem saber qual modelo está por
trás. O backend é escolhido por `TRIAGEM_MODEL_BACKEND`:
- `tfidf-logreg`   → models/tfidf/logreg/pipeline.joblib (scikit-learn)
- `bert-onnx`      → models/bert/model.onnx (PubMedBERT fp32)
- `bert-onnx-int8` → models/bert/model.int8.onnx (quantizado; padrão)

Os backends BERT rodam só com `tokenizers` + `onnxruntime`, sem torch na imagem.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from triagem.config import CONFIG, Backend, Configuracao

ARQUIVOS_BERT = {"bert-onnx": "model.onnx", "bert-onnx-int8": "model.int8.onnx"}


@dataclass(frozen=True)
class Predicao:
    classe_id: int  # 1..5, como no dataset
    classe: str
    probabilidades: dict[str, float]  # nome da classe → probabilidade
    backend: str
    versao_modelo: str


class Preditor(Protocol):
    backend: str
    versao_modelo: str

    def prever(self, texto: str) -> Predicao: ...


def _ler_metadata(pasta: Path) -> dict:
    return json.loads((pasta / "metadata.json").read_text(encoding="utf-8"))


def _classes(metadados: dict) -> dict[int, str]:
    # JSON guarda chaves como texto: {"1": "neoplasms", ...}
    return {int(k): v for k, v in metadados["classes"].items()}


def _versao(metadados: dict) -> str:
    """`nome@data (commit)`, com `+mod` se o artefato saiu de um `src/` diferente do commit.

    Vale para o export e para o treino que o originou (BERT): em qualquer dos dois, `+mod` diz que o
    código exato que gerou o modelo não está em nenhum commit.
    """
    codigo = metadados.get("codigo") or {}
    commit = codigo.get("commit")
    if not commit:
        return f"{metadados['nome']}@{metadados['criado_em']} (sem-commit)"
    registros = [codigo, (metadados.get("treino") or {}).get("codigo") or {}]
    modificado = any(r.get("src_igual_ao_commit") is False for r in registros)
    return (
        f"{metadados['nome']}@{metadados['criado_em']} ({commit[:8]}{'+mod' if modificado else ''})"
    )


def _montar(
    classe_id: int, probas: np.ndarray, ids: list[int], classes, backend, versao
) -> Predicao:
    return Predicao(
        classe_id=classe_id,
        classe=classes[classe_id],
        probabilidades={classes[i]: round(float(p), 6) for i, p in zip(ids, probas)},
        backend=backend,
        versao_modelo=versao,
    )


class PreditorTfidf:
    backend = "tfidf-logreg"

    def __init__(self, pasta: Path):
        metadados = _ler_metadata(pasta)
        self.pipeline = joblib.load(pasta / metadados.get("artefato", "pipeline.joblib"))
        self.classes = _classes(metadados)
        self.versao_modelo = _versao(metadados)
        self._ids = [int(c) for c in self.pipeline.classes_]

    def prever(self, texto: str) -> Predicao:
        probas = self.pipeline.predict_proba([texto])[0]
        classe_id = self._ids[int(np.argmax(probas))]
        return _montar(classe_id, probas, self._ids, self.classes, self.backend, self.versao_modelo)


class PreditorBertOnnx:
    def __init__(self, pasta: Path, backend: str = "bert-onnx-int8", threads: int = 4):
        metadados = _ler_metadata(pasta)
        self.backend = backend
        self.classes = _classes(metadados)
        self.versao_modelo = _versao(metadados)
        self._ids = sorted(self.classes)  # saída 0..4 do modelo ↔ rótulos 1..5

        self.tokenizer = Tokenizer.from_file(str(pasta / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=metadados["max_length"])
        self.tokenizer.no_padding()  # um texto por chamada: sem padding, sem custo extra

        opcoes = ort.SessionOptions()
        opcoes.intra_op_num_threads = threads
        opcoes.inter_op_num_threads = 1
        self.sessao = ort.InferenceSession(
            str(pasta / ARQUIVOS_BERT[backend]), opcoes, providers=["CPUExecutionProvider"]
        )
        self._entradas = {entrada.name for entrada in self.sessao.get_inputs()}

    def prever(self, texto: str) -> Predicao:
        codificado = self.tokenizer.encode(texto)
        alimentacao = {
            "input_ids": np.array([codificado.ids], dtype=np.int64),
            "attention_mask": np.array([codificado.attention_mask], dtype=np.int64),
            "token_type_ids": np.array([codificado.type_ids], dtype=np.int64),
        }
        logits = self.sessao.run(
            ["logits"], {k: v for k, v in alimentacao.items() if k in self._entradas}
        )[0][0]
        exp = np.exp(logits - logits.max())  # softmax estável
        probas = exp / exp.sum()
        classe_id = self._ids[int(np.argmax(probas))]
        return _montar(classe_id, probas, self._ids, self.classes, self.backend, self.versao_modelo)


def carregar_preditor(
    backend: Backend | None = None, dir_modelos: Path | None = None, cfg: Configuracao = CONFIG
) -> Preditor:
    """Fábrica: backend do argumento ou de `TRIAGEM_MODEL_BACKEND` (padrão: bert-onnx-int8)."""
    backend = backend or cfg.model_backend
    dir_modelos = dir_modelos or cfg.dir_modelos
    if backend == "tfidf-logreg":
        return PreditorTfidf(dir_modelos / "tfidf" / "logreg")
    if backend in ARQUIVOS_BERT:
        return PreditorBertOnnx(dir_modelos / "bert", backend, cfg.onnx_threads)
    raise ValueError(f"backend desconhecido: {backend!r} (use um de {list(Backend.__args__)})")
