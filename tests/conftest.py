import json
from pathlib import Path

import numpy as np
import onnx
import pandas as pd
import pytest
from onnx import TensorProto, helper
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors

from triagem.config import Configuracao

AMOSTRA = Path(__file__).parent / "data" / "sample.csv"
CLASSES = {
    1: "neoplasms",
    2: "digestive system diseases",
    3: "nervous system diseases",
    4: "cardiovascular diseases",
    5: "general pathological conditions",
}


@pytest.fixture
def classes() -> dict[int, str]:
    return CLASSES


@pytest.fixture(autouse=True)
def sem_mlflow(monkeypatch):
    """Nenhum teste envia runs ao DagsHub, mesmo com um .env configurado na máquina."""
    monkeypatch.setattr("triagem.registro.aplicar_credenciais_mlflow", lambda: False)


@pytest.fixture
def amostra() -> pd.DataFrame:
    """Amostra estratificada do treino real (50 por classe), versionada no Git."""
    return pd.read_csv(AMOSTRA)


@pytest.fixture
def multirrotulo() -> pd.DataFrame:
    """Dados construídos de propósito com o padrão do dataset: o mesmo resumo em várias linhas.

    40 resumos distintos, 8 por classe; os 10 primeiros reaparecem com outro rótulo e com caixa e
    espaços diferentes — que devem contar como o mesmo resumo.
    """
    linhas = [
        {"medical_abstract": f"resumo {i} sobre a condicao", "condition_label": i % 5 + 1}
        for i in range(40)
    ]
    repetidos = [
        {"medical_abstract": f"RESUMO {i}   sobre a condicao", "condition_label": (i + 1) % 5 + 1}
        for i in range(10)
    ]
    return pd.DataFrame(linhas + repetidos)


@pytest.fixture
def cfg_amostra(tmp_path, amostra) -> Configuracao:
    """Configuração com data/raw e models/ temporários, usando a amostra como treino."""
    dir_raw = tmp_path / "raw"
    dir_raw.mkdir()
    amostra.to_csv(dir_raw / "medical_tc_train.csv", index=False)
    pd.DataFrame(
        {"condition_label": list(CLASSES), "condition_name": list(CLASSES.values())}
    ).to_csv(dir_raw / "medical_tc_labels.csv", index=False)
    return Configuracao(dir_raw=dir_raw, dir_modelos=tmp_path / "models")


@pytest.fixture
def bert_falso(tmp_path) -> Path:
    """Pasta models/bert com um 'BERT' minúsculo: mesmo formato de entrada e saída do real.

    O grafo calcula logits = média(input_ids) × W + b.
    Serve para testar a tubulação da API — tokenizar, rodar no ONNX Runtime, softmax, mapear a
    classe.
    """
    pasta = tmp_path / "models" / "bert"
    pasta.mkdir(parents=True)

    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "cancer": 4, "heart": 5, "brain": 6}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Lowercase()
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    tokenizer.save(str(pasta / "tokenizer.json"))

    entradas = [
        helper.make_tensor_value_info(nome, TensorProto.INT64, ["batch", "seq"])
        for nome in ("input_ids", "attention_mask", "token_type_ids")
    ]
    saida = helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 5])
    pesos = helper.make_tensor("W", TensorProto.FLOAT, [1, 5], np.zeros(5, dtype=np.float32))
    vies = helper.make_tensor(
        "b", TensorProto.FLOAT, [5], np.array([0, 0, 5, 0, 0], dtype=np.float32)
    )
    nos = [
        helper.make_node("Cast", ["input_ids"], ["ids_f"], to=TensorProto.FLOAT),
        helper.make_node("ReduceMean", ["ids_f"], ["media"], axes=[1], keepdims=1),
        helper.make_node("MatMul", ["media", "W"], ["xw"]),
        helper.make_node("Add", ["xw", "b"], ["logits"]),
    ]
    grafo = helper.make_graph(nos, "bert_falso", entradas, [saida], initializer=[pesos, vies])
    modelo = helper.make_model(grafo, opset_imports=[helper.make_opsetid("", 17)])
    modelo.ir_version = 8
    for arquivo in ("model.onnx", "model.int8.onnx"):
        onnx.save(modelo, pasta / arquivo)

    (pasta / "metadata.json").write_text(
        json.dumps(
            {
                "nome": "bert-onnx",
                "criado_em": "2026-10-25T00:00:00+00:00",
                "max_length": 8,
                "classes": CLASSES,
            }
        )
    )
    return pasta
