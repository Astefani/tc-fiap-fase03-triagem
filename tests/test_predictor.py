import json

import pytest

from triagem.predictor import carregar_preditor
from triagem.train import avaliar_candidato, publicar, treinar_candidato


def _confere(predicao, backend, classes):
    assert predicao.backend == backend
    assert predicao.classe_id in classes and predicao.classe == classes[predicao.classe_id]
    assert set(predicao.probabilidades) == set(classes.values())
    assert sum(predicao.probabilidades.values()) == pytest.approx(1.0, abs=1e-4)


def test_preditor_tfidf(cfg_amostra, classes):
    treinar_candidato(cfg_amostra)
    avaliar_candidato(cfg_amostra)
    publicar(cfg_amostra)
    preditor = carregar_preditor("tfidf-logreg", cfg=cfg_amostra)
    _confere(preditor.prever("malignant tumor of the breast"), "tfidf-logreg", classes)


@pytest.mark.parametrize("backend", ["bert-onnx", "bert-onnx-int8"])
def test_preditor_bert_onnx(bert_falso, backend, classes):
    preditor = carregar_preditor(backend, dir_modelos=bert_falso.parent)
    predicao = preditor.prever("Cancer of the HEART and brain")
    _confere(predicao, backend, classes)
    assert predicao.classe_id == 3  # o viés do modelo falso favorece o índice 2 → classe 3


def test_bert_trunca_texto_longo(bert_falso, classes):
    preditor = carregar_preditor("bert-onnx-int8", dir_modelos=bert_falso.parent)
    assert len(preditor.tokenizer.encode("cancer " * 500).ids) == 8  # max_length do metadata
    _confere(preditor.prever("cancer " * 500), "bert-onnx-int8", classes)


def test_backend_desconhecido(cfg_amostra):
    with pytest.raises(ValueError, match="backend desconhecido"):
        carregar_preditor("svm", cfg=cfg_amostra)


@pytest.mark.parametrize(
    ("codigo", "treino", "esperado"),
    [
        (None, None, "(sem-commit)"),
        ({"commit": "abc12345ff", "src_igual_ao_commit": True}, None, "(abc12345)"),
        ({"commit": "abc12345ff", "src_igual_ao_commit": False}, None, "(abc12345+mod)"),
        # export limpo, mas o treino que originou o modelo tinha código não commitado
        (
            {"commit": "abc12345ff", "src_igual_ao_commit": True},
            {"codigo": {"commit": "abc12345ff", "src_igual_ao_commit": False}},
            "(abc12345+mod)",
        ),
    ],
)
def test_versao_do_modelo_marca_codigo_nao_commitado(bert_falso, codigo, treino, esperado):
    caminho = bert_falso / "metadata.json"
    metadados = json.loads(caminho.read_text())
    metadados.update({"codigo": codigo, "treino": treino})
    caminho.write_text(json.dumps(metadados))
    preditor = carregar_preditor("bert-onnx-int8", dir_modelos=bert_falso.parent)
    assert preditor.versao_modelo.endswith(esperado)
