import logging

import pytest
from fastapi.testclient import TestClient

from triagem.api import criar_app
from triagem.config import Configuracao
from triagem.schemas import TEXTO_MAX, TEXTO_MIN

TEXTO = "Cancer of the HEART and brain in a cohort of adult patients"


@pytest.fixture
def cliente(bert_falso):
    cfg = Configuracao(dir_modelos=bert_falso.parent, model_backend="bert-onnx-int8")
    with TestClient(criar_app(cfg)) as c:  # o `with` dispara o lifespan (carga do modelo)
        yield c


@pytest.fixture
def cliente_sem_modelo(tmp_path):
    cfg = Configuracao(dir_modelos=tmp_path / "vazio", model_backend="bert-onnx-int8")
    with TestClient(criar_app(cfg)) as c:
        yield c


def test_health(cliente):
    assert cliente.get("/health").json() == {"status": "ok", "backend": None, "versao_modelo": None}


def test_ready_informa_backend_e_versao(cliente):
    resposta = cliente.get("/ready")
    assert resposta.status_code == 200
    assert resposta.json()["backend"] == "bert-onnx-int8"
    assert resposta.json()["versao_modelo"].startswith("bert-onnx@")


def test_predict_valido(cliente, classes):
    resposta = cliente.post("/predict", json={"texto": TEXTO})
    assert resposta.status_code == 200
    corpo = resposta.json()
    assert corpo["classe_id"] == 3 and corpo["classe"] == classes[3]  # viés do BERT falso
    assert set(corpo["probabilidades"]) == set(classes.values())
    assert sum(corpo["probabilidades"].values()) == pytest.approx(1.0, abs=1e-4)
    assert corpo["backend"] == "bert-onnx-int8"


@pytest.mark.parametrize(
    "corpo",
    [
        {},  # sem o campo
        {"texto": ""},
        {"texto": " " * 100},  # só espaços: vazio depois do strip
        {"texto": "x" * (TEXTO_MIN - 1)},
        {"texto": "x" * (TEXTO_MAX + 1)},
        {"texto": 123},
        {"text": TEXTO},  # campo com nome errado
        {"texto": TEXTO, "extra": 1},
    ],
)
def test_predict_rejeita_entrada_invalida(cliente, corpo):
    assert cliente.post("/predict", json=corpo).status_code == 422


def test_texto_do_laudo_nao_vai_para_o_log(cliente, caplog):
    with caplog.at_level(logging.INFO, logger="triagem.api"):
        cliente.post("/predict", json={"texto": TEXTO})
    assert "predicao backend=bert-onnx-int8" in caplog.text
    assert "Cancer" not in caplog.text


def test_sem_modelo_esta_vivo_mas_nao_pronto(cliente_sem_modelo):
    assert cliente_sem_modelo.get("/health").status_code == 200
    assert cliente_sem_modelo.get("/ready").status_code == 503
    assert cliente_sem_modelo.post("/predict", json={"texto": TEXTO}).status_code == 503


def test_raiz_redireciona_para_a_documentacao(cliente):
    resposta = cliente.get("/", follow_redirects=False)
    assert resposta.status_code == 307
    assert resposta.headers["location"] == "/docs"
