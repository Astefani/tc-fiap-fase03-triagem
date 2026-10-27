import json

import joblib
import pytest

from triagem.train import (
    ARTEFATO,
    avaliar_candidato,
    dir_candidato,
    dir_producao,
    montar_pipeline,
    publicar,
    treinar_candidato,
)


def test_pipeline_usa_a_configuracao_escolhida():
    pipeline = montar_pipeline()
    assert pipeline.named_steps["tfidf"].max_features == 50_000
    assert pipeline.named_steps["clf"].C == 3.0


def test_treinar_candidato_grava_artefato_e_metadata(cfg_amostra):
    destino = treinar_candidato(cfg_amostra)
    metadados = json.loads((destino / "metadata.json").read_text())
    assert (destino / ARTEFATO).exists()
    assert metadados["backend"] == "tfidf-logreg"
    assert set(metadados["classes"]) == {"1", "2", "3", "4", "5"}
    assert ARTEFATO in metadados["arquivos"]
    assert 0 <= metadados["metricas"]["f1_macro"] <= 1
    pipeline = joblib.load(destino / ARTEFATO)
    assert pipeline.predict(["malignant tumor of the breast"])[0] in {1, 2, 3, 4, 5}


def test_primeiro_retreino_sobe_sem_comparacao(cfg_amostra):
    treinar_candidato(cfg_amostra)
    decisao = avaliar_candidato(cfg_amostra)
    assert decisao.aprovado and decisao.criterio == "sem_modelo_atual"
    gate = json.loads((dir_candidato(cfg_amostra) / "metadata.json").read_text())["gate"]
    assert gate["aprovado"] is True


def test_publicar_substitui_a_producao(cfg_amostra):
    treinar_candidato(cfg_amostra)
    avaliar_candidato(cfg_amostra)
    destino = publicar(cfg_amostra)
    assert destino == dir_producao(cfg_amostra)
    assert (destino / ARTEFATO).exists()
    assert not destino.with_name(destino.name + ".novo").exists()


def test_publicar_recusa_candidato_sem_aprovacao(cfg_amostra):
    treinar_candidato(cfg_amostra)  # sem avaliar: não há decisão do gate
    with pytest.raises(RuntimeError, match="não aprovado"):
        publicar(cfg_amostra)
