import json
import sys
import types

from triagem.registro import registrar_execucao, verificar_pasta, versao_dos_dados


def test_registrar_execucao_grava_metadata_com_hashes(tmp_path):
    destino = tmp_path / "modelo"
    destino.mkdir()
    (destino / "pipeline.joblib").write_bytes(b"pesos")
    caminho = registrar_execucao(
        destino, nome="logreg", params={"C": 1.0}, metricas={"f1_macro": 0.66}
    )
    metadados = json.loads(caminho.read_text())
    assert caminho.name == "metadata.json"
    assert metadados["params"] == {"C": 1.0}
    assert {"criado_em", "dados_md5", "codigo", "versoes", "arquivos"} <= set(metadados)
    assert list(metadados["arquivos"]) == ["pipeline.joblib"]
    assert "mlflow_run_id" not in metadados  # a fixture sem_mlflow desliga o envio


def test_verificar_pasta_detecta_arquivo_faltando_ou_alterado(tmp_path):
    (tmp_path / "a.onnx").write_bytes(b"1")
    (tmp_path / "b.json").write_bytes(b"2")
    registrar_execucao(tmp_path, nome="x", params={}, metricas={})
    assert verificar_pasta(tmp_path) == []
    (tmp_path / "a.onnx").write_bytes(b"alterado")
    (tmp_path / "b.json").unlink()
    assert sorted(verificar_pasta(tmp_path)) == ["diferente: a.onnx", "faltando: b.json"]


def test_versao_dos_dados_le_o_ponteiro_do_dvc(tmp_path):
    ponteiro = tmp_path / "raw.dvc"
    ponteiro.write_text("outs:\n- md5: abc123.dir\n  path: raw\n")
    assert versao_dos_dados(ponteiro) == "abc123.dir"
    assert versao_dos_dados(tmp_path / "inexistente.dvc") is None


def test_com_mlflow_configurado_envia_params_achatados(tmp_path, monkeypatch):
    chamadas = {}

    class RunFalso:
        info = types.SimpleNamespace(run_id="run-123")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    mlflow_falso = types.SimpleNamespace(
        set_experiment=lambda nome: chamadas.setdefault("experimento", nome),
        start_run=lambda run_name: RunFalso(),
        log_params=lambda p: chamadas.setdefault("params", p),
        log_metrics=lambda m: chamadas.setdefault("metricas", m),
        set_tags=lambda t: chamadas.setdefault("tags", t),
        log_artifact=lambda c: chamadas.setdefault("artefato", c),
    )
    monkeypatch.setitem(sys.modules, "mlflow", mlflow_falso)
    monkeypatch.setattr("triagem.registro.aplicar_credenciais_mlflow", lambda: True)

    caminho = registrar_execucao(
        tmp_path,
        nome="bert",
        params={"tfidf": {"min_df": 2}},
        metricas={"bert-onnx": {"f1_macro": 0.69}},
    )
    assert chamadas["experimento"] == "triagem-laudos"
    assert chamadas["params"] == {"tfidf.min_df": 2}
    assert chamadas["metricas"] == {"bert-onnx.f1_macro": 0.69}
    assert json.loads(caminho.read_text())["mlflow_run_id"] == "run-123"
