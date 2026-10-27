"""Grava sempre um metadata.json junto ao artefato. Se o MLflow estiver configurado (.env com
MLFLOW_TRACKING_URI e credenciais do DagsHub) e o cliente instalado (--with rastreamento),
registra também parâmetros, métricas e etiquetas no MLflow.
O modelo em si não vai para o MLflow, pois quem o versiona é o DVC.

Uso antes do 'dvc add': python -m triagem.registro verificar models/bert
"""

import hashlib
import json
import logging
import platform
import re
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from triagem.config import PONTEIRO_DADOS, ConfiguracaoMLflow, aplicar_credenciais_mlflow
from triagem.versao import versao_do_codigo

log = logging.getLogger(__name__)


def versao_dos_dados(ponteiro: Path = PONTEIRO_DADOS) -> str | None:
    """md5 do `data/raw` segundo o ponteiro do DVC para ligar o modelo à versão exata dos dados."""
    if not ponteiro.exists():
        return None
    achado = re.search(r"md5:\s*(\S+)", ponteiro.read_text())
    return achado.group(1) if achado else None


def hash_arquivo(caminho: Path) -> str:
    sha = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1 << 20), b""):
            sha.update(bloco)
    return sha.hexdigest()


def hashes_da_pasta(pasta: Path) -> dict[str, str]:
    """sha256 de cada arquivo da pasta, exceto o próprio metadata.json."""
    return {
        p.relative_to(pasta).as_posix(): hash_arquivo(p)
        for p in sorted(pasta.rglob("*"))
        if p.is_file() and p.name != "metadata.json" and p.name != ".DS_Store"
    }


def _achatar(dados: dict, prefixo: str = "") -> dict:
    """{"tfidf": {"min_df": 2}} → {"tfidf.min_df": 2}"""
    plano = {}
    for chave, valor in dados.items():
        nome = f"{prefixo}{chave}"
        if isinstance(valor, dict):
            plano.update(_achatar(valor, f"{nome}."))
        else:
            plano[nome] = valor
    return plano


def _registrar_no_mlflow(metadados: dict, caminho_metadata: Path) -> str | None:
    """Envia o run ao MLflow. Falha vira aviso: rastreamento não pode derrubar um treino."""
    if not aplicar_credenciais_mlflow():
        return None
    try:
        import mlflow
    except ImportError:
        log.warning(
            "MLflow configurado no .env, mas não instalado (poetry install --with rastreamento)"
        )
        return None
    try:
        mlflow.set_experiment(ConfiguracaoMLflow().experiment_name)
        with mlflow.start_run(run_name=metadados["nome"]) as run:
            mlflow.log_params(_achatar(metadados["params"]))
            mlflow.log_metrics(
                {
                    k: v
                    for k, v in _achatar(metadados["metricas"]).items()
                    if isinstance(v, int | float)
                }
            )
            codigo = metadados["codigo"]
            mlflow.set_tags(
                {
                    "dados_md5": metadados["dados_md5"],
                    "codigo.commit": codigo.get("commit"),
                    "codigo.src_igual_ao_commit": codigo.get("src_igual_ao_commit"),
                    "codigo.origem": codigo.get("origem"),
                    "maquina": platform.node(),
                }
            )
            mlflow.log_artifact(str(caminho_metadata))
            return run.info.run_id
    except Exception as erro:  # rede, credencial, servidor fora do ar
        log.warning("Falha ao registrar no MLflow (o metadata.json foi gravado): %s", erro)
        return None


def registrar_execucao(
    destino: Path, *, nome: str, params: dict, metricas: dict, extras: dict | None = None
) -> Path:
    """Grava `destino/metadata.json` e, se configurado, o run no MLflow."""
    destino.mkdir(parents=True, exist_ok=True)
    metadados = {
        "nome": nome,
        "criado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "params": params,
        "metricas": metricas,
        "dados_md5": versao_dos_dados(),
        "codigo": versao_do_codigo(),
        "versoes": {
            "python": platform.python_version(),
            "scikit-learn": version("scikit-learn"),
        },
        "arquivos": hashes_da_pasta(destino),
        **(extras or {}),
    }
    caminho = destino / "metadata.json"
    caminho.write_text(json.dumps(metadados, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    run_id = _registrar_no_mlflow(metadados, caminho)
    if run_id:
        metadados["mlflow_run_id"] = run_id
        caminho.write_text(
            json.dumps(metadados, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return caminho


def verificar_pasta(pasta: Path) -> list[str]:
    """Compara os arquivos da pasta com os hashes do metadata.json. Lista vazia = tudo íntegro.

    Usado antes do `dvc add`: garante que a pasta copiada da máquina de treino chegou inteira.
    """
    esperado = json.loads((pasta / "metadata.json").read_text())["arquivos"]
    encontrado = hashes_da_pasta(pasta)
    problemas = [f"faltando: {a}" for a in esperado if a not in encontrado]
    problemas += [
        f"diferente: {a}" for a in esperado if a in encontrado and encontrado[a] != esperado[a]
    ]
    problemas += [f"sobrando: {a}" for a in encontrado if a not in esperado]
    return problemas


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "verificar":
        problemas = verificar_pasta(Path(sys.argv[2]))
        print("\n".join(problemas) if problemas else f"{sys.argv[2]}: íntegra, pode versionar.")
        sys.exit(1 if problemas else 0)
    print("uso: python -m triagem.registro verificar <pasta>")
    sys.exit(2)
