"""Versão do código que gerou um modelo — também em cópias do projeto sem `.git`.

Onde há Git, o hook `post-commit` grava o commit e o hash de `src/` *como commitado* em
`diversos/versao-codigo.json`, que acompanha as cópias sem `.git`. No treino, `versao_do_codigo`
recalcula o hash de `src/` *no disco* e compara: se bater, o código é exatamente o daquele commit;
se não, havia alteração não commitada em `src/` quando o modelo foi treinado.

Uso pelo hook: python -m triagem.versao gravar
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from triagem.config import RAIZ

ARQUIVO_VERSAO = RAIZ / "diversos" / "versao-codigo.json"
PASTA_CODIGO = "src"


def hash_arquivos(conteudos: dict[str, bytes]) -> str:
    """Hash de um conjunto de arquivos: caminho + conteúdo, em ordem de caminho."""
    total = hashlib.sha256()
    for caminho in sorted(conteudos):
        total.update(caminho.encode() + b"\0" + hashlib.sha256(conteudos[caminho]).digest())
    return total.hexdigest()


def _git(*args: str, raiz: Path = RAIZ) -> str:
    return (
        subprocess.run(["git", *args], cwd=raiz, capture_output=True, check=True)
        .stdout.decode()
        .strip()
    )


def tem_git(raiz: Path = RAIZ) -> bool:
    return (raiz / ".git").exists()


def hash_src_no_disco(raiz: Path = RAIZ) -> str:
    """Hash de `src/` como está no disco (ignora caches do Python)."""
    arquivos = [
        p
        for p in (raiz / PASTA_CODIGO).rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    ]
    return hash_arquivos({p.relative_to(raiz).as_posix(): p.read_bytes() for p in arquivos})


def hash_src_no_commit(raiz: Path = RAIZ) -> str:
    """Hash de `src/` como está no último commit (lê do Git, não do disco)."""
    caminhos = _git("ls-files", PASTA_CODIGO, raiz=raiz).splitlines()
    conteudos = {
        c: subprocess.run(
            ["git", "show", f"HEAD:{c}"], cwd=raiz, capture_output=True, check=True
        ).stdout
        for c in caminhos
    }
    return hash_arquivos(conteudos)


def gravar(destino: Path = ARQUIVO_VERSAO, raiz: Path = RAIZ) -> dict:
    """Hook post-commit: grava o commit e o hash de src/ commitado num arquivo sincronizado."""
    versao = {
        "commit": _git("rev-parse", "HEAD", raiz=raiz),
        "hash_src": hash_src_no_commit(raiz),
        "gravado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(versao, indent=2) + "\n", encoding="utf-8")
    return versao


def versao_do_codigo(raiz: Path = RAIZ, arquivo: Path = ARQUIVO_VERSAO) -> dict:
    """Commit do código em uso e se `src/` no disco é idêntico a ele.

    Com `.git`, pergunta ao Git. Sem `.git`, lê o arquivo gravado pelo hook.
    """
    no_disco = hash_src_no_disco(raiz)
    if tem_git(raiz):
        commit, commitado = _git("rev-parse", "HEAD", raiz=raiz), hash_src_no_commit(raiz)
        origem = "git"
    elif arquivo.exists():
        gravado = json.loads(arquivo.read_text())
        commit, commitado = gravado["commit"], gravado["hash_src"]
        origem = "hook"
    else:
        return {"commit": None, "src_igual_ao_commit": None, "origem": "desconhecida"}
    return {"commit": commit, "src_igual_ao_commit": no_disco == commitado, "origem": origem}


if __name__ == "__main__":
    if sys.argv[1:] == ["gravar"]:
        v = gravar()
        print(f"versao-codigo: {v['commit'][:8]} (src {v['hash_src'][:8]})")
    else:
        print(json.dumps(versao_do_codigo(), indent=2))
