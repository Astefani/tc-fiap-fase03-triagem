import json

from triagem.versao import hash_arquivos, versao_do_codigo


def test_hash_nao_depende_da_ordem_e_muda_com_o_conteudo():
    a = {"src/a.py": b"x = 1", "src/b.py": b"y = 2"}
    invertido = dict(reversed(list(a.items())))
    assert hash_arquivos(a) == hash_arquivos(invertido)
    assert hash_arquivos(a) != hash_arquivos({**a, "src/b.py": b"y = 3"})


def _projeto_sem_git(tmp_path, conteudo="x = 1"):
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "src" / "pkg" / "m.py").write_text(conteudo)
    (tmp_path / "src" / "pkg" / "__pycache__").mkdir()
    (tmp_path / "src" / "pkg" / "__pycache__" / "m.cpython-312.pyc").write_bytes(b"lixo")
    return tmp_path


def test_sem_git_usa_o_arquivo_do_hook_e_detecta_codigo_igual(tmp_path):
    raiz = _projeto_sem_git(tmp_path)
    arquivo = tmp_path / "versao.json"
    hash_commit = hash_arquivos({"src/pkg/m.py": b"x = 1"})  # o cache .pyc não entra
    arquivo.write_text(json.dumps({"commit": "abc123", "hash_src": hash_commit}))
    assert versao_do_codigo(raiz, arquivo) == {
        "commit": "abc123",
        "src_igual_ao_commit": True,
        "origem": "hook",
    }


def test_sem_git_detecta_codigo_nao_commitado(tmp_path):
    raiz = _projeto_sem_git(tmp_path, conteudo="x = 2  # alterado depois do commit")
    arquivo = tmp_path / "versao.json"
    arquivo.write_text(
        json.dumps({"commit": "abc123", "hash_src": hash_arquivos({"src/pkg/m.py": b"x = 1"})})
    )
    assert versao_do_codigo(raiz, arquivo)["src_igual_ao_commit"] is False


def test_sem_git_e_sem_arquivo_marca_versao_desconhecida(tmp_path):
    raiz = _projeto_sem_git(tmp_path)
    assert versao_do_codigo(raiz, tmp_path / "inexistente.json")["origem"] == "desconhecida"
