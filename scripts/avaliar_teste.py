"""Avaliação final no teste oficial do dataset (2.888 linhas).

O teste oficial fica intocado durante todo o desenvolvimento: hiperparâmetros, escolha de modelo e
gate usam só a validação. Ele é usado **uma vez**, com os modelos finais, para a tabela do README.

Para cada backend: F1-macro, acurácia e acerto por resumo — no total e separados em "texto novo" ×
"texto visto no treino". A F1 de validação gravada no metadata entra ao lado, para comparação.

Saída: tabela no terminal e um JSON em 'docs/avaliacao/' com os modelos, o commit e os dados.

Uso:
    python scripts/avaliar_teste.py
    python scripts/avaliar_teste.py --teste ensaio.csv --sem-salvar   # sem tocar no teste oficial
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

from triagem.config import CONFIG, RAIZ
from triagem.data import (
    COLUNA_ROTULO,
    COLUNA_TEXTO,
    carregar_rotulos,
    carregar_treino,
    validar_schema,
)
from triagem.evaluate import metricas_no_teste
from triagem.predictor import carregar_preditor
from triagem.registro import versao_dos_dados
from triagem.versao import versao_do_codigo

BACKENDS = ["tfidf-logreg", "bert-onnx", "bert-onnx-int8"]
DIR_SAIDA = RAIZ / "docs" / "avaliacao"


def f1_de_validacao(backend: str) -> float | None:
    pasta = CONFIG.dir_modelos / ("tfidf/logreg" if backend == "tfidf-logreg" else "bert")
    metricas = json.loads((pasta / "metadata.json").read_text())["metricas"]
    return (metricas.get(backend) or metricas).get("f1_macro")


def prever_todos(backend: str, textos: pd.Series) -> tuple[list[int], str, float]:
    """Prevê cada texto único uma vez (o mesmo resumo aparece em várias linhas)."""
    preditor = carregar_preditor(backend)
    inicio = time.perf_counter()
    por_texto = {t: preditor.prever(t).classe_id for t in textos.drop_duplicates()}
    return [por_texto[t] for t in textos], preditor.versao_modelo, time.perf_counter() - inicio


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backends", nargs="+", default=BACKENDS)
    parser.add_argument("--teste", type=Path, default=CONFIG.dir_raw / "medical_tc_test.csv")
    parser.add_argument("--sem-salvar", action="store_true")
    args = parser.parse_args(argv)

    treino = carregar_treino()
    teste = validar_schema(pd.read_csv(args.teste))
    nomes = carregar_rotulos()
    resultados = {}
    for backend in args.backends:
        previsto, versao, segundos = prever_todos(backend, teste[COLUNA_TEXTO])
        por_classe = f1_score(teste[COLUNA_ROTULO], previsto, average=None, labels=sorted(nomes))
        resultados[backend] = {
            **metricas_no_teste(teste, previsto, treino),
            "f1_por_classe": {nomes[c]: float(f) for c, f in zip(sorted(nomes), por_classe)},
            "f1_validacao": f1_de_validacao(backend),
            "versao_modelo": versao,
            "segundos": round(segundos, 1),
        }
        r = resultados[backend]
        nan = float("nan")  # subconjunto vazio (ex.: ensaio com dados do próprio treino)
        r_novo, r_visto = (
            r[k] | {"f1_macro": r[k].get("f1_macro", nan)}
            for k in ("texto_novo", "texto_visto_no_treino")
        )
        print(
            f"{backend:15} F1 teste {r['total']['f1_macro']:.4f}"
            f" (validação {r['f1_validacao']:.4f})"
            f" · novo {r_novo['f1_macro']:.4f} ({r_novo['linhas']} linhas)"
            f" · visto {r_visto['f1_macro']:.4f} ({r_visto['linhas']} linhas)"
            f" · acerto/resumo {r['total']['acerto_por_resumo']:.4f}"
        )

    if args.sem_salvar:
        return
    agora = datetime.now(timezone.utc)
    relatorio = {
        "criado_em": agora.isoformat(timespec="seconds"),
        "arquivo_teste": str(
            args.teste.relative_to(RAIZ) if args.teste.is_relative_to(RAIZ) else args.teste
        ),
        "linhas_teste": len(teste),
        "dados_md5": versao_dos_dados(),
        "codigo": versao_do_codigo(),
        "resultados": resultados,
    }
    destino = DIR_SAIDA / f"teste-oficial_{agora:%Y-%m-%d_%H%M}.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(relatorio, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nsalvo em {destino.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
