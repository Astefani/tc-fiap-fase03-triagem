"""Retreino do TF-IDF + LogReg: treina um candidato, compara com o modelo em produção e publica.

É o ciclo que a DAG do Airflow executa. Os hiperparâmetros vêm da busca do `02-comparacao-modelos`.
Não é a configuração de maior F1 na CV, e sim a mais simples entre as que empatam com ela: a
diferença para a melhor (0,6564 × 0,6571) é menor que o erro-padrão da CV (~0,003), então as duas
são indistinguíveis, e a escolhida (vocabulário de 50k termos, C=3) treina 2× mais rápido e gera um
artefato menor.

Uso:
    python -m triagem.train treinar     # treina e grava em models/tfidf/candidato
    python -m triagem.train avaliar     # gate: candidato × models/tfidf/logreg
    python -m triagem.train publicar    # candidato → logreg (só se o gate aprovou)
    python -m triagem.train retreinar   # os três em sequência
"""

import argparse
import json
import logging
import shutil
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from triagem.config import CONFIG, Configuracao
from triagem.data import (
    COLUNA_ROTULO,
    COLUNA_TEXTO,
    carregar_rotulos,
    carregar_treino,
    conjuntos_de_rotulos,
    normalizar_texto,
    separar_validacao,
)
from triagem.evaluate import (
    DecisaoGate,
    Desempenho,
    calcular_metricas,
    decidir,
    medir_latencia,
    medir_latencia_lado_a_lado,
)
from triagem.registro import registrar_execucao

log = logging.getLogger(__name__)

ARTEFATO = "pipeline.joblib"
PARAMS_TFIDF = {
    "lowercase": True,
    "stop_words": "english",
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.9,
    "sublinear_tf": True,
    "max_features": 50_000,
}
PARAMS_LOGREG = {"C": 3.0, "max_iter": 2000, "class_weight": "balanced"}


def dir_candidato(cfg: Configuracao = CONFIG) -> Path:
    return cfg.dir_modelos / "tfidf" / "candidato"


def dir_producao(cfg: Configuracao = CONFIG) -> Path:
    return cfg.dir_modelos / "tfidf" / "logreg"


def montar_pipeline(seed: int = CONFIG.seed) -> Pipeline:
    """TF-IDF + LogReg num pipeline único: o artefato salvo recebe texto cru e devolve a classe."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(**PARAMS_TFIDF)),
            ("clf", LogisticRegression(**PARAMS_LOGREG, random_state=seed)),
        ]
    )


def _dados(cfg: Configuracao) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    treino = carregar_treino(cfg.dir_raw)
    tr, va = separar_validacao(treino, cfg.frac_validacao, cfg.seed)
    return tr, va, conjuntos_de_rotulos(treino)


def treinar_candidato(cfg: Configuracao = CONFIG) -> Path:
    """Treina sem a validação, mede nela e grava artefato + metadata no candidato."""
    tr, va, conjuntos = _dados(cfg)
    inicio = time.perf_counter()
    pipeline = montar_pipeline(cfg.seed).fit(tr[COLUNA_TEXTO], tr[COLUNA_ROTULO])
    tempo_treino = time.perf_counter() - inicio

    previsto = pipeline.predict(va[COLUNA_TEXTO])
    metricas = calcular_metricas(
        va[COLUNA_ROTULO], previsto, normalizar_texto(va[COLUNA_TEXTO]), conjuntos
    )
    metricas["treino_s"] = round(tempo_treino, 2)

    destino = dir_candidato(cfg)
    if destino.exists():
        shutil.rmtree(destino)
    destino.mkdir(parents=True)
    joblib.dump(pipeline, destino / ARTEFATO)
    registrar_execucao(
        destino,
        nome="tfidf-logreg",
        params={"tfidf": PARAMS_TFIDF, "clf": {**PARAMS_LOGREG, "random_state": cfg.seed}},
        metricas=metricas,
        extras={
            "backend": "tfidf-logreg",
            "artefato": ARTEFATO,
            "classes": carregar_rotulos(cfg.dir_raw),
            "linhas_treino": len(tr),
        },
    )
    log.info("candidato treinado: F1-macro %.4f em %.1f s", metricas["f1_macro"], tempo_treino)
    return destino


def avaliar_candidato(cfg: Configuracao = CONFIG) -> DecisaoGate:
    """Gate: mede candidato e produção na mesma validação e no mesmo benchmark, e decide (A ou B).

    A decisão é gravada no metadata.json do candidato, para ficar junto do artefato que ela aprovou.
    """
    _, va, _ = _dados(cfg)
    textos = va[COLUNA_TEXTO].tolist()
    candidato = joblib.load(dir_candidato(cfg) / ARTEFATO)
    f1_candidato = calcular_metricas(va[COLUNA_ROTULO], candidato.predict(textos))["f1_macro"]

    caminho_producao = dir_producao(cfg) / ARTEFATO
    if caminho_producao.exists():
        producao = joblib.load(caminho_producao)
        f1_producao = calcular_metricas(va[COLUNA_ROTULO], producao.predict(textos))["f1_macro"]
        lat_candidato, lat_producao = medir_latencia_lado_a_lado(
            lambda t: candidato.predict([t]), lambda t: producao.predict([t]), textos
        )
        atual = Desempenho(f1_producao, lat_producao["p95_ms"])
    else:
        lat_candidato = medir_latencia(lambda t: candidato.predict([t]), textos)
        atual = None

    decisao = decidir(Desempenho(f1_candidato, lat_candidato["p95_ms"]), atual, cfg)
    caminho_meta = dir_candidato(cfg) / "metadata.json"
    metadados = json.loads(caminho_meta.read_text())
    metadados["gate"] = decisao.como_dict()
    caminho_meta.write_text(json.dumps(metadados, indent=2, ensure_ascii=False) + "\n")
    log.info("gate: %s — %s", "APROVADO" if decisao.aprovado else "REPROVADO", decisao.motivo)
    return decisao


def publicar(cfg: Configuracao = CONFIG) -> Path:
    """Substitui o modelo em produção pelo candidato, se o gate dele aprovou.

    Troca por renomeação (cópia completa ao lado e depois `rename`): a API nunca encontra a pasta de
    produção pela metade.
    """
    origem, destino = dir_candidato(cfg), dir_producao(cfg)
    gate = json.loads((origem / "metadata.json").read_text()).get("gate")
    if not gate or not gate["aprovado"]:
        raise RuntimeError(f"candidato não aprovado pelo gate: {gate and gate['motivo']}")

    temporario = destino.with_name(destino.name + ".novo")
    antigo = destino.with_name(destino.name + ".antigo")
    for pasta in (temporario, antigo):
        if pasta.exists():
            shutil.rmtree(pasta)
    shutil.copytree(origem, temporario)
    if destino.exists():
        destino.rename(antigo)
    temporario.rename(destino)
    if antigo.exists():
        shutil.rmtree(antigo)
    log.info("publicado em %s", destino)
    return destino


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Treino do TF-IDF + LogReg com gate.")
    parser.add_argument("comando", choices=["treinar", "avaliar", "publicar", "retreinar"])
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", force=True
    )

    if args.comando in ("treinar", "retreinar"):
        treinar_candidato()
    if args.comando in ("avaliar", "retreinar"):
        decisao = avaliar_candidato()
        if args.comando == "retreinar" and not decisao.aprovado:
            log.info("modelo em produção mantido")
            return
    if args.comando in ("publicar", "retreinar"):
        publicar()


if __name__ == "__main__":
    main()
