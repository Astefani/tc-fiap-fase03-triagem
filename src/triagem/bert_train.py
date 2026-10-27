"""Fine-tuning do PubMedBERT (BiomedBERT) no split do projeto.

Roda nativo numa máquina com GPU (CUDA) — fora do Docker e do Airflow. Exige o grupo 'train'
(poetry install --with train). Hiperparâmetros validados no '02-comparacao-modelos': lr 2e-5,
3 épocas, batch 16, max_length 256, e o modelo da **última** época.

Saída em 'models/bert-trabalho/' (fora do Git): pesos e tokenizer
em 'hf/' e o 'metadata.json' do treino. O que é servido sai do 'bert_export'.

Uso:
    python -m triagem.bert_train                         # treino completo
    python -m triagem.bert_train --max-amostras 200 --epocas 1 --saida /tmp/bert   # teste rápido
"""

import argparse
import logging
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

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
from triagem.evaluate import calcular_metricas
from triagem.registro import registrar_execucao

log = logging.getLogger(__name__)

MODELO_BASE = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
MAX_LENGTH = 256
LR = 2e-5
EPOCAS = 3
BATCH = 16


def dispositivo() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class DatasetTexto(torch.utils.data.Dataset):
    """Tokeniza uma vez, sem padding: o collator completa cada batch só até o maior texto dele."""

    def __init__(self, tokenizer, textos: pd.Series, rotulos: pd.Series):
        self.codificado = tokenizer(list(textos), truncation=True, max_length=MAX_LENGTH)
        self.rotulos = [int(r) - 1 for r in rotulos]  # dataset 1..5 → modelo 0..4

    def __len__(self) -> int:
        return len(self.rotulos)

    def __getitem__(self, i: int) -> dict:
        return {**{k: v[i] for k, v in self.codificado.items()}, "labels": self.rotulos[i]}


def _reduzir(dados: pd.DataFrame, max_amostras: int | None, seed: int) -> pd.DataFrame:
    if not max_amostras or max_amostras >= len(dados):
        return dados
    frac = max_amostras / len(dados)
    return dados.groupby(COLUNA_ROTULO, group_keys=False).sample(frac=frac, random_state=seed)


def treinar_bert(
    saida: Path,
    epocas: int = EPOCAS,
    max_amostras: int | None = None,
    cfg: Configuracao = CONFIG,
) -> Path:
    set_seed(cfg.seed)
    treino = carregar_treino(cfg.dir_raw)
    tr, va = separar_validacao(treino, cfg.frac_validacao, cfg.seed)
    tr, va = _reduzir(tr, max_amostras, cfg.seed), _reduzir(va, max_amostras, cfg.seed)
    nomes = carregar_rotulos(cfg.dir_raw)
    ids = sorted(nomes)

    tokenizer = AutoTokenizer.from_pretrained(MODELO_BASE)
    modelo = AutoModelForSequenceClassification.from_pretrained(
        MODELO_BASE,
        num_labels=len(ids),
        id2label={c - 1: nomes[c] for c in ids},
        label2id={nomes[c]: c - 1 for c in ids},
    )
    ds_tr = DatasetTexto(tokenizer, tr[COLUNA_TEXTO], tr[COLUNA_ROTULO])
    ds_va = DatasetTexto(tokenizer, va[COLUNA_TEXTO], va[COLUNA_ROTULO])
    disp = dispositivo()
    log.info("treino: %d linhas, %d época(s), %s", len(ds_tr), epocas, disp)

    def compute_metrics(avaliacao):
        logits, rotulos = avaliacao
        return {"f1_macro": f1_score(rotulos, np.argmax(logits, axis=-1), average="macro")}

    with TemporaryDirectory() as temporaria:  # checkpoints não ficam no projeto
        args = TrainingArguments(
            output_dir=temporaria,
            num_train_epochs=epocas,
            learning_rate=LR,
            per_device_train_batch_size=BATCH,
            per_device_eval_batch_size=BATCH * 2,
            weight_decay=0.01,
            warmup_steps=int(0.1 * np.ceil(len(ds_tr) / BATCH) * epocas),
            eval_strategy="epoch",
            save_strategy="no",  # modelo da última época, como na CV pareada
            logging_steps=50,
            bf16=disp == "cuda" and torch.cuda.is_bf16_supported(),
            seed=cfg.seed,
            report_to="none",  # o rastreamento é um run por treino, via registrar_execucao
        )
        trainer = Trainer(
            model=modelo,
            args=args,
            train_dataset=ds_tr,
            eval_dataset=ds_va,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_metrics,
        )
        inicio = time.perf_counter()
        trainer.train()
        minutos = (time.perf_counter() - inicio) / 60
        logits = trainer.predict(ds_va).predictions

    pasta_hf = saida / "hf"
    trainer.save_model(pasta_hf)
    tokenizer.save_pretrained(pasta_hf)  # grava o tokenizer.json que a API vai usar

    previsto = logits.argmax(axis=-1) + 1
    metricas = calcular_metricas(
        va[COLUNA_ROTULO],
        previsto,
        normalizar_texto(va[COLUNA_TEXTO]),
        conjuntos_de_rotulos(treino),
    )
    metricas["treino_min"] = round(minutos, 2)
    f1_por_epoca = [
        round(e["eval_f1_macro"], 4) for e in trainer.state.log_history if "eval_f1_macro" in e
    ]
    registrar_execucao(
        saida,
        nome="bert-treino",
        params={
            "modelo_base": MODELO_BASE,
            "max_length": MAX_LENGTH,
            "lr": LR,
            "epocas": epocas,
            "batch": BATCH,
            "max_amostras": max_amostras,
        },
        metricas=metricas,
        extras={
            "dispositivo": disp,
            "torch": torch.__version__,
            "classes": nomes,
            "max_length": MAX_LENGTH,
            "f1_por_epoca": f1_por_epoca,
            "linhas_treino": len(tr),
        },
    )
    log.info("F1-macro %.4f | %.1f min | salvo em %s", metricas["f1_macro"], minutos, saida)
    return saida


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fine-tuning do PubMedBERT.")
    parser.add_argument("--saida", type=Path, default=CONFIG.dir_modelos / "bert-trabalho")
    parser.add_argument("--epocas", type=int, default=EPOCAS)
    parser.add_argument("--max-amostras", type=int, default=None, help="só para teste rápido")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", force=True
    )
    treinar_bert(args.saida, args.epocas, args.max_amostras)


if __name__ == "__main__":
    main()
