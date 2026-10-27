"""Export do PubMedBERT treinado para ONNX fp32 e INT8, com as checagens que liberam o artefato.

Lê 'models/bert-trabalho/hf/' (saída do 'bert_train') e grava em 'models/bert/' **o que é servido**:
'model.onnx', 'model.int8.onnx', 'tokenizer.json' e 'metadata.json'.
É essa pasta que o DVC versiona.

Checagens, nesta ordem (qualquer falha interrompe):
1. paridade numérica: logits do PyTorch × ONNX fp32 iguais dentro da tolerância;
2. caminho de produção: o 'PreditorBertOnnx' (tokenizers + onnxruntime, sem torch) dá a mesma classe
   que o PyTorch — prova que a tokenização da API bate com a do treino;
3. qualidade de cada variante (fp32 e INT8) na validação, pelo mesmo preditor que a API usa.

Exige o grupo 'train'. Uso: python -m triagem.bert_export
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from triagem.config import CONFIG, Configuracao
from triagem.data import (
    COLUNA_ROTULO,
    COLUNA_TEXTO,
    carregar_treino,
    conjuntos_de_rotulos,
    normalizar_texto,
    separar_validacao,
)
from triagem.evaluate import calcular_metricas
from triagem.predictor import ARQUIVOS_BERT, PreditorBertOnnx
from triagem.registro import registrar_execucao

log = logging.getLogger(__name__)

ENTRADAS = ["input_ids", "attention_mask", "token_type_ids"]
OPSET = 17
TOLERANCIA_LOGITS = 1e-3
N_PARIDADE = 20


def _exportar_onnx(modelo, tokenizer, caminho: Path) -> None:
    exemplo = tokenizer("texto de exemplo para o export", return_tensors="pt")
    torch.onnx.export(
        modelo,
        tuple(exemplo[k] for k in ENTRADAS),
        caminho,
        input_names=ENTRADAS,
        output_names=["logits"],
        # eixos dinâmicos: o grafo aceita qualquer batch e qualquer comprimento de texto
        dynamic_axes={**{k: {0: "batch", 1: "seq"} for k in ENTRADAS}, "logits": {0: "batch"}},
        opset_version=OPSET,
        dynamo=False,  # exportador clássico (TorchScript)
    )


def exportar(
    origem: Path, destino: Path, max_validacao: int | None = None, cfg: Configuracao = CONFIG
) -> Path:
    meta_treino = json.loads((origem.parent / "metadata.json").read_text())
    max_length = meta_treino["max_length"]
    tokenizer = AutoTokenizer.from_pretrained(origem)
    # "eager" = atenção em operações simples, que exportam bem para ONNX
    modelo = AutoModelForSequenceClassification.from_pretrained(
        origem, attn_implementation="eager"
    ).eval()

    if destino.exists():
        shutil.rmtree(destino)
    destino.mkdir(parents=True)
    _exportar_onnx(modelo, tokenizer, destino / ARQUIVOS_BERT["bert-onnx"])
    quantize_dynamic(
        destino / ARQUIVOS_BERT["bert-onnx"],
        destino / ARQUIVOS_BERT["bert-onnx-int8"],
        weight_type=QuantType.QInt8,
    )
    shutil.copy(origem / "tokenizer.json", destino / "tokenizer.json")
    # metadata provisório: o preditor precisa de max_length e classes para as checagens abaixo
    provisorio = {
        "nome": "bert-onnx",
        "criado_em": "provisorio",
        "max_length": max_length,
        "classes": meta_treino["classes"],
    }
    (destino / "metadata.json").write_text(json.dumps(provisorio))

    treino = carregar_treino(cfg.dir_raw)
    _, va = separar_validacao(treino, cfg.frac_validacao, cfg.seed)
    if max_validacao:
        va = va.head(max_validacao)
    textos = va[COLUNA_TEXTO].tolist()

    # 1 e 2. paridade com o PyTorch
    with torch.inference_mode():
        codificado = tokenizer(
            textos[:N_PARIDADE],
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        logits_torch = modelo(**codificado).logits.numpy()
    preditores = {b: PreditorBertOnnx(destino, b, cfg.onnx_threads) for b in ARQUIVOS_BERT}
    fp32 = preditores["bert-onnx"]
    logits_onnx = np.vstack(
        [
            fp32.sessao.run(
                ["logits"],
                {k: codificado[k][i : i + 1].numpy().astype(np.int64) for k in ENTRADAS},
            )[0]
            for i in range(len(codificado["input_ids"]))
        ]
    )
    diferenca = float(np.abs(logits_onnx - logits_torch).max())
    if diferenca > TOLERANCIA_LOGITS:
        raise RuntimeError(
            f"ONNX fp32 diverge do PyTorch: diferença máxima de logit {diferenca:.5f}"
        )
    classes_torch = logits_torch.argmax(axis=1) + 1
    classes_api = np.array([fp32.prever(t).classe_id for t in textos[:N_PARIDADE]])
    mesma_classe = float(np.mean(classes_api == classes_torch))
    if mesma_classe < 1.0:
        raise RuntimeError(
            f"caminho da API diverge do PyTorch em {1 - mesma_classe:.0%} dos textos"
        )

    # 3. qualidade de cada variante, pelo mesmo caminho que a API usa
    conjuntos = conjuntos_de_rotulos(treino)
    metricas = {}
    for backend, preditor in preditores.items():
        previsto = [preditor.prever(t).classe_id for t in textos]
        metricas[backend] = calcular_metricas(
            va[COLUNA_ROTULO], previsto, normalizar_texto(va[COLUNA_TEXTO]), conjuntos
        )
        log.info("%s: F1-macro %.4f", backend, metricas[backend]["f1_macro"])

    tamanhos = {
        arquivo: round((destino / arquivo).stat().st_size / 2**20, 1)
        for arquivo in ARQUIVOS_BERT.values()
    }
    registrar_execucao(
        destino,
        nome="bert-onnx",
        params={"opset": OPSET, "quantizacao": "dinamica-int8", "max_length": max_length},
        metricas=metricas,
        extras={
            "classes": meta_treino["classes"],
            "max_length": max_length,
            "paridade": {"max_diff_logits": diferenca, "mesma_classe_api_x_torch": mesma_classe},
            "tamanho_mb": tamanhos,
            "treino": {
                "criado_em": meta_treino["criado_em"],
                "codigo": meta_treino.get("codigo"),
                "metricas_pytorch": meta_treino["metricas"],
                "mlflow_run_id": meta_treino.get("mlflow_run_id"),
            },
        },
    )
    log.info("exportado em %s (%s)", destino, tamanhos)
    return destino


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export do PubMedBERT para ONNX fp32 e INT8.")
    parser.add_argument("--origem", type=Path, default=CONFIG.dir_modelos / "bert-trabalho" / "hf")
    parser.add_argument("--destino", type=Path, default=CONFIG.dir_modelos / "bert")
    parser.add_argument("--max-validacao", type=int, default=None, help="só para teste rápido")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", force=True
    )
    exportar(args.origem, args.destino, args.max_validacao)


if __name__ == "__main__":
    main()
