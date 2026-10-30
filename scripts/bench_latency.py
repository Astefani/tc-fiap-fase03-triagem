"""Benchmark de latência: 1 texto por chamada, em percentis.

Dois modos:
- inprocess: chama o preditor direto e mede só o modelo (tokenização + inferência);
- http: chama a API (no container) e soma serialização, framework e rede local.

Backends do modo inprocess:
- tfidf-logreg, bert-onnx, bert-onnx-int8: os mesmos preditores que a API usa;
- bert-pytorch: o modelo original, sem otimização. Exige o grupo 'train' e
  'models/bert-trabalho/hf', que só existe na máquina de treino.

Protocolo: textos únicos da validação (split do projeto), warmup, N chamadas sequenciais e threads
fixas (TRIAGEM_ONNX_THREADS, também aplicado ao PyTorch). No modo inprocess, os backends são medidos
**em blocos**, um de cada vez, com uma pausa entre eles.

Por que não intercalar (lição de 29/10): PyTorch e ONNX Runtime mantêm cada um seu grupo de threads,
que continua girando em espera ativa por alguns milissegundos depois de cada inferência. Chamando os
backends texto a texto, a inferência seguinte começa com as threads da anterior ainda ocupando os
núcleos — com a CPU fixada em 4 núcleos, o INT8 chegou a sair mais lento que o fp32. `--intercalar`
continua disponível para backends sem paralelismo interno (é o que o gate faz com o TF-IDF).

Saída: tabela no terminal e um JSON em 'docs/latencia/' com máquina, versões, parâmetros e o commit
do código. O 'docs/latencia.md' é escrito a partir desses arquivos.

Uso:
    python scripts/bench_latency.py inprocess
    python scripts/bench_latency.py inprocess --backends bert-pytorch bert-onnx bert-onnx-int8
    taskset -c 0,2,4,6 python scripts/bench_latency.py inprocess --rotulo pcores   # núcleos fixos
    python scripts/bench_latency.py http --url http://localhost:8000 --rotulo cpus2
"""

import argparse
import json
import os
import platform
import subprocess
import time
from collections.abc import Callable
from datetime import datetime, timezone
from importlib.metadata import version
from itertools import cycle, islice
from pathlib import Path

import numpy as np

from triagem.config import CONFIG, RAIZ
from triagem.data import COLUNA_TEXTO, carregar_treino, separar_validacao
from triagem.predictor import carregar_preditor
from triagem.versao import versao_do_codigo

BACKENDS_SERVIDOS = ["tfidf-logreg", "bert-onnx", "bert-onnx-int8"]
DIR_PYTORCH = CONFIG.dir_modelos / "bert-trabalho" / "hf"
DIR_SAIDA = RAIZ / "docs" / "latencia"


def textos_de_validacao() -> list[str]:
    """Resumos únicos da validação: a mesma entrada para todos os backends e todas as máquinas."""
    _, validacao = separar_validacao(carregar_treino())
    return validacao[COLUNA_TEXTO].drop_duplicates().tolist()


def estatisticas(tempos_ms: list[float]) -> dict[str, float]:
    tempos = np.asarray(tempos_ms)
    p50, p95, p99 = np.percentile(tempos, [50, 95, 99])
    return {
        "p50_ms": float(p50),
        "p95_ms": float(p95),
        "p99_ms": float(p99),
        "media_ms": float(tempos.mean()),
        "max_ms": float(tempos.max()),
        # chamadas sequenciais: 1 requisição por vez, sem concorrência
        "req_por_s": float(1000 / tempos.mean()),
        "n": len(tempos),
    }


PAUSA_ENTRE_BLOCOS_S = 2.0  # tempo para as threads do backend anterior pararem de girar


def medir_em_blocos(
    funcoes: dict[str, Callable[[str], object]], textos: list[str], n: int, warmup: int
) -> dict[str, dict[str, float]]:
    """Um backend de cada vez: warmup + N chamadas, pausa, próximo backend."""
    textos = list(islice(cycle(textos), warmup + n))
    resultados = {}
    for i, (nome, prever) in enumerate(funcoes.items()):
        if i:
            time.sleep(PAUSA_ENTRE_BLOCOS_S)
        for texto in textos[:warmup]:
            prever(texto)
        tempos = []
        for texto in textos[warmup:]:
            inicio = time.perf_counter()
            prever(texto)
            tempos.append((time.perf_counter() - inicio) * 1000)
        resultados[nome] = estatisticas(tempos)
    return resultados


def medir_intercalado(
    funcoes: dict[str, Callable[[str], object]], textos: list[str], n: int, warmup: int
) -> dict[str, dict[str, float]]:
    """Para cada texto, chama todos os backends em sequência; mede cada chamada separadamente."""
    textos = list(islice(cycle(textos), warmup + n))
    for texto in textos[:warmup]:
        for prever in funcoes.values():
            prever(texto)
    tempos: dict[str, list[float]] = {nome: [] for nome in funcoes}
    for texto in textos[warmup:]:
        for nome, prever in funcoes.items():
            inicio = time.perf_counter()
            prever(texto)
            tempos[nome].append((time.perf_counter() - inicio) * 1000)
    return {nome: estatisticas(t) for nome, t in tempos.items()}


def _metadata(pasta: Path) -> dict:
    caminho = pasta / "metadata.json"
    return json.loads(caminho.read_text(encoding="utf-8")) if caminho.exists() else {}


def _mb(*arquivos: Path) -> float:
    return round(sum(a.stat().st_size for a in arquivos) / 2**20, 1)


def preparar_pytorch(threads: int) -> tuple[Callable[[str], object], dict]:
    """O BERT original, em CPU e fp32 (sem otimização)."""
    if not DIR_PYTORCH.exists():
        raise SystemExit(f"bert-pytorch: {DIR_PYTORCH} não existe (só na máquina de treino)")
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(threads)
    max_length = _metadata(CONFIG.dir_modelos / "bert")["max_length"]
    tokenizer = AutoTokenizer.from_pretrained(DIR_PYTORCH)
    modelo = AutoModelForSequenceClassification.from_pretrained(DIR_PYTORCH).eval()

    @torch.inference_mode()
    def prever(texto: str) -> int:
        entrada = tokenizer(texto, truncation=True, max_length=max_length, return_tensors="pt")
        return int(modelo(**entrada).logits.argmax(-1))

    f1 = (_metadata(CONFIG.dir_modelos / "bert").get("treino") or {}).get("metricas_pytorch", {})
    info = {
        "tamanho_mb": _mb(*DIR_PYTORCH.glob("*.safetensors")),
        "f1_macro": f1.get("f1_macro"),
        "torch": torch.__version__,
    }
    return prever, info


def preparar_servido(backend: str) -> tuple[Callable[[str], object], dict]:
    preditor = carregar_preditor(backend)
    if backend == "tfidf-logreg":
        pasta = CONFIG.dir_modelos / "tfidf" / "logreg"
        meta = _metadata(pasta)
        return preditor.prever, {
            "tamanho_mb": _mb(pasta / meta.get("artefato", "pipeline.joblib")),
            "f1_macro": meta["metricas"]["f1_macro"],
            "versao_modelo": preditor.versao_modelo,
        }
    pasta = CONFIG.dir_modelos / "bert"
    arquivo = "model.onnx" if backend == "bert-onnx" else "model.int8.onnx"
    return preditor.prever, {
        "tamanho_mb": _mb(pasta / arquivo),
        "f1_macro": _metadata(pasta)["metricas"][backend]["f1_macro"],
        "versao_modelo": preditor.versao_modelo,
    }


def modo_inprocess(
    backends: list[str], textos: list[str], n: int, warmup: int, intercalar: bool
) -> dict:
    funcoes, info = {}, {}
    for backend in backends:
        if backend == "bert-pytorch":
            funcoes[backend], info[backend] = preparar_pytorch(CONFIG.onnx_threads)
        else:
            funcoes[backend], info[backend] = preparar_servido(backend)
    medir = medir_intercalado if intercalar else medir_em_blocos
    medidas = medir(funcoes, textos, n, warmup)
    return {b: {**medidas[b], **info[b]} for b in backends}


def modo_http(url: str, textos: list[str], n: int, warmup: int) -> dict:
    import httpx

    with httpx.Client(base_url=url, timeout=30) as cliente:  # keep-alive: mede a API, não o TCP
        pronto = cliente.get("/ready")
        pronto.raise_for_status()
        backend = pronto.json()["backend"]

        def prever(texto: str) -> None:
            cliente.post("/predict", json={"texto": texto}).raise_for_status()

        medidas = medir_em_blocos({backend: prever}, textos, n, warmup)
    return {backend: {**medidas[backend], "versao_modelo": pronto.json()["versao_modelo"]}}


def _cpu() -> str:
    if platform.system() == "Darwin":
        comando = ["sysctl", "-n", "machdep.cpu.brand_string"]
        return subprocess.run(comando, capture_output=True, text=True).stdout.strip()
    try:
        for linha in Path("/proc/cpuinfo").read_text().splitlines():
            if linha.startswith("model name"):
                return linha.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def ambiente() -> dict:
    return {
        "sistema": platform.platform(),
        "arquitetura": platform.machine(),
        "cpu": _cpu(),
        "nucleos": os.cpu_count(),
        "python": platform.python_version(),
        "onnxruntime": version("onnxruntime"),
        "scikit-learn": version("scikit-learn"),
        "tokenizers": version("tokenizers"),
        "codigo": versao_do_codigo(),
    }


def imprimir(resultados: dict) -> None:
    colunas = ["p50_ms", "p95_ms", "p99_ms", "req_por_s", "tamanho_mb", "f1_macro"]
    print(f"\n{'backend':<16}" + "".join(f"{c:>12}" for c in colunas))
    for backend, r in resultados.items():
        valores = [r.get(c) for c in colunas]
        print(
            f"{backend:<16}"
            + "".join(f"{v:>12.2f}" if v is not None else f"{'—':>12}" for v in valores)
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("modo", choices=["inprocess", "http"])
    parser.add_argument("--backends", nargs="+", default=BACKENDS_SERVIDOS)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("-n", type=int, default=1000, help="chamadas medidas por backend")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--intercalar", action="store_true", help="inprocess: texto a texto")
    parser.add_argument("--rotulo", default="", help="sufixo do arquivo (ex.: pcores)")
    parser.add_argument("--sem-salvar", action="store_true")
    args = parser.parse_args(argv)

    textos = textos_de_validacao()
    if args.modo == "inprocess":
        resultados = modo_inprocess(args.backends, textos, args.n, args.warmup, args.intercalar)
    else:
        resultados = modo_http(args.url, textos, args.n, args.warmup)
    imprimir(resultados)

    if args.sem_salvar:
        return
    agora = datetime.now(timezone.utc)
    relatorio = {
        "criado_em": agora.isoformat(timespec="seconds"),
        "modo": args.modo,
        "parametros": {
            "n": args.n,
            "warmup": args.warmup,
            "threads": CONFIG.onnx_threads,
            "ordem": "intercalada" if args.intercalar else "blocos",
            "afinidade_cpu": sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None,
            "textos_unicos": len(textos),
            "url": args.url if args.modo == "http" else None,
            "rotulo": args.rotulo or None,
        },
        "ambiente": ambiente(),
        "resultados": resultados,
    }
    partes = [agora.strftime("%Y-%m-%d_%H%M"), platform.machine(), args.modo, args.rotulo]
    destino = DIR_SAIDA / ("_".join(p for p in partes if p) + ".json")
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(relatorio, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nsalvo em {destino.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
