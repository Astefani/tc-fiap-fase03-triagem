"""Gerador de carga: tráfego contínuo na API para popular o Grafana.

Envia resumos da validação para /predict e, numa fração das requisições, um texto curto demais
(→ 422), para o painel de erros ter o que mostrar. Com '--concorrencia' > 1, várias requisições
ficam em andamento ao mesmo tempo: é o cenário em que as threads do ONNX Runtime disputam os
núcleos.

Sem os dados reais (antes do 'dvc pull'), usa a amostra de tests/data/sample.csv.

Uso:
    python scripts/load_gen.py                                    # 5 req/s por 5 min
    python scripts/load_gen.py --rps 20 --duracao 120 --concorrencia 4 --erros 0.05
    python scripts/load_gen.py --rps 0 --duracao 60 --concorrencia 8  # sem limite de taxa
"""

import argparse
import random
import threading
import time
from collections import Counter

import httpx
import numpy as np
import pandas as pd

from triagem.config import CONFIG, RAIZ
from triagem.data import COLUNA_TEXTO, carregar_treino, separar_validacao

AMOSTRA = RAIZ / "tests" / "data" / "sample.csv"
TEXTO_INVALIDO = "curto"  # abaixo do mínimo do contrato → 422


def carregar_textos() -> list[str]:
    if (CONFIG.dir_raw / "medical_tc_train.csv").exists():
        _, validacao = separar_validacao(carregar_treino())
        return validacao[COLUNA_TEXTO].drop_duplicates().tolist()
    print("dados reais ausentes: usando tests/data/sample.csv")
    return pd.read_csv(AMOSTRA)[COLUNA_TEXTO].tolist()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument(
        "--rps", type=float, default=5, help="requisições/s no total (0 = sem limite)"
    )
    parser.add_argument("--duracao", type=float, default=300, help="segundos")
    parser.add_argument("--concorrencia", type=int, default=1, help="requisições simultâneas")
    parser.add_argument("--erros", type=float, default=0.05, help="fração de requisições inválidas")
    parser.add_argument("--seed", type=int, default=CONFIG.seed)
    args = parser.parse_args(argv)

    textos = carregar_textos()
    status: Counter[str] = Counter()
    tempos_ms: list[float] = []
    trava = threading.Lock()
    fim = time.monotonic() + args.duracao
    # cada worker segura uma fatia da taxa total
    intervalo = args.concorrencia / args.rps if args.rps > 0 else 0.0

    def worker(indice: int) -> None:
        sorteio = random.Random(args.seed + indice)
        proximo = time.monotonic() + sorteio.random() * intervalo  # espalha a largada
        with httpx.Client(base_url=args.url, timeout=30) as cliente:
            while time.monotonic() < fim:
                if intervalo:
                    time.sleep(max(0.0, proximo - time.monotonic()))
                    proximo += intervalo
                texto = TEXTO_INVALIDO if sorteio.random() < args.erros else sorteio.choice(textos)
                inicio = time.perf_counter()
                try:
                    codigo = str(cliente.post("/predict", json={"texto": texto}).status_code)
                except httpx.HTTPError as erro:
                    codigo = type(erro).__name__
                ms = (time.perf_counter() - inicio) * 1000
                with trava:
                    status[codigo] += 1
                    if codigo == "200":
                        tempos_ms.append(ms)

    print(
        f"{args.url}/predict · {args.rps or 'sem limite de'} req/s · {args.duracao:.0f} s · "
        f"concorrência {args.concorrencia} · {args.erros:.0%} inválidas"
    )
    workers = [
        threading.Thread(target=worker, args=(i,), daemon=True) for i in range(args.concorrencia)
    ]
    inicio = time.monotonic()
    for t in workers:
        t.start()
    try:
        for t in workers:  # acompanha o progresso a cada 10 s até o último terminar
            while t.is_alive():
                t.join(timeout=10)
                with trava:
                    total = sum(status.values())
                print(f"  {time.monotonic() - inicio:5.0f} s · {total} requisições")
    except KeyboardInterrupt:
        print("interrompido")
    decorrido = time.monotonic() - inicio

    total = sum(status.values())
    taxa = total / decorrido
    print(f"\n{total} requisições em {decorrido:.0f} s ({taxa:.1f} req/s) · {dict(status)}")
    if tempos_ms:
        p50, p95, p99 = np.percentile(tempos_ms, [50, 95, 99])
        print(f"latência no cliente (200): p50 {p50:.1f} · p95 {p95:.1f} · p99 {p99:.1f} ms")


if __name__ == "__main__":
    main()
