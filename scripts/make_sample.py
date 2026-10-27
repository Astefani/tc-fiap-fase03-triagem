"""Gera 'tests/data/sample.csv': amostra pequena e estratificada do treino oficial.

A amostra vai para o Git e permite que a suíte de testes rode no CI sem DVC e sem credencial
do DagsHub. Determinística: rodar de novo produz o mesmo arquivo.

Uso: poetry run python scripts/make_sample.py
"""

from triagem.config import RAIZ, SEED
from triagem.data import COLUNA_ROTULO, carregar_treino

POR_CLASSE = 50
DESTINO = RAIZ / "tests" / "data" / "sample.csv"


def main() -> None:
    treino = carregar_treino()
    amostra = treino.groupby(COLUNA_ROTULO, group_keys=False).sample(
        n=POR_CLASSE, random_state=SEED
    )
    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    amostra.sort_index().to_csv(DESTINO, index=False)
    print(f"{DESTINO.relative_to(RAIZ)}: {len(amostra)} linhas ({POR_CLASSE} por classe)")


if __name__ == "__main__":
    main()
