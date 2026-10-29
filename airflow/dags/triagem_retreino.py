"""DAG de retreino do TF-IDF + LogReg: verificar dados → treinar → avaliar → gate → publicar.

A DAG apenas orquestra as tasks que chamam funções do pacote triagem, já cobertas pelos testes.
Elas rodam no venv do projeto.

Um candidato reprovado não gera falha. A task 'gate' encerra o caminho, pulando a task 'publicar'.
Rodar duas vezes com os mesmos dados demonstra isso:
o primeiro run publica (quando não há modelo em produção)
o segundo reprova (mesmo F1, mesma latência).

Disparo manual, pela UI ou por 'airflow dags trigger triagem_retreino'.
"""

from datetime import datetime

from airflow.sdk import dag, task

PYTHON_TRIAGEM = "/opt/triagem/venv/bin/python"


@dag(
    dag_id="triagem_retreino",
    schedule=None,
    start_date=datetime(2026, 10, 1),
    catchup=False,
    tags=["triagem", "retreino"],
    doc_md=__doc__,
)
def triagem_retreino():
    @task.external_python(python=PYTHON_TRIAGEM, expect_airflow=False)
    def verificar_dados() -> dict:
        """Os CSVs existem e passam na validação de schema (texto não vazio, rótulo de 1 a 5)."""
        from triagem.config import CONFIG
        from triagem.data import carregar_rotulos, carregar_treino

        treino = carregar_treino(CONFIG.dir_raw)  # levanta ErroSchema se o schema não bater
        rotulos = carregar_rotulos(CONFIG.dir_raw)
        print(f"treino: {len(treino)} linhas, {len(rotulos)} classes em {CONFIG.dir_raw}")
        return {"linhas": len(treino), "classes": len(rotulos)}

    @task.external_python(python=PYTHON_TRIAGEM, expect_airflow=False)
    def treinar() -> str:
        """Treina o candidato e grava artefato + metadata em models/tfidf/candidato."""
        import logging

        from triagem.train import treinar_candidato

        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
        return str(treinar_candidato())

    @task.external_python(python=PYTHON_TRIAGEM, expect_airflow=False)
    def avaliar() -> dict:
        """Gate: candidato × produção na mesma validação e no mesmo benchmark de latência."""
        import logging

        from triagem.train import avaliar_candidato

        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
        decisao = avaliar_candidato()
        return {
            "aprovado": decisao.aprovado,
            "criterio": decisao.criterio,
            "motivo": decisao.motivo,
        }

    # Roda no próprio Airflow: só lê a decisão. Retornar False pula tudo o que vem depois.
    @task.short_circuit
    def gate(decisao: dict) -> bool:
        print(f"{'APROVADO' if decisao['aprovado'] else 'REPROVADO'} — {decisao['motivo']}")
        return decisao["aprovado"]

    @task.external_python(python=PYTHON_TRIAGEM, expect_airflow=False)
    def publicar() -> str:
        """Troca o modelo em produção (models/tfidf/logreg) pelo candidato aprovado."""
        import logging

        from triagem.train import publicar as publicar_candidato

        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
        return str(publicar_candidato())

    decisao = avaliar()
    verificar_dados() >> treinar() >> decisao
    gate(decisao) >> publicar()


triagem_retreino()
