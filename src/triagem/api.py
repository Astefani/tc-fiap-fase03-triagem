"""API de triagem: recebe o texto e devolve a classe prevista.

O modelo é carregado uma vez, na subida. Se o carregamento falhar, a API sobe mesmo assim:
/health responde (o processo está vivo)
/ready e /predict respondem 503 (o serviço não está pronto, diferenciando liveness e readiness).

O texto do laudo **nunca** vai para o log: é dado de saúde (LGPD, art. 11). Loga-se só o tamanho, a
classe e o tempo de inferência.

Uso: uvicorn triagem.api:app --host 0.0.0.0 --port 8000
"""

import logging
import time
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from triagem import metrics
from triagem.config import CONFIG, Configuracao
from triagem.predictor import Preditor, carregar_preditor
from triagem.schemas import EntradaPredicao, SaidaPredicao, Status

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("triagem.api")


def _preditor_pronto(request: Request) -> Preditor:
    preditor = request.app.state.preditor
    if preditor is None:
        raise HTTPException(status_code=503, detail="modelo não carregado")
    return preditor


def criar_app(cfg: Configuracao = CONFIG) -> FastAPI:
    """Monta a aplicação; os testes passam a própria configuração (modelos em pasta temporária)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.preditor = None
        metrics.MODELO_CARREGADO.set(0)
        try:
            preditor = carregar_preditor(cfg=cfg)
            app.state.preditor = preditor
            metrics.MODELO_CARREGADO.set(1)
            metrics.MODELO.info({"backend": preditor.backend, "versao": preditor.versao_modelo})
            log.info("modelo carregado: %s %s", preditor.backend, preditor.versao_modelo)
        except Exception:
            log.exception("falha ao carregar o modelo %s de %s", cfg.model_backend, cfg.dir_modelos)
        yield

    app = FastAPI(
        title="Triagem de laudos médicos",
        description="Classifica o texto de um laudo em 5 categorias de condição.",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Mede toda requisição, inclusive as que terminam em 422 ou 503. O rótulo é o molde da rota
    # ("/predict"), não o caminho pedido: um caminho inventado não cria série nova no Prometheus.
    @app.middleware("http")
    async def medir(request: Request, call_next):
        inicio = time.perf_counter()
        status = 500  # se o handler levantar exceção, conta como erro interno
        try:
            resposta = await call_next(request)
            status = resposta.status_code
            return resposta
        finally:
            rota = request.scope.get("route")
            endpoint = rota.path if rota else "outro"
            if endpoint != "/metrics":
                metrics.REQUISICOES.labels(endpoint, request.method, str(status)).inc()
                metrics.LATENCIA_HTTP.labels(endpoint).observe(time.perf_counter() - inicio)

    @app.get("/", include_in_schema=False)
    def raiz():
        return RedirectResponse("/docs")

    @app.get("/metrics", include_in_schema=False)
    def expor_metricas():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/health", response_model=Status)
    def health():
        return Status(status="ok")

    @app.get(
        "/ready", response_model=Status, responses={503: {"description": "Modelo não carregado"}}
    )
    def ready(request: Request):
        preditor = _preditor_pronto(request)
        return Status(
            status="pronto", backend=preditor.backend, versao_modelo=preditor.versao_modelo
        )

    @app.post(
        "/predict",
        response_model=SaidaPredicao,
        responses={503: {"description": "Modelo não carregado"}},
    )
    def predict(entrada: EntradaPredicao, request: Request):
        preditor = _preditor_pronto(request)
        inicio = time.perf_counter()
        predicao = preditor.prever(entrada.texto)
        segundos = time.perf_counter() - inicio
        ms = segundos * 1000
        metrics.LATENCIA_INFERENCIA.labels(predicao.backend).observe(segundos)
        metrics.PREDICOES.labels(predicao.backend, predicao.classe).inc()
        metrics.TAMANHO_TEXTO.observe(len(entrada.texto))
        log.info(
            "predicao backend=%s classe=%d caracteres=%d inferencia_ms=%.1f",
            predicao.backend,
            predicao.classe_id,
            len(entrada.texto),
            ms,
        )
        return SaidaPredicao(**asdict(predicao))

    return app


app = criar_app()
