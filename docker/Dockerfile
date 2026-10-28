# Imagem da API de triagem com somente o necessário para servir.
# O modelo entra por volume em /app/models, assim a troca de modelo não exige rebuild.
#
# Build (na raiz do repositório):  docker build -f docker/Dockerfile -t triagem-api .
# Run:  docker run --rm -p 8000:8000 -v "$PWD/models:/app/models:ro" triagem-api

FROM python:3.12-slim

# POETRY_VIRTUALENVS_CREATE=false instala no Python do sistema: o container já é o ambiente isolado.
ENV POETRY_VERSION=2.4.1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

WORKDIR /app

# Dependências antes do código: enquanto pyproject e lock não mudarem, esta camada vem do cache e um
# rebuild depois de editar o código não reinstala nada. Só o grupo main — train, rastreamento e dev
# ficam de fora.
COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-root && rm -rf /root/.cache

COPY src/ ./src/
RUN poetry install --only main

# Usuário não-root, por segurança.
RUN useradd --create-home --uid 1000 app
USER app

EXPOSE 8000

# Saudável = pronto para atender (/ready), não só vivo (/health): sem modelo no volume, o container
# aparece como unhealthy. Sem curl na imagem slim, o teste usa o próprio Python.
HEALTHCHECK --interval=15s --timeout=3s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/ready')"]

CMD ["uvicorn", "triagem.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
