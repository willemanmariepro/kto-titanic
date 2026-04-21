# TODO: implémenter le Dockerfile de l'API, vous pouvez vous inspirer de ce qui est fait pour les autres projets
# (et vous référer au cours)
FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock .python-version ./
COPY ./src/titanic/api ./src/titanic/api

RUN uv sync -n --group api

EXPOSE 8080

ENTRYPOINT ["uv", "-n", "run", "--no-sync", "api"]
