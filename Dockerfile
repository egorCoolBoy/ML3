FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml poetry.lock* ./
RUN pip install --no-cache-dir poetry && poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi --no-root

COPY . .
RUN poetry install --no-interaction --no-ansi

RUN mkdir -p /app/data /app/data/chroma

ENV DATA_DIR=/app/data
ENV CHROMA_DIR=/app/data/chroma
ENV SQLITE_PATH=/app/data/memory.db
ENV LOG_FILE=/app/data/log_file.log

CMD ["python", "-m", "mindly.cli", "chat"]
