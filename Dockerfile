FROM python:3.12-slim AS builder
WORKDIR /install
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --prefix=/install/deps --no-cache-dir -r requirements.txt

FROM python:3.12-slim AS runtime
RUN useradd -m -u 1000 appuser
WORKDIR /app
COPY --from=builder /install/deps /usr/local
COPY app/          ./app/
COPY scripts/      ./scripts/
COPY entrypoint.sh ./entrypoint.sh
RUN chmod +x entrypoint.sh && mkdir -p models && chown -R appuser:appuser /app
USER appuser
ENV PORT=8080 MODEL_DIR=/app/models PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"
ENTRYPOINT ["./entrypoint.sh"]
