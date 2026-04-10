# ---- Build stage ----
FROM python:3.11-slim AS builder

WORKDIR /app

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# ---- Runtime stage ----
FROM python:3.11-slim

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

COPY app/ ./app/
COPY agent/ ./agent/
COPY rag/ ./rag/

RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# 배포시에는 --reload 는 코드를 고칠 때마다 다시 읽어와라는 뜻이므로 절대 쓰지 않습니다.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]