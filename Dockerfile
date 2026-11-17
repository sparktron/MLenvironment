FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml requirements-lock.txt ./
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements-lock.txt

COPY . .
RUN pip install --no-cache-dir --no-deps -e .

ENTRYPOINT ["python", "-m", "rl_framework.cli.main"]
