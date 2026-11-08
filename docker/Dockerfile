# Optional containerized deployment. Not built/tested inside the sandbox
# used to develop this project (no Docker daemon available there) — verify
# `docker build .` locally before relying on it.
FROM python:3.11-slim

WORKDIR /srv/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV FLASK_HOST=0.0.0.0
ENV FLASK_PORT=5000
EXPOSE 5000

CMD ["python", "run.py"]
