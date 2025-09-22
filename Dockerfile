ARG PYTORCH_TARGET=cpu

FROM python:3.12.2-slim-bookworm AS cpu_base
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS gpu_base

FROM ${PYTORCH_TARGET}_base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN if [ "$PYTORCH_TARGET" = "gpu" ]; then \
        apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3-pip \
        && rm -rf /var/lib/apt/lists/* \
        && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1; \
    fi

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "src/ai_math_tutor/app.py", "--server.port=8501", "--server.address=0.0.0.0"]