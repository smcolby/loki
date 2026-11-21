# Breeze TTS 2 on ROCm PyTorch, run on demand by a supervisor that also serves the Studio
ARG ROCM_VERSION=7.2.4
ARG TORCH_VERSION=2.9.1

FROM docker.io/rocm/pytorch:rocm${ROCM_VERSION}_ubuntu24.04_py3.12_pytorch_release_${TORCH_VERSION}

ARG BREEZE_REF

# Fetch only the pinned Breeze commit
WORKDIR /opt/breeze
RUN test -n "${BREEZE_REF}" || { echo "BREEZE_REF is empty; run loki setup to write the speech service" >&2; exit 1; }
RUN git init -q . \
    && git remote add origin https://github.com/breezeblue-ai/breeze-tts.git \
    && git fetch -q --depth 1 origin "${BREEZE_REF}" \
    && git checkout -q FETCH_HEAD

# Pin the base image's ROCm torch stack so no dependency replaces it with a CUDA wheel, then
# install Breeze's dependencies; qwen-tts goes in without its dependencies, which pull in gradio
COPY constraints.txt /tmp/constraints.txt
RUN pip freeze | grep -E '^(torch|torchaudio|torchvision|triton|pytorch-triton)' > /tmp/torch-pins.txt \
    && pip install --no-cache-dir -c /tmp/torch-pins.txt -c /tmp/constraints.txt \
        transformers accelerate huggingface-hub numpy soundfile fastapi uvicorn python-multipart \
        librosa onnxruntime einops sox soxr aiohttp av \
    && pip install --no-cache-dir --no-deps qwen-tts==0.1.1 \
    && pip freeze | grep -E '^(torch|torchaudio|triton|transformers|qwen-tts)=='

COPY speech_supervisor.py /opt/speech/speech_supervisor.py
COPY studio/ /opt/speech/studio/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Breeze starts on the first generation request, so the supervisor answers health checks right away
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s CMD curl -sf http://localhost:8080/health || exit 1

ENTRYPOINT ["python", "/opt/speech/speech_supervisor.py"]
