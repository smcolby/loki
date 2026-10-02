# Strata built for one AMD GPU architecture from a pinned commit, using AMD's TheRock ROCm wheels
FROM docker.io/library/ubuntu:24.04 AS build

ARG STRATA_REF
ARG GPU_TARGETS=gfx1100
ARG ROCM_VERSION=7.10.0a20251120

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl git libatomic1 python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Fetch only the pinned commit
WORKDIR /opt/strata
RUN test -n "${STRATA_REF}" || { echo "STRATA_REF is empty; run loki setup to write .env" >&2; exit 1; }
RUN git init -q . \
    && git remote add origin https://github.com/Niko1221/Strata.git \
    && git fetch -q --depth 1 origin "${STRATA_REF}" \
    && git checkout -q FETCH_HEAD

# Install ROCm into the venv and compile the HIP engine the way Strata's setup.py does
ENV STRATA_ROCM_VERSION=${ROCM_VERSION}
RUN python3 -m venv .venv \
    && .venv/bin/pip install --no-cache-dir --quiet -r requirements.txt \
    && .venv/bin/python -c "import setup; setup.build_engine_hip({'arch': '${GPU_TARGETS}'}, setup.get_llama_cpp())"

# Stage the engine, the server sources, the ROCm libraries it links or dlopens, and the GPU's BLAS kernels
RUN SDK=$(.venv/bin/rocm-sdk path --root) \
    && LIBS=$(find .venv/lib -maxdepth 4 -type d -path '*_rocm_sdk_libraries_*/lib') \
    && { test -d "$LIBS" || { echo "no ROCm libraries package for ${GPU_TARGETS} in .venv" >&2; exit 1; }; } \
    && mkdir -p /out/lib /out/engine \
    && cp engine/strata engine/BUILD.json /out/engine/ \
    && cp -r serve tools data requirements.txt /out/ \
    && .venv/bin/python -c "import setup, shutil; t = setup.hipblaslt_table('${GPU_TARGETS}', setup.rocm_root('${GPU_TARGETS}')[1]); t and shutil.copy(t, '/out/hipblaslt-tuning.txt')" \
    && DLOPENED="$SDK/lib/libamd_comgr.so.3 $SDK/lib/libhiprtc.so.7 $SDK/lib/libhiprtc-builtins.so.7" \
    && for lib in $DLOPENED $(for f in engine/strata $DLOPENED; do \
               LD_LIBRARY_PATH="$SDK/lib:$SDK/lib/llvm/lib:$LIBS" ldd "$f" | awk '$3 ~ /\.venv/ {print $3}'; \
           done | sort -u); do \
           cp -L "$lib" /out/lib/; \
       done \
    && cp -r "$LIBS/rocblas" "$LIBS/hipblaslt" /out/lib/

FROM docker.io/library/ubuntu:24.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libatomic1 libgomp1 python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /out /opt/strata
WORKDIR /opt/strata

# Fail the build if the engine or a staged library has an unresolved dependency
RUN missing=$(for f in engine/strata lib/*.so*; do \
                  LD_LIBRARY_PATH=/opt/strata/lib ldd "$f" | grep "not found" | sed "s|^|$f: |"; \
              done) \
    && { test -z "$missing" || { echo "$missing" >&2; exit 1; }; }
RUN python3 -m venv .venv && .venv/bin/pip install --no-cache-dir --quiet -r requirements.txt
COPY entrypoint.py /opt/strata/entrypoint.py

ENV PYTHONUNBUFFERED=1 \
    ROCBLAS_TENSILE_LIBPATH=/opt/strata/lib/rocblas/library \
    HIPBLASLT_TENSILE_LIBPATH=/opt/strata/lib/hipblaslt/library

# The model loads on the first request, so the server answers health checks right away
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD curl -sf http://localhost:8080/health || exit 1

ENTRYPOINT [".venv/bin/python", "/opt/strata/entrypoint.py"]
