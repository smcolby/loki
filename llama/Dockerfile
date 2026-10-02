# llama-server built for one AMD GPU architecture from a pinned llama.cpp commit
ARG ROCM_VERSION=7.2.4

FROM docker.io/rocm/dev-ubuntu-24.04:${ROCM_VERSION}-complete AS build

ARG LLAMA_CPP_REF
ARG GPU_TARGETS=gfx1100

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates cmake git ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Fetch only the pinned commit
WORKDIR /src
RUN test -n "${LLAMA_CPP_REF}" || { echo "LLAMA_CPP_REF is empty; run loki setup to write .env" >&2; exit 1; }
RUN git init -q . \
    && git remote add origin https://github.com/ggml-org/llama.cpp.git \
    && git fetch -q --depth 1 origin "${LLAMA_CPP_REF}" \
    && git checkout -q FETCH_HEAD

# Compile the server against the HIP backend for the target architecture only
RUN HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
    cmake -S . -B build -G Ninja \
        -DGGML_HIP=ON \
        -DGPU_TARGETS="${GPU_TARGETS}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
    && cmake --build build --target llama-server

# Stage the server, its shared libraries, and the ROCm runtime it links against
RUN mkdir -p /out/bin /out/lib \
    && cp build/bin/llama-server /out/bin/ \
    && cp -P build/bin/*.so* /out/lib/ \
    && for lib in $(ldd build/bin/llama-server build/bin/libggml-hip.so | awk '$3 ~ /^\/opt\/rocm/ {print $3}' | sort -u); do \
           cp -L "$lib" /out/lib/; \
       done \
    && cp -r /opt/rocm/lib/rocblas /opt/rocm/lib/hipblaslt /out/lib/

# Drop BLAS kernels built for other architectures (about 4.5GB)
RUN find /out/lib/rocblas /out/lib/hipblaslt -type f -name '*gfx*' ! -name "*${GPU_TARGETS}*" -delete

FROM docker.io/library/ubuntu:24.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl libdrm-amdgpu1 libdrm2 libelf1t64 libgomp1 libnuma1 libzstd1 zlib1g \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /out /opt/llama

ENV LD_LIBRARY_PATH=/opt/llama/lib \
    ROCBLAS_TENSILE_LIBPATH=/opt/llama/lib/rocblas/library \
    HIPBLASLT_TENSILE_LIBPATH=/opt/llama/lib/hipblaslt/library

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD curl -sf http://localhost:8080/health || exit 1

ENTRYPOINT ["/opt/llama/bin/llama-server"]
