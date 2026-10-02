# stable-diffusion.cpp's sd-server built for one AMD GPU architecture from a pinned commit,
# run on demand by a supervisor that frees the GPU on unload
ARG ROCM_VERSION=7.2.4

FROM docker.io/rocm/dev-ubuntu-24.04:${ROCM_VERSION}-complete AS build

ARG SD_CPP_REF
ARG GPU_TARGETS=gfx1100

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates cmake git ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Fetch only the pinned commit and the submodules the server links
WORKDIR /src
RUN test -n "${SD_CPP_REF}" || { echo "SD_CPP_REF is empty; run loki setup to write the image services" >&2; exit 1; }
RUN git init -q . \
    && git remote add origin https://github.com/leejet/stable-diffusion.cpp.git \
    && git fetch -q --depth 1 origin "${SD_CPP_REF}" \
    && git checkout -q FETCH_HEAD \
    && git submodule update -q --init --depth 1 ggml thirdparty/libwebp

# Compile the server against the HIP backend for the target architecture only
RUN cmake -S . -B build -G Ninja \
        -DCMAKE_C_COMPILER="$(hipconfig -l)/clang" \
        -DCMAKE_CXX_COMPILER="$(hipconfig -l)/clang++" \
        -DSD_HIPBLAS=ON \
        -DSD_WEBM=OFF \
        -DSD_SERVER_BUILD_FRONTEND=OFF \
        -DGPU_TARGETS="${GPU_TARGETS}" \
        -DAMDGPU_TARGETS="${GPU_TARGETS}" \
        -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --target sd-server

# Stage the server and the ROCm runtime it links against
RUN mkdir -p /out/bin /out/lib \
    && cp build/bin/sd-server /out/bin/ \
    && for lib in $(ldd build/bin/sd-server | awk '$3 ~ /^\/opt\/rocm/ {print $3}' | sort -u); do \
           cp -L "$lib" /out/lib/; \
       done \
    && cp -r /opt/rocm/lib/rocblas /opt/rocm/lib/hipblaslt /out/lib/

# Drop BLAS kernels built for other architectures (about 4.5GB)
RUN find /out/lib/rocblas /out/lib/hipblaslt -type f -name '*gfx*' ! -name "*${GPU_TARGETS}*" -delete

FROM docker.io/library/ubuntu:24.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl libdrm-amdgpu1 libdrm2 libelf1t64 libgomp1 libnuma1 libzstd1 python3 zlib1g \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /out /opt/sd
COPY sd_supervisor.py /opt/sd/sd_supervisor.py

ENV PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH=/opt/sd/lib \
    ROCBLAS_TENSILE_LIBPATH=/opt/sd/lib/rocblas/library \
    HIPBLASLT_TENSILE_LIBPATH=/opt/sd/lib/hipblaslt/library

# sd-server starts on the first request, so the supervisor answers health checks right away
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s CMD curl -sf http://localhost:8080/health || exit 1

ENTRYPOINT ["python3", "/opt/sd/sd_supervisor.py"]
