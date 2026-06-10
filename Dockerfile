# Relief3D — self-contained CPU photogrammetry web app.
#
# One image: builds OpenMVG (SfM) + OpenMVS (dense/mesh/texture) from source, then
# layers the Flask app on top. The app calls the engine binaries directly (no
# sibling container, no docker socket). Build + run:
#   docker build -t relief3d .
#   docker run -p 5002:5002 -v "$PWD/data:/data" relief3d
# or:  docker compose up -d --build
#
# x86 only — OpenMVS TextureMesh needs AVX (won't run on Apple Silicon).
# Engine pins: OpenMVG v2.1, OpenMVS v2.4.0, VCGLib 658ba36 (reproducible).

# =========================================================================
#  Stage 1: Tailwind/DaisyUI CSS bundle
# =========================================================================
FROM node:20-alpine AS css-builder
WORKDIR /build
COPY tailwind.config.js input.css ./
COPY templates/ ./templates/
RUN npm init -y >/dev/null && \
    npm install --no-audit --no-fund tailwindcss@3 daisyui@4 && \
    npx tailwindcss -i ./input.css -o ./output.css --minify

# =========================================================================
#  Stage 2: engine builder — compiles OpenMVG + OpenMVS (no CUDA, headless).
#  OpenMVS is built FIRST against pristine system Eigen 3.4; OpenMVG bundles its
#  own Eigen and is built last to avoid the clash.
# =========================================================================
FROM ubuntu:26.04 AS engine-builder
ARG DEBIAN_FRONTEND=noninteractive
ARG VCG_COMMIT=658ba36d0a5666650da6e066b4794efc5a463407

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates pkg-config \
        libpng-dev libjpeg-dev libtiff-dev \
        libxxf86vm-dev libxi-dev libxrandr-dev \
        libeigen3-dev \
        libboost-all-dev \
        libnanoflann-dev \
        libjxl-dev \
        libopencv-dev \
        libcgal-dev \
        libblas-dev liblapack-dev libsuitesparse-dev \
        libceres-dev \
        libglu1-mesa-dev freeglut3-dev \
    && rm -rf /var/lib/apt/lists/*

# VCGLib (OpenMVS header-only dependency) — pinned by commit
WORKDIR /opt
RUN git clone https://github.com/cdcseacave/VCG.git vcglib \
    && git -C vcglib checkout "${VCG_COMMIT}"

# OpenMVS (dense / mesh / texture) — built against system Eigen 3.4
RUN git clone --recursive --branch v2.4.0 --depth 1 --shallow-submodules \
        https://github.com/cdcseacave/openMVS.git
WORKDIR /opt/openMVS_build
RUN sed -i 's/COMPONENTS iostreams program_options system serialization/COMPONENTS iostreams program_options serialization/' \
        /opt/openMVS/CMakeLists.txt \
    && sed -i '/IMWRITE_JPEGXL_QUALITY/d' /opt/openMVS/libs/Common/Types.inl \
    && cmake /opt/openMVS \
        -DCMAKE_BUILD_TYPE=Release \
        -DVCG_ROOT=/opt/vcglib \
        -DOpenMVS_USE_CUDA=OFF \
        -DOpenMVS_USE_OPENMP=ON \
        -DOpenMVS_USE_PYTHON=OFF \

    && make -j"$(nproc)" \
    && make install \
    && ldconfig

# OpenMVG (SfM) — built last. Clone non-recursively and init only the submodules
# we actually compile: cereal (serialization, required) and osi_clp (linear
# solver used by the GLOBAL SfM engine). The glfw submodule is skipped — it's
# only for openMVG's GUI/examples, which we don't build (BUILD_EXAMPLES=OFF), so
# fetching it is both dead weight and a needless network failure point.
WORKDIR /opt
RUN git clone --branch v2.1 --depth 1 https://github.com/openMVG/openMVG.git \
    && git -C openMVG submodule update --init --depth 1 \
        src/dependencies/cereal src/dependencies/osi_clp
WORKDIR /opt/openMVG_build
RUN cmake -DCMAKE_BUILD_TYPE=RELEASE \
        -DOpenMVG_BUILD_TESTS=OFF \
        -DOpenMVG_BUILD_EXAMPLES=OFF \
        -DOpenMVG_BUILD_DOC=OFF \
        /opt/openMVG/src \
    && make -j"$(nproc)" \
    && make install \
    && ldconfig

# =========================================================================
#  Stage 3: runtime — engine runtime libs + binaries + Python app.
# =========================================================================
FROM ubuntu:26.04
ARG DEBIAN_FRONTEND=noninteractive

# Runtime-only shared libs the OpenMVG/OpenMVS binaries link against, plus
# Python for the web app. (The ldd gate below fails the build if any engine
# lib is still missing, so an under-specified package surfaces at build time.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libpng16-16 libjpeg-turbo8 libtiff6 \
        libgomp1 \
        libgmp10 libmpfr6 \
        libceres4t64 libcholmod5 libcxsparse4 libspqr4 libblas3 liblapack3 \
        libglu1-mesa \
        libxxf86vm1 libxi6 libxrandr2 \
        libboost-iostreams1.90.0 libboost-program-options1.90.0 \
        libboost-serialization1.90.0 libboost-filesystem1.90.0 \
        libopencv-core410 libopencv-imgproc410 libopencv-imgcodecs410 \
        libopencv-calib3d410 libopencv-features2d410 libopencv-flann410 \
        python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Carry over everything OpenMVG/OpenMVS installed (binaries + libs + share)
COPY --from=engine-builder /usr/local /usr/local
RUN ldconfig
ENV PATH="/usr/local/bin/OpenMVS:${PATH}"

# Verify every engine binary resolves its shared libs (fails loudly otherwise)
RUN set -e; \
    bins="$(ls /usr/local/bin/openMVG_main_* /usr/local/bin/OpenMVS/* 2>/dev/null)"; \
    for b in $bins; do \
        ldd "$b" 2>/dev/null | grep -q 'not found' \
            && { echo "MISSING LIBS for $b:"; ldd "$b" | grep 'not found'; exit 1; } || true; \
    done; \
    command -v openMVG_main_SfMInit_ImageListing >/dev/null \
    && command -v DensifyPointCloud >/dev/null \
    && echo "OK: OpenMVG + OpenMVS runtime self-contained"

# ---- Python app ----
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt
# Fail the build (not a job at runtime) if the pinned aruco API isn't present.
RUN python3 -c "import cv2.aruco as a; a.ArucoDetector; a.getPredefinedDictionary(a.DICT_APRILTAG_36h11)"

COPY app.py openmvg.py georef.py ./
COPY templates/ ./templates/
COPY --from=css-builder /build/output.css ./static/tailwind.css

ENV PYTHONUNBUFFERED=1 RELIEF3D_DATA=/data
EXPOSE 5002
CMD ["gunicorn", "--bind", "0.0.0.0:5002", "--workers", "1", "--timeout", "0", "app:app"]
