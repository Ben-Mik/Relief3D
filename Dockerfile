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
# Engine pins: OpenMVG v2.1, OpenMVS v2.4.0, CGAL v6.0.1, VCGLib 658ba36.
#
# Base = Ubuntu 24.04 to match upstream OpenMVS's tested toolchain (GCC 13,
# apt OpenCV 4.6, source Eigen/CGAL). The engine build is SINGLE-STAGE — same
# pattern as upstream's docker/Dockerfile + buildInDocker.sh: build deps stay
# in the image and serve as the runtime libs, so there is no separate runtime
# stage to keep package-pinned in sync (that split is what caused the U26
# t64/410/jxl runtime-pin churn). CSS is the only separate stage (needs Node).

# =========================================================================
#  Stage 1: Tailwind/DaisyUI CSS bundle (Node — kept out of the final image)
# =========================================================================
FROM node:20-alpine AS css-builder
WORKDIR /build
COPY tailwind.config.js input.css ./
COPY templates/ ./templates/
RUN npm init -y >/dev/null && \
    npm install --no-audit --no-fund tailwindcss@3 daisyui@4 && \
    npx tailwindcss -i ./input.css -o ./output.css --minify

# =========================================================================
#  Stage 2: engine + app — Ubuntu 24.04 / GCC 13, single stage.
# =========================================================================
FROM ubuntu:24.04
ARG DEBIAN_FRONTEND=noninteractive
ARG VCG_COMMIT=658ba36d0a5666650da6e066b4794efc5a463407

# Build + runtime deps in one go (single stage = build libs ARE runtime libs).
# OpenCV is apt 4.6 — the version upstream builds against on 24.04; texture
# corruption was already ruled out as version-related, so 4.6 is the simplest
# match. OpenMVG keeps its bundled osi_clp submodule (U24's cmake 3.28 accepts
# its old cmake_minimum, so no policy patch needed). graphviz provides neato
# for OpenMVG's match-graph step (silences the 'neato: not found' warning).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates pkg-config \
        libpng-dev libjpeg-dev libtiff-dev \
        libglu1-mesa-dev libglew-dev libglfw3-dev \
        libxxf86vm-dev libxi-dev libxrandr-dev \
        libeigen3-dev \
        libboost-all-dev \
        libnanoflann-dev \
        libopencv-dev \
        libgmp-dev libmpfr-dev zlib1g-dev \
        libblas-dev liblapack-dev libsuitesparse-dev \
        libceres-dev \
        graphviz \
        python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# ---- CGAL v6.0.1 from source (upstream approach; header-only, fast install) ----
WORKDIR /opt
RUN git clone --branch v6.0.1 --depth 1 https://github.com/CGAL/cgal.git
WORKDIR /opt/cgal_build
RUN cmake ../cgal -DCMAKE_BUILD_TYPE=Release && make install

# ---- VCGLib (OpenMVS header-only dependency) — pinned by commit ----
WORKDIR /opt
RUN git clone https://github.com/cdcseacave/VCG.git vcglib \
    && git -C vcglib checkout "${VCG_COMMIT}"

# ---- OpenMVS v2.4.0 ----
# Single patch on U24: apt OpenCV 4.6 lacks IMWRITE_JPEGXL_QUALITY (added in
# 4.12); delete that line — harmless since we output PNG, not JXL. (No Boost
# 'system' patch needed here: U24's Boost 1.83 still provides the component.)
RUN git clone --recursive --branch v2.4.0 --depth 1 --shallow-submodules \
        https://github.com/cdcseacave/openMVS.git
WORKDIR /opt/openMVS_build
RUN sed -i '/IMWRITE_JPEGXL_QUALITY/d' /opt/openMVS/libs/Common/Types.inl \
    && cmake /opt/openMVS \
        -DCMAKE_BUILD_TYPE=Release \
        -DVCG_ROOT=/opt/vcglib \
        -DOpenMVS_USE_CUDA=OFF \
        -DOpenMVS_USE_OPENMP=ON \
        -DOpenMVS_USE_PYTHON=OFF \
    && make -j"$(nproc)" \
    && make install \
    && ldconfig

# ---- OpenMVG v2.1 (SfM) ----
# Clone non-recursively; init only the submodules we compile: cereal
# (serialization, required) and osi_clp (linear solver for GLOBAL SfM). glfw is
# GUI-only and skipped (BUILD_EXAMPLES=OFF).
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
ENV PATH="/usr/local/bin/OpenMVS:${PATH}"

# Sanity check: the key binaries resolve and run (single stage, so the libs
# they linked against at build time are present by construction).
RUN command -v openMVG_main_SfMInit_ImageListing >/dev/null \
    && command -v DensifyPointCloud >/dev/null \
    && command -v TextureMesh >/dev/null \
    && echo "OK: OpenMVG + OpenMVS present"

# Drop the heavy source trees (binaries + libs are installed to /usr/local).
RUN rm -rf /opt/cgal /opt/cgal_build /opt/openMVS /opt/openMVS_build \
           /opt/openMVG /opt/openMVG_build /opt/vcglib

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
