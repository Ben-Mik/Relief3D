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
# Engine pins: OpenMVG v2.1, OpenMVS v2.4.0, OpenCV 4.12.0, VCGLib 658ba36.

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
#  Stage 2: engine builder — compiles Eigen, OpenCV, CGAL, OpenMVG, OpenMVS.
#
#  Build order follows upstream's buildInDocker.sh (Eigen → OpenCV → CGAL →
#  OpenMVS), with OpenMVG added after. OpenCV is built from source at 4.12.0
#  rather than apt so OpenMVS gets the version it targets and we avoid
#  packaging quirks. Ubuntu 26 patches noted inline.
# =========================================================================
FROM ubuntu:26.04 AS engine-builder
ARG DEBIAN_FRONTEND=noninteractive
ARG VCG_COMMIT=658ba36d0a5666650da6e066b4794efc5a463407

# Build-time deps (same set upstream uses, plus Ceres/SuiteSparse for OpenMVG)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates pkg-config \
        libpng-dev libjpeg-dev libtiff-dev \
        libglu1-mesa-dev libglew-dev libglfw3-dev \
        libxxf86vm-dev libxi-dev libxrandr-dev \
        libboost-iostreams-dev libboost-program-options-dev \
        libboost-system-dev libboost-serialization-dev \
        libboost-filesystem-dev \
        libgmp-dev libmpfr-dev zlib1g-dev \
        libblas-dev liblapack-dev libsuitesparse-dev \
        libceres-dev \
        libnanoflann-dev libjxl-dev \
    && rm -rf /var/lib/apt/lists/*

# ---- Eigen 3.4 from source (upstream approach; header-only, no runtime lib) ----
WORKDIR /opt
RUN git clone --branch 3.4 --depth 1 https://gitlab.com/libeigen/eigen.git
WORKDIR /opt/eigen_build
RUN cmake ../eigen -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    && make install

# ---- OpenCV 4.12.0 from source ----
# 4.12 is the version OpenMVS 2.4 targets; it adds IMWRITE_JPEGXL_QUALITY (so
# we no longer need to patch that out of OpenMVS). Minimal build — only the
# modules OpenMVS links against; no GUI/video/extra deps.
WORKDIR /opt
RUN git clone --branch 4.12.0 --depth 1 https://github.com/opencv/opencv.git
WORKDIR /opt/opencv_build
RUN cmake ../opencv \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DBUILD_TESTS=OFF \
        -DBUILD_PERF_TESTS=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_opencv_apps=OFF \
        -DWITH_OPENEXR=OFF \
        -DWITH_FFMPEG=OFF \
        -DWITH_GTK=OFF \
        -DWITH_QT=OFF \
    && make -j"$(nproc)" \
    && make install \
    && ldconfig

# ---- CGAL v6.0.1 from source (upstream approach; mostly header-only) ----
WORKDIR /opt
RUN git clone --branch v6.0.1 --depth 1 https://github.com/CGAL/cgal.git
WORKDIR /opt/cgal_build
RUN cmake ../cgal -DCMAKE_BUILD_TYPE=Release && make install

# ---- VCGLib (OpenMVS header-only dependency) — pinned by commit ----
WORKDIR /opt
RUN git clone https://github.com/cdcseacave/VCG.git vcglib \
    && git -C vcglib checkout "${VCG_COMMIT}"

# ---- OpenMVS v2.4.0 ----
# Ubuntu 26 patch: Boost 1.90 makes boost_system header-only with no cmake
# config — remove it from the COMPONENTS list so find_package succeeds.
# (IMWRITE_JPEGXL_QUALITY patch no longer needed — OpenCV 4.12 has it.)
RUN git clone --recursive --branch v2.4.0 --depth 1 --shallow-submodules \
        https://github.com/cdcseacave/openMVS.git
WORKDIR /opt/openMVS_build
RUN sed -i 's/COMPONENTS iostreams program_options system serialization/COMPONENTS iostreams program_options serialization/' \
        /opt/openMVS/CMakeLists.txt \
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
# Clone non-recursively; init only cereal + osi_clp (glfw is GUI-only, skipped).
# CMAKE_POLICY_VERSION_MINIMUM: Ubuntu 26's cmake 4.2 rejects the ancient
# cmake_minimum_required in vendored osi_clp.
WORKDIR /opt
RUN git clone --branch v2.1 --depth 1 https://github.com/openMVG/openMVG.git \
    && git -C openMVG submodule update --init --depth 1 \
        src/dependencies/cereal src/dependencies/osi_clp
WORKDIR /opt/openMVG_build
RUN cmake -DCMAKE_BUILD_TYPE=RELEASE \
        -DOpenMVG_BUILD_TESTS=OFF \
        -DOpenMVG_BUILD_EXAMPLES=OFF \
        -DOpenMVG_BUILD_DOC=OFF \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        /opt/openMVG/src \
    && make -j"$(nproc)" \
    && make install \
    && ldconfig

# =========================================================================
#  Stage 3: runtime — engine runtime libs + binaries + Python app.
# =========================================================================
FROM ubuntu:26.04
ARG DEBIAN_FRONTEND=noninteractive

# Runtime shared libs the engine binaries link against + Python for the app.
# OpenCV comes from the engine-builder COPY below (built 4.12.0 in /usr/local),
# so no apt opencv packages here. The ldd gate fails the build if anything is
# missing, so under-specified packages surface at build time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libpng16-16 libjpeg-turbo8 libtiff6 \
        libjxl0.11 libwebp7 libwebpmux3 libwebpdemux2 \
        libgomp1 \
        libgmp10 libmpfr6 \
        libceres4t64 libcholmod5 libcxsparse4 libspqr4 libblas3 liblapack3 \
        libglu1-mesa \
        libxxf86vm1 libxi6 libxrandr2 \
        libboost-iostreams1.90.0 libboost-program-options1.90.0 \
        libboost-serialization1.90.0 libboost-filesystem1.90.0 \
        python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Carry over everything OpenMVG/OpenMVS/OpenCV installed (binaries + libs + share)
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
