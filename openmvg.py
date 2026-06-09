"""
Reconstruction engine: OpenMVG (SfM) -> our georef -> OpenMVS (dense/mesh/texture).

The OpenMVG/OpenMVS binaries are installed in this image (see Dockerfile), so we
call them directly as subprocesses — no sibling container. Approach B: georeference
the sfm_data *between* SfM and dense, so dense/mesh run in the local metric frame
(metric --edge-length means real cm). Georef is BEST-EFFORT — the reconstruction
always completes; a report says what happened (markers used/dropped, RMS, or the
failure reason).

Requires x86 with AVX (OpenMVS TextureMesh); won't run on Apple Silicon.
"""
import glob
import os
import subprocess
import georef
from PIL import Image

SENSOR_DB = "/usr/local/lib/openMVG/sensor_width_camera_database.txt"


def _run(work_dir, shell, progress=None, stage=""):
    """Run a chained shell pipeline (set -e) with work_dir as the working directory.
       stderr is merged into stdout so OpenMVS errors (which it logs to stdout) are
       captured in chronological order for a meaningful failure message."""
    if progress and stage:
        progress(stage)
    r = subprocess.run(["bash", "-c", shell], cwd=work_dir,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"stage '{stage}' failed:\n{r.stdout[-3000:]}")
    return r.stdout


def reconstruct(work_dir, options, gcp_coords=None, observations=None,
                preset_offset=None, progress=None):
    """Full pipeline. Photos must already be at  work_dir/images.
       Returns {"mesh_path": <obj>, "georef": <report dict>}.
       options: feature_preset, sfm_engine, resolution_level, max_resolution,
                edge_length, texture_size, ransac_threshold."""
    o = options
    M, R = "ovg/matches", "ovg/recon"

    # ---- OpenMVG SfM (-> sfm_data.json incl. structure, for georef + transform) ----
    sfm = (
        f"set -e;"
        f"rm -rf ovg mvs; mkdir -p {M} {R};"
        f"openMVG_main_SfMInit_ImageListing -i images -o {M} -d {SENSOR_DB} -c 3;"
        f"openMVG_main_ComputeFeatures -i {M}/sfm_data.json -o {M} -m SIFT -p {o['feature_preset']};"
        f"openMVG_main_PairGenerator -i {M}/sfm_data.json -o {M}/pairs.bin;"
        f"openMVG_main_ComputeMatches -i {M}/sfm_data.json -p {M}/pairs.bin -o {M}/matches.putative.bin;"
        f"openMVG_main_GeometricFilter -i {M}/sfm_data.json -m {M}/matches.putative.bin -o {M}/matches.f.bin;"
        f"openMVG_main_SfM --sfm_engine {o['sfm_engine']} --input_file {M}/sfm_data.json --match_dir {M} --output_dir {R};"
        f"openMVG_main_ConvertSfM_DataFormat -i {R}/sfm_data.bin -o {R}/sfm_data.json -V -I -E -S"
    )
    _run(work_dir, sfm, progress, "SfM")
    sfm_json = os.path.join(work_dir, "ovg", "recon", "sfm_data.json")

    # ---- georef (best-effort): compute on the poses, apply to sfm_data before dense ----
    report = {"georeferenced": False, "reason": "no GCPs provided"}
    if gcp_coords:
        if progress:
            progress("Georeferencing")
        report = georef.georeference(
            os.path.join(work_dir, "images"), sfm_json, gcp_coords,
            preset_offset=preset_offset, threshold=float(o["ransac_threshold"]),
            observations=observations)
        if report.get("georeferenced"):
            georef.apply_to_sfm_data(sfm_json, report["scale"],
                                     report["rotation"], report["translation"])

    # ---- OpenMVS dense / mesh / texture ----
    edge = f" --edge-length {o['edge_length']}" if float(o.get("edge_length") or 0) > 0 else ""
    simplify = ""
    mvs = (
        f"set -e;"
        f"mkdir -p mvs;"
        f"openMVG_main_openMVG2openMVS -i {R}/sfm_data.json -o mvs/scene.mvs -d mvs/undist;"
        f"cd mvs;"
        f"DensifyPointCloud scene.mvs --resolution-level {o['resolution_level']} --max-resolution {o['max_resolution']};"
        f"ReconstructMesh scene_dense.mvs{edge};"
        f"{simplify}"
        f"TextureMesh scene_dense.mvs --mesh-file scene_dense_mesh.ply --export-type obj"
    )
    _run(work_dir, mvs, progress, "Dense / mesh / texture")

    # Post-process: downsample texture pages to user-requested size.
    # TextureMesh picks its own atlas size based on input images; we resize
    # after the fact so UV quality is never capped during processing.
    tex_size = int(o.get("texture_out_size") or 0)
    if tex_size > 0:
        if progress:
            progress("Resizing texture")
        for tex in glob.glob(os.path.join(work_dir, "mvs", "scene_dense_texture*.png")):
            img = Image.open(tex)
            if img.width > tex_size or img.height > tex_size:
                img = img.resize((tex_size, tex_size), Image.LANCZOS)
                img.save(tex)

    return {
        "mesh_path": os.path.join(work_dir, "mvs", "scene_dense_texture.obj"),
        "georef": report,
    }
