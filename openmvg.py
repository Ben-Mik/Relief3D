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


_IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def _downscale_images(work_dir, pct, observations):
    """Downscale the pipeline's working images to `pct`% of their dimensions,
       preserving EXIF. OpenMVG reads focal length + camera model from EXIF and
       re-derives the pixel focal from the (new) image width, so the camera
       intrinsics stay correct after resizing. Marker observations are in
       original-pixel space, so scale them by the same factor.
       Returns the (possibly scaled) observations; input is left untouched."""
    if not pct or pct >= 100:
        return observations
    factor = pct / 100.0
    img_dir = os.path.join(work_dir, "images")
    for name in os.listdir(img_dir):
        if not name.lower().endswith(_IMG_EXTS):
            continue
        path = os.path.join(img_dir, name)
        im = Image.open(path)
        exif = im.info.get("exif")
        new = (max(1, round(im.width * factor)), max(1, round(im.height * factor)))
        im = im.resize(new, Image.LANCZOS)
        kw = {"quality": 95}
        if exif:
            kw["exif"] = exif
        im.save(path, **kw)
    if not observations:
        return observations
    return {fn: {mid: [xy[0] * factor, xy[1] * factor] for mid, xy in marks.items()}
            for fn, marks in observations.items()}


def reconstruct(work_dir, options, gcp_coords=None, observations=None,
                preset_offset=None, progress=None):
    """Full pipeline. Photos must already be at  work_dir/images.
       Returns {"mesh_path": <obj>, "georef": <report dict>}.
       options: feature_preset, sfm_engine, max_image_pct, resolution_level,
                max_resolution, edge_length, decimate, texture_out_size,
                ransac_threshold."""
    o = options
    M, R = "ovg/matches", "ovg/recon"

    # Downscale the working images first (markers were detected full-res at
    # upload). This is the main SfM speed lever and sets the texture-source
    # ceiling; OpenMVS then auto-sizes the texture atlas to match.
    if progress:
        progress("Preparing images")
    observations = _downscale_images(
        work_dir, int(o.get("max_image_pct") or 0), observations)

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
    # Detail levers on ReconstructMesh (both built into OpenMVS v2.3.0, quadric
    # edge-collapse simplification). edge-length is metric when georeferenced
    # (real-world triangle size); decimate is a scale-independent ratio (0..1].
    # Keeping the mesh small here also keeps TextureMesh fast.
    edge = f" --edge-length {o['edge_length']}" if float(o.get("edge_length") or 0) > 0 else ""
    decimate = f" --decimate {o['decimate']}" if 0 < float(o.get("decimate") or 0) < 1 else ""
    roi = f" --roi-border {o['roi_border']}" if float(o.get("roi_border") or 0) > 0 else ""
    mvs = (
        f"set -e;"
        f"mkdir -p mvs;"
        f"openMVG_main_openMVG2openMVS -i {R}/sfm_data.json -o mvs/scene.mvs -d mvs/undist;"
        f"cd mvs;"
        f"DensifyPointCloud scene.mvs --resolution-level {o['resolution_level']} --max-resolution {o['max_resolution']}{roi};"
        f"ReconstructMesh scene_dense.mvs{edge}{decimate}{roi};"
        # Export PLY (the default): OpenMVS v2.3.0's OBJ writer segfaults during
        # export on this build, while PLY is reliable. PLY carries UVs + a
        # sidecar texture PNG; the 3D-Annotator three.js loader takes PLY + PNG.
        f"TextureMesh scene_dense.mvs --mesh-file scene_dense_mesh.ply -o scene_dense_mesh_texture.mvs"
    )
    _run(work_dir, mvs, progress, "Dense / mesh / texture")

    # Post-process: resize texture pages to the user-requested size. OpenMVS
    # auto-sizes the atlas during texturing (so UV quality is never capped at
    # processing time); this is a final, absolute resize. It scales up as well
    # as down — an upscaled atlas can help the annotator's pixel-based picking.
    # Aspect-preserving (longest edge = tex_size) so UVs never distort.
    tex_size = int(o.get("texture_out_size") or 0)
    if tex_size > 0:
        if progress:
            progress("Resizing texture")
        for tex in glob.glob(os.path.join(work_dir, "mvs", "scene_dense_mesh_texture*.png")):
            img = Image.open(tex)
            scale = tex_size / max(img.width, img.height)
            if scale != 1:
                new = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
                img.resize(new, Image.LANCZOS).save(tex)

    return {
        "mesh_path": os.path.join(work_dir, "mvs", "scene_dense_mesh_texture.ply"),
        "georef": report,
    }
