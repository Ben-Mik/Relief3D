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
import time
import georef
from PIL import Image

SENSOR_DB = "/usr/local/lib/openMVG/sensor_width_camera_database.txt"


def _run(work_dir, shell, progress=None, stage="", log_path=None):
    """Run a chained shell pipeline (set -e) with work_dir as the working directory.
       stderr is merged into stdout so OpenMVS errors (which it logs to stdout) are
       captured in chronological order for a meaningful failure message. When
       log_path is given, the exact command + full engine output for this stage are
       appended there — kept even on failure — so post-mortem diagnosis has the
       engine's own words (coverage, faces-not-covered, masked views, seam-leveling)
       instead of guessing from the rendered result."""
    if progress and stage:
        progress(stage)
    t0 = time.monotonic()
    r = subprocess.run(["bash", "-c", shell], cwd=work_dir,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    elapsed = time.monotonic() - t0
    if log_path:
        with open(log_path, "a") as f:
            f.write(f"\n{'='*72}\n# stage: {stage}  ({elapsed:.1f}s)\n{'='*72}\n$ {shell}\n\n{r.stdout}\n")
    if r.returncode != 0:
        raise RuntimeError(f"stage '{stage}' failed:\n{r.stdout[-3000:]}")
    return r.stdout


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
        if not name.lower().endswith(georef.PHOTO_EXTS):
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
                preset_offset=None, progress=None, log_path=None):
    """Full pipeline. Photos must already be at  work_dir/images.
       Returns {"mesh_path": <obj>, "georef": <report dict>}.
       options: feature_preset, sfm_engine, max_image_pct, resolution_level,
                max_resolution, edge_length, decimate, roi_border, close_holes,
                texture_out_size, ransac_threshold.
       log_path: if given, the full per-stage engine output is written there
                 (kept on the hub for post-mortem; not shipped to the annotator)."""
    o = options
    M, R = "ovg/matches", "ovg/recon"
    t_start = time.monotonic()

    if log_path:
        with open(log_path, "w") as f:
            f.write("Relief3D engine log\n===================\n")
            f.write(f"options: {options}\n")

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
    _run(work_dir, sfm, progress, "SfM", log_path)
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
    if log_path:
        with open(log_path, "a") as f:
            f.write(f"\n# georef: {report}\n")

    # ---- OpenMVS dense / mesh / texture ----
    # Detail levers on ReconstructMesh (quadric edge-collapse simplification).
    # edge-length is metric when georeferenced
    # (real-world triangle size); decimate is a scale-independent ratio (0..1].
    # Keeping the mesh small here also keeps TextureMesh fast.
    edge = f" --edge-length {o['edge_length']}" if float(o.get("edge_length") or 0) > 0 else ""
    decimate = f" --decimate {o['decimate']}" if 0 < float(o.get("decimate") or 0) < 1 else ""
    # Debug toggle: RELIEF3D_TEX_THREADS=1 forces single-threaded texturing — tests
    # whether the silent texture corruption is an OpenMP/threading race in this build.
    tex_mt = (f" --max-threads {os.environ['RELIEF3D_TEX_THREADS']}"
              if os.environ.get("RELIEF3D_TEX_THREADS") else "")
    # Hole filling (TextureMesh --close-holes): max hole size, as a boundary-edge
    # count, that gets auto-filled. 0 = disabled, OpenMVS default = 30; higher
    # fills bigger gaps but risks bridging real voids (e.g. a genuine pit edge).
    # Raw user value; unset -> omit the flag so OpenMVS's own default (30) applies.
    close_holes = (f" --close-holes {int(o['close_holes'])}"
                   if str(o.get("close_holes") or "").strip() != "" else "")
    # Uncovered faces get a dark neutral grey instead of OpenMVS's default
    # alarm-orange. Dark (not light) because uncovered faces are usually in
    # recessed/shadowed pockets of complex geometry, where a dark fill blends in
    # as natural shadow rather than glowing as "missing data".
    empty_color = 4210752  # 0x404040 (64,64,64) dark neutral grey
    # OpenMVS 2.4 estimates a region-of-interest from a robust *core* of points and
    # crops the mesh to it — and it's ON by default (--estimate-roi 1.1,
    # --crop-to-roi true), so a normal run silently trims the result, and on sparse
    # scenes it can trim everything. We drive it explicitly from one value
    # (`roi_border`, the UI "Auto-boundaries"):
    #   --estimate-roi is a MULTIPLIER on the core box (EnlargePercent: m_ext *= x),
    #   so 1.0 = tight to the subject, 1.2 = +20%, and <1 SHRINKS into the subject.
    # Hence only >=1 enables cropping (passed straight through); anything else turns
    # ROI fully OFF so the whole model is reconstructed.
    roi_scale = float(o.get("roi_border") or 0)
    if roi_scale >= 1:
        densify_roi, recon_roi = f" --estimate-roi {roi_scale}", ""
    else:
        densify_roi = " --estimate-roi 0 --crop-to-roi false"
        recon_roi = " --crop-to-roi false"
    mvs = (
        f"set -e;"
        f"mkdir -p mvs;"
        f"openMVG_main_openMVG2openMVS -i {R}/sfm_data.json -o mvs/scene.mvs -d mvs/undist;"
        f"cd mvs;"
        f"DensifyPointCloud scene.mvs --resolution-level {o['resolution_level']} --max-resolution {o['max_resolution']}{densify_roi};"
        f"ReconstructMesh scene_dense.mvs{edge}{decimate}{recon_roi};"
        # --export-type obj: OBJ + MTL + JPG, the portable textured-mesh format for
        # external DCC tools (Blender/MeshLab auto-texture via the MTL). The 2.3.0
        # OBJ-writer segfault that forced PLY before is fixed in 2.4.0. The
        # annotator loads OBJ + the image and ignores the MTL; external tools use
        # it. (OBJ is ASCII so far larger than binary PLY — lean on --decimate for
        # interactive meshes.)
        # --max-texture-size 0 = unbounded -> ONE atlas sized to fit all patches,
        # instead of the default 8192 cap that spills into a second (mostly empty)
        # page. The texture_out_size resize below then scales that single atlas.
        f"TextureMesh scene_dense.mvs --mesh-file scene_dense_mesh.ply{tex_mt}{close_holes}"
        f" --max-texture-size 0 --empty-color {empty_color}"
        f" --export-type obj -o scene_dense_mesh_texture.mvs"
    )
    _run(work_dir, mvs, progress, "Dense / mesh / texture", log_path)

    # Post-process: resize texture pages to the user-requested size. OpenMVS
    # auto-sizes the atlas during texturing (so UV quality is never capped at
    # processing time); this is a final, absolute resize. It scales up as well
    # as down — an upscaled atlas can help the annotator's pixel-based picking.
    # Aspect-preserving (longest edge = tex_size) so UVs never distort.
    tex_size = int(o.get("texture_out_size") or 0)
    if tex_size > 0:
        if progress:
            progress("Resizing texture")
        t_resize = time.monotonic()
        # OBJ export writes the atlas as JPG (..._map_Kd.jpg); PLY would write PNG.
        # Match either, but never the .obj/.mtl themselves.
        texes = (glob.glob(os.path.join(work_dir, "mvs", "scene_dense_mesh_texture*.jpg"))
                 + glob.glob(os.path.join(work_dir, "mvs", "scene_dense_mesh_texture*.png")))
        for tex in texes:
            img = Image.open(tex)
            scale = tex_size / max(img.width, img.height)
            if scale != 1:
                new = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
                # Preserve the source format; keep JPG high-quality to avoid
                # compounding lossy recompression on the atlas.
                save_kw = {"quality": 95} if tex.lower().endswith((".jpg", ".jpeg")) else {}
                img.resize(new, Image.LANCZOS).save(tex, **save_kw)
                if log_path:
                    with open(log_path, "a") as f:
                        f.write(f"\n# resized {os.path.basename(tex)} "
                                f"{img.width}x{img.height} -> {new[0]}x{new[1]}\n")
        if log_path:
            with open(log_path, "a") as f:
                f.write(f"# texture resize: {time.monotonic() - t_resize:.1f}s\n")

    if log_path:
        with open(log_path, "a") as f:
            f.write(f"\n# TOTAL: {time.monotonic() - t_start:.1f}s\n")

    return {
        "mesh_path": os.path.join(work_dir, "mvs", "scene_dense_mesh_texture.obj"),
        "georef": report,
    }
