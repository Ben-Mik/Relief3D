from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    redirect,
    url_for,
    session,
    send_from_directory,
    abort,
)
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix
import os
import uuid
import json
import threading
import traceback
import shutil
import time
import queue
import io
import zipfile
import requests
from datetime import datetime
import cv2
import openmvg  # OpenMVG→georef→OpenMVS engine
from dotenv import load_dotenv

# Load configuration from .env (if it exists) into os.environ.
# Existing env vars take precedence so you can still override per-shell.
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

app = Flask(__name__)
app.secret_key = os.urandom(24)
CORS(app)

# Honor headers set by Traefik when we're served behind a reverse proxy
# under a sub-path (e.g. /odm-link/). x_prefix=1 reads X-Forwarded-Prefix
# and sets SCRIPT_NAME so url_for() builds correct URLs.
app.wsgi_app = ProxyFix(
    app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1
)

# Configuration — default assumes the 3D-Annotator api is reachable on the
# shared `traefik` docker network. Override via env vars in .env if you want a
# different endpoint (e.g. running Relief3D natively against host services).
ANNOTATOR_BASE_URL = os.environ.get("ANNOTATOR_BASE_URL", "http://api:8000")
# All mutable state lives under RELIEF3D_DATA so a single volume persists it
# (default "." keeps native dev unchanged; compose sets it to /data).
DATA_DIR = os.environ.get("RELIEF3D_DATA", ".")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs")
JOBS_FILE = os.path.join(DATA_DIR, "jobs", "jobs.json")
JOBS_LOCK = threading.Lock()
PHOTO_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
# Retention window (hours) for both auto-sweeps: abandoned pending uploads and
# leftover meshes from failed/local-only runs. Tune per excavation via env.
SWEEP_HOURS = float(os.environ.get("RELIEF3D_SWEEP_HOURS", 8))

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(JOBS_FILE), exist_ok=True)


# ─── Job storage ────────────────────────────────────────────────────────────

def load_jobs():
    with JOBS_LOCK:
        if not os.path.exists(JOBS_FILE):
            return {}
        with open(JOBS_FILE) as f:
            return json.load(f)


def save_jobs(jobs):
    with JOBS_LOCK:
        with open(JOBS_FILE, "w") as f:
            json.dump(jobs, f, indent=2)


# Shared-offset presets ({name: {"offset": [x,y,z]}}) — co-register models in one frame.
OFFSET_PRESETS_FILE = os.path.join(DATA_DIR, "offset_presets.json")


def load_offset_presets():
    if not os.path.exists(OFFSET_PRESETS_FILE):
        return {}
    with open(OFFSET_PRESETS_FILE) as f:
        return json.load(f)


def save_offset_presets(presets):
    with open(OFFSET_PRESETS_FILE, "w") as f:
        json.dump(presets, f, indent=2)


def update_job(job_id, **updates):
    jobs = load_jobs()
    if job_id in jobs:
        jobs[job_id].update(updates)
        save_jobs(jobs)


def _drop_meshes(d):
    """Remove the heavy textured-mesh files, keeping report.txt. Used after a
       successful (re)upload — the model is then safe in the annotator."""
    for f in os.listdir(d):
        if f.lower().endswith((".obj", ".mtl", ".png", ".jpg", ".jpeg")):
            os.remove(os.path.join(d, f))


def _can_reupload(job):
    """True when a finished job has no annotator model yet but its mesh is still
       on disk (upload failed, or local-only run not swept yet) — so re-upload works."""
    if job.get("model_id") or job.get("status") != "done":
        return False
    d = job.get("output_dir")
    return bool(d) and os.path.isdir(d) and any(
        f.lower().endswith(".obj") for f in os.listdir(d))


def _can_retry(job):
    """True when a finished job's uploaded photos are still on disk — so the whole
       job can be re-run (with prior settings) until they hit the sweep window."""
    if job.get("status") not in ("done", "failed"):
        return False
    d = job.get("upload_dir")
    return bool(d) and os.path.isdir(d) and any(
        f.lower().endswith(PHOTO_EXTS) for f in os.listdir(d))


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return redirect(url_for("history"))


@app.route("/projects/<int:project_id>/new-job", methods=["GET", "POST"])
def new_job(project_id):
    # Capture token from URL on first visit
    token_from_url = request.args.get("token")
    if token_from_url:
        session["token"] = token_from_url

    if request.method == "POST":
        token = session.get("token") or request.form.get("token")
        # Photos were already uploaded on selection (POST /api/upload); use that job.
        job_id = request.form.get("job_id", "").strip()
        job_upload_dir = os.path.join(UPLOAD_DIR, job_id)
        if not job_id or not os.path.isdir(job_upload_dir):
            return render_template(
                "new_job.html",
                project_id=project_id,
                has_token=bool(token),
                error="No uploaded photos found — select photos first.",
            )
        files = [f for f in os.listdir(job_upload_dir)
                 if f.lower().endswith(PHOTO_EXTS)]
        if not files:
            return render_template(
                "new_job.html",
                project_id=project_id,
                has_token=bool(token),
                error="No photos in upload.",
            )

        # Compute the persistent output folder name up front so the rest of
        # the job writes straight into it (no <job_id>/ intermediate dir).
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        model_name = request.form.get("model_name") or (
            f"Relief3D_project{project_id}_{timestamp}"
        )
        # Filesystem-safe folder name (handles slashes, spaces, etc.)
        safe_folder = "".join(
            c if c.isalnum() or c in "-_." else "_" for c in model_name
        ).strip("._") or "model"
        # Avoid collision if a same-name job already ran in this minute:
        # append _2, _3, ... until we find a free folder.
        base = os.path.join(OUTPUT_DIR, f"{safe_folder}_{timestamp}")
        job_output_dir = base
        suffix = 2
        while os.path.exists(job_output_dir):
            job_output_dir = f"{base}_{suffix}"
            suffix += 1
        os.makedirs(job_output_dir)
        # Engine flags (per-flag dev controls). Cast numerics.
        options = {
            "feature_preset": request.form.get("feature_preset", "HIGH"),
            "sfm_engine": request.form.get("sfm_engine", "INCREMENTAL"),
            "resolution_level": int(request.form.get("resolution_level", 1)),
            "max_resolution": int(request.form.get("max_resolution", 2560)),
            "edge_length": float(request.form.get("edge_length") or 0),
            "texture_out_size": int(request.form.get("texture_out_size") or 0),
            "ransac_threshold": float(request.form.get("ransac_threshold") or 0.05),
        }

        # Optional GCP coords (parallel form arrays; keep valid rows) -> {id:(x,y,z)}.
        crs = (request.form.get("gcp_crs") or "").strip()
        gcp_ids = request.form.getlist("gcp_id")
        gcp_xs = request.form.getlist("gcp_x")
        gcp_ys = request.form.getlist("gcp_y")
        gcp_zs = request.form.getlist("gcp_z")
        gcp_coords = {}
        for i in range(len(gcp_ids)):
            zr = gcp_zs[i] if i < len(gcp_zs) else ""
            if not (gcp_ids[i].strip() and gcp_xs[i].strip() and gcp_ys[i].strip()):
                continue
            try:
                gcp_coords[int(gcp_ids[i])] = (
                    float(gcp_xs[i]), float(gcp_ys[i]),
                    float(zr) if zr.strip() else 0.0)
            except ValueError:
                continue

        # Reviewed marker observations from the GCP tool, if saved.
        observations = None
        obs_path = os.path.join(job_upload_dir, "gcp_observations.json")
        if os.path.exists(obs_path):
            raw = json.load(open(obs_path))
            observations = {fn: {int(k): tuple(v) for k, v in m.items()}
                            for fn, m in raw.items()}

        jobs = load_jobs()
        jobs[job_id] = {
            "status": "queued", "progress": 0, "step": "Queued",
            "project_id": project_id, "model_name": model_name,
            "photo_count": len(files), "options": options,
            "gcp_count": len(gcp_coords), "crs": crs if gcp_coords else None,
            # Stored so a retry can re-render the form with every prior setting.
            "gcp_coords": {str(k): list(v) for k, v in gcp_coords.items()},
            "offset_preset": request.form.get("offset_preset", "").strip(),
            "upload_dir": job_upload_dir, "output_dir": job_output_dir,
            "created_at": datetime.now().isoformat(),
        }
        save_jobs(jobs)

        # Shared-offset preset (or None = auto-centroid).
        preset_offset = None
        preset_name = request.form.get("offset_preset", "").strip()
        if preset_name:
            preset_offset = load_offset_presets().get(preset_name, {}).get("offset")

        JOB_QUEUE.put((job_id, job_upload_dir, job_output_dir, options, gcp_coords,
                       observations, preset_offset, token, model_name, project_id))
        return redirect(url_for("job_status", job_id=job_id))

    # Retry: re-render the form prefilled from a finished job whose photos persist.
    prefill = None
    retry_id = request.args.get("retry")
    if retry_id:
        job = load_jobs().get(retry_id)
        if job and _can_retry(job):
            prefill = {
                "job_id": retry_id, "model_name": job.get("model_name"),
                "photo_count": job.get("photo_count"), "options": job.get("options"),
                "crs": job.get("crs"), "gcp_coords": job.get("gcp_coords") or {},
                "offset_preset": job.get("offset_preset") or "",
            }
    return render_template(
        "new_job.html",
        project_id=project_id,
        has_token=bool(session.get("token")),
        offset_presets=list(load_offset_presets().keys()),
        prefill=prefill,
    )


@app.route("/jobs/<job_id>")
def job_status(job_id):
    jobs = load_jobs()
    job = jobs.get(job_id)
    if not job:
        return redirect(url_for("history"))
    job["can_reupload"] = _can_reupload(job)
    job["can_retry"] = _can_retry(job)
    return render_template("job.html", job=job, job_id=job_id)


@app.route("/jobs/<job_id>/reupload", methods=["POST"])
def reupload(job_id):
    """Re-push a finished job's kept mesh to the annotator (failed/local-only runs).
       Uses the current session token; clears the mesh on success like a fresh job."""
    job = load_jobs().get(job_id)
    token = session.get("token")
    if not job or not token or not _can_reupload(job):
        abort(400)
    try:
        model_id = upload_to_annotator(job["project_id"], job["model_name"],
                                       job["output_dir"], token)
        _drop_meshes(job["output_dir"])
        update_job(job_id, model_id=model_id, upload_error=None, step="Done")
    except Exception as e:
        update_job(job_id, upload_error=str(e))
    return redirect(url_for("job_status", job_id=job_id))


@app.route("/history")
def history():
    jobs = load_jobs()
    sorted_jobs = sorted(
        jobs.items(),
        key=lambda x: x[1].get("created_at", ""),
        reverse=True,
    )
    for _id, job in sorted_jobs:
        job["can_reupload"] = _can_reupload(job)
        job["can_retry"] = _can_retry(job)
    return render_template("history.html", jobs=sorted_jobs)


@app.route("/api/jobs/<job_id>")
def api_job(job_id):
    jobs = load_jobs()
    if job_id not in jobs:
        return jsonify({"error": "Not found"}), 404
    return jsonify(jobs[job_id])


# ---------------------------------------------------------------------------
# GCP placement tool (engine-independent). Detects coded markers per image and
# lets the user confirm / manually place missed ones. Produces observations
# {filename: {marker_id: [u, v]}} that any engine's file-handler consumes.
# ---------------------------------------------------------------------------
def detect_all_markers(photos_dir):
    """-> {filename: [{id, cx, cy}]} for every AprilTag 36h11 found (perspective-correct centre)."""
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
        cv2.aruco.DetectorParameters(),
    )
    out = {}
    for filename in sorted(os.listdir(photos_dir)):
        if not filename.lower().endswith(PHOTO_EXTS):
            continue
        img = cv2.imread(os.path.join(photos_dir, filename), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        corners, ids, _ = detector.detectMarkers(img)
        if ids is None:
            continue
        marks = []
        for c, i in zip(corners, ids.flatten()):
            cx, cy = c[0].mean(axis=0)
            marks.append({"id": int(i), "cx": round(float(cx), 1), "cy": round(float(cy), 1)})
        if marks:
            out[filename] = marks
    return out


def _job_upload_dir(job_id):
    d = os.path.join(UPLOAD_DIR, job_id)
    if not os.path.isdir(d):
        abort(404)
    return d


@app.route("/jobs/<job_id>/gcp")
def gcp_tool(job_id):
    _job_upload_dir(job_id)
    return render_template("gcp_tool.html", job_id=job_id)


@app.route("/uploads/<job_id>/<path:filename>")
def serve_upload(job_id, filename):
    return send_from_directory(_job_upload_dir(job_id), filename)


@app.route("/api/jobs/<job_id>/markers")
def api_markers(job_id):
    d = _job_upload_dir(job_id)
    images = sorted(f for f in os.listdir(d) if f.lower().endswith(PHOTO_EXTS))
    return jsonify({"images": images, "detected": detect_all_markers(d)})


@app.route("/api/jobs/<job_id>/observations", methods=["POST"])
def api_save_observations(job_id):
    d = _job_upload_dir(job_id)
    payload = request.get_json(force=True) or {}
    observations = payload.get("observations", {})
    with open(os.path.join(d, "gcp_observations.json"), "w") as f:
        json.dump(observations, f, indent=2)
    return jsonify({"ok": True, "images": len(observations)})


def sweep_uploads(max_age_h):
    """Delete upload dirs older than max_age_h that are abandoned (no started job) or
       finished (done/failed). Queued/processing jobs keep their photos; a finished job
       keeps them for the retry window, then they're reclaimed here."""
    jobs = load_jobs()
    cutoff = time.time() - max_age_h * 3600
    for name in os.listdir(UPLOAD_DIR):
        d = os.path.join(UPLOAD_DIR, name)
        if not os.path.isdir(d) or os.path.getmtime(d) >= cutoff:
            continue
        job = jobs.get(name)
        if job is None or job.get("status") in ("done", "failed"):
            shutil.rmtree(d, ignore_errors=True)


def sweep_stale_outputs(max_age_h=8):
    """Drop heavy mesh files from output dirs older than max_age_h, keeping report.txt.
       A successful upload already clears its own mesh inline; this catches the leftovers
       from failed or local-only (no-token) runs so they don't pile up indefinitely."""
    cutoff = time.time() - max_age_h * 3600
    for name in os.listdir(OUTPUT_DIR):
        d = os.path.join(OUTPUT_DIR, name)
        if not os.path.isdir(d) or os.path.getmtime(d) >= cutoff:
            continue
        for f in os.listdir(d):
            if f.lower().endswith((".obj", ".mtl", ".png", ".jpg", ".jpeg")):
                os.remove(os.path.join(d, f))


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """Upload-on-select: save photos to a fresh pending job, detect markers, return summary."""
    sweep_uploads(SWEEP_HOURS)
    sweep_stale_outputs(SWEEP_HOURS)
    files = request.files.getlist("photos")
    if not files:
        return jsonify({"error": "no photos"}), 400
    job_id = str(uuid.uuid4())[:8]
    d = os.path.join(UPLOAD_DIR, job_id)
    os.makedirs(d)
    for f in files:
        f.save(os.path.join(d, os.path.basename(f.filename)))
    detected = detect_all_markers(d)
    images = sorted(x for x in os.listdir(d) if x.lower().endswith(PHOTO_EXTS))
    n_markers = len({m["id"] for ms in detected.values() for m in ms})
    return jsonify({
        "job_id": job_id,
        "n_images": len(images),
        "n_with_markers": len(detected),
        "n_markers": n_markers,
    })


@app.route("/api/offset-presets", methods=["POST"])
def api_add_offset_preset():
    d = request.get_json(force=True) or {}
    name = (d.get("name") or "").strip()
    offset = d.get("offset")
    if not name or not (isinstance(offset, list) and len(offset) == 3):
        return jsonify({"error": "name + [x,y,z] required"}), 400
    presets = load_offset_presets()
    presets[name] = {"offset": [float(v) for v in offset]}
    save_offset_presets(presets)
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Annotator upload: zip the textured mesh and push it to 3D-Annotator.
# Contract (verified against ~/3D-Annotator): the three.js loader REQUIRES the
# MTL and only accepts jpg/jpeg/png textures — so we pack obj + mtl + texture.
# The model sits in the LOCAL frame; the annotator's ModelData has no geo field,
# so report.txt (settings + georef + offset) rides along in the zip as the human-
# readable record of how to map vertices back to real-world coords
# (real-world = local + offset). The loader ignores the unmatched .txt.
# ---------------------------------------------------------------------------
def _write_report(output_dir, job_id, options, report):
    """Human-readable report.txt shipped with the model: how it was built + how to
       place it back in the world. Mesh stats come from the produced OBJ."""
    job = load_jobs().get(job_id, {})
    o = options
    obj = next((f for f in os.listdir(output_dir) if f.lower().endswith(".obj")), None)
    nv = nf = 0
    if obj:
        with open(os.path.join(output_dir, obj)) as fh:
            for line in fh:
                if line[:2] == "v ":
                    nv += 1
                elif line[:2] == "f ":
                    nf += 1
    edge = float(o["edge_length"])
    L = [
        "Relief3D processing report",
        "==========================",
        f"Model:   {job.get('model_name', '')}",
        f"Project: #{job.get('project_id', '')}",
        f"Created: {job.get('created_at', '')}",
        f"Photos:  {job.get('photo_count', '')}",
        f"Mesh:    {nv:,} vertices · {nf:,} faces",
        "",
        "Settings",
        "--------",
        f"Features (-p):            {o['feature_preset']}",
        f"SfM engine:               {o['sfm_engine']}",
        f"Densify resolution-level: {o['resolution_level']}",
        f"Densify max-resolution:   {o['max_resolution']} px",
        f"Mesh edge-length:         {edge} m" + ("" if edge else " (off)"),
        f"Texture output size:      {o['texture_out_size'] or 'unchanged'}",
        f"Georef RANSAC threshold:  {o['ransac_threshold']} m",
        "",
        "Georeferencing",
        "--------------",
    ]
    if report.get("georeferenced"):
        x, y, z = report["offset"]
        L += [
            "Status: georeferenced",
            f"CRS:    {report.get('crs') or '(unspecified)'}",
            f"Offset: [{x:.4f}, {y:.4f}, {z:.4f}]",
            "        real-world = local + offset",
            f"Scale:  {report['scale']:.6f}",
            f"Markers used:    {report['inliers']}",
            f"Markers dropped: {report['outliers'] or 'none'}",
            f"RMS:    {report['rms_mm']:.1f} mm",
        ]
    else:
        L += [
            "Status: NOT georeferenced",
            f"Reason: {report.get('reason', 'no GCPs provided')}",
            "Model is in a local frame (arbitrary scale & orientation).",
        ]
    with open(os.path.join(output_dir, "report.txt"), "w") as f:
        f.write("\n".join(L) + "\n")


def _zip_textured_mesh(mesh_dir):
    """In-memory baseFile.zip: obj + mtl + textures, plus report.txt sidecar."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(os.listdir(mesh_dir)):
            if f.lower().endswith((".obj", ".mtl", ".png", ".jpg", ".jpeg", ".txt")):
                z.write(os.path.join(mesh_dir, f), arcname=f)
    buf.seek(0)
    return buf


def upload_to_annotator(project_id, model_name, mesh_dir, token):
    """Create a modelData entry, then PUT the textured-mesh zip. Returns model id."""
    auth = {"Authorization": f"Token {token}"}
    r = requests.post(
        f"{ANNOTATOR_BASE_URL}/api/v1/modelData/",
        json={"project_id": project_id, "name": model_name,
              "modelType": "texture_mesh", "annotationType": "index"},
        headers=auth)
    if not r.ok:
        raise RuntimeError(f"modelData create failed [{r.status_code}]: {r.text}")
    model_id = r.json().get("modelData_id")
    if not model_id:
        raise RuntimeError(f"modelData create returned no id: {r.text}")
    up = requests.put(
        f"{ANNOTATOR_BASE_URL}/api/v1/modelData/{model_id}/baseFile",
        files={"file": ("baseFile.zip", _zip_textured_mesh(mesh_dir), "application/zip")},
        data={"fileFormat": "application/zip"}, headers=auth)
    if not up.ok:
        raise RuntimeError(f"baseFile upload failed [{up.status_code}]: {up.text}")
    return model_id


# ---------------------------------------------------------------------------
# Job pipeline: single-concurrency queue → OpenMVG → georef → OpenMVS → package.
# ---------------------------------------------------------------------------
def process_relief_job(job_id, upload_dir, output_dir, options, gcp_coords,
                       observations, preset_offset, token, model_name, project_id):
    def progress(step):
        update_job(job_id, status="processing", step=step)

    work_dir = os.path.join(output_dir, "work")
    try:
        update_job(job_id, status="processing", step="Preparing")
        images_dir = os.path.join(work_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        for f in os.listdir(upload_dir):
            if f.lower().endswith(PHOTO_EXTS):
                shutil.copy(os.path.join(upload_dir, f), images_dir)

        result = openmvg.reconstruct(work_dir, options, gcp_coords or None,
                                     observations=observations,
                                     preset_offset=preset_offset, progress=progress)
        report = result["georef"]

        # Collect the textured mesh (obj + mtl + texture image) into output_dir.
        # Key on extension, not name, so the texture is caught whatever OpenMVS calls it.
        # (mvs top-level images = the texture(s); depth maps are .dmap, undist images are in a subdir.)
        mvs = os.path.join(work_dir, "mvs")
        produced = [f for f in os.listdir(mvs)
                    if f.lower().endswith((".obj", ".mtl", ".png", ".jpg", ".jpeg"))]
        for f in produced:
            shutil.copy(os.path.join(mvs, f), output_dir)
        # Tag the report with the CRS (held on the job record) so it fully
        # describes the real-world frame. The structured report is kept on the
        # job record (below); report.txt is the human-readable copy shipped in the zip.
        report["crs"] = load_jobs().get(job_id, {}).get("crs")
        _write_report(output_dir, job_id, options, report)

        # Upload to the annotator. Reconstruction is already done, so an upload
        # failure does NOT fail the job — it stays "done" with no model_id and an
        # upload_error; the mesh is kept and the UI offers a re-upload. (No token =
        # local-only run, also re-uploadable later from a tokened session.)
        model_id, upload_error = None, None
        if token:
            update_job(job_id, status="processing", step="Uploading to annotator")
            try:
                model_id = upload_to_annotator(project_id, model_name, output_dir, token)
                _drop_meshes(output_dir)  # safe in annotator now; keep report.txt
            except Exception as e:
                upload_error = str(e)

        update_job(job_id, status="done",
                   step="Done" if (model_id or not token) else "Upload failed",
                   georeferenced=bool(report.get("georeferenced")),
                   georef=report, outputs=produced,
                   model_id=model_id, upload_error=upload_error)
        shutil.rmtree(work_dir, ignore_errors=True)  # success: keep output_dir, drop intermediates
    except Exception as e:
        traceback.print_exc()
        update_job(job_id, status="failed", step="Failed", error=str(e))
        # work_dir is intentionally kept on failure for inspection; the output
        # sweep reclaims it later like any other stale output.


JOB_QUEUE = queue.Queue()


def _job_worker():
    while True:
        args = JOB_QUEUE.get()
        try:
            process_relief_job(*args)
        finally:
            JOB_QUEUE.task_done()


threading.Thread(target=_job_worker, daemon=True).start()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=False)
