"""
Decoupled georeferencing — engine-agnostic.

Needs only: camera poses (from any SfM), the photos, and GCP survey coords.
  aruco detect -> triangulate markers in SfM frame
  -> RANSAC Umeyama (7-DOF, auto-drops outliers, flags them)
  -> apply the similarity transform to the SfM poses + structure (before dense/mesh).

Output is in a LOCAL frame (a shared offset subtracted) so multiple models using the
same offset preset line up together (e.g. in Blender). Real-world = local + offset.

Locked design decisions:
  - RANSAC rejection: a bad marker (wrong survey coord / weak triangulation) is dropped
    automatically AND reported as an outlier, so the user knows before trusting the result.
  - Offset presets: pass a fixed offset to co-register multiple models in one frame.
  - Residuals are ours (from this fit) — no engine residual logic.
"""
import os, glob, json, itertools
import numpy as np
import cv2

APRILTAG_DICT = cv2.aruco.DICT_APRILTAG_36h11

# Photo file types the pipeline accepts. Defined here (the lowest-level module)
# and imported by app.py + openmvg.py so there's a single source of truth.
PHOTO_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


# ----------------------------- marker detection -----------------------------
def _quad_center(q):
    """Projected centre of a square = intersection of its diagonals (perspective-correct)."""
    p1, p2, p3, p4 = q
    x1, y1 = p1; x2, y2 = p3; x3, y3 = p2; x4, y4 = p4
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(d) < 1e-9:
        return q.mean(axis=0)
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d
    return np.array([px, py])


def detect_observations(images_dir, dict_id=APRILTAG_DICT):
    """-> {filename: {marker_id: (u, v)}}  perspective-correct centres."""
    det = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(dict_id),
        cv2.aruco.DetectorParameters())
    obs = {}
    for p in sorted(f for f in glob.glob(os.path.join(images_dir, "*"))
                    if f.lower().endswith(PHOTO_EXTS)):
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        corners, ids, _ = det.detectMarkers(img)
        if ids is None:
            continue
        fn = os.path.basename(p)
        for c, i in zip(corners, ids.flatten()):
            u, v = _quad_center(c[0])
            obs.setdefault(fn, {})[int(i)] = (float(u), float(v))
    return obs


# ----------------------------- pose loading (OpenMVG) -----------------------
def load_poses_openmvg(poses_json):
    """OpenMVG sfm_data JSON -> (intr, extr, views).
       intr id->(f,cx,cy,[k1,k2,k3]); extr id->(R,C); views filename->(id_intr,id_pose)."""
    d = json.load(open(poses_json))
    intr = {}
    for it in d["intrinsics"]:
        data = it["value"]["ptr_wrapper"]["data"]
        cx, cy = data["principal_point"]
        intr[it["key"]] = (data["focal_length"], cx, cy, data.get("disto_k3", [0, 0, 0]))
    extr = {}
    for e in d["extrinsics"]:
        extr[e["key"]] = (np.array(e["value"]["rotation"], float),
                          np.array(e["value"]["center"], float))
    views = {}
    for v in d["views"]:
        data = v["value"]["ptr_wrapper"]["data"]
        if data["id_pose"] in extr:
            views[data["filename"]] = (data["id_intrinsic"], data["id_pose"])
    return intr, extr, views


# ----------------------------- geometry -------------------------------------
def _undistort_norm(u, v, f, cx, cy, k):
    k1, k2, k3 = k
    xd, yd = (u - cx) / f, (v - cy) / f
    xu, yu = xd, yd
    for _ in range(10):
        r2 = xu * xu + yu * yu
        fac = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 ** 3
        xu, yu = xd / fac, yd / fac
    return np.array([xu, yu, 1.0])


def _triangulate(rays_and_poses):
    A = []
    for ray, R, C in rays_and_poses:
        P = np.hstack([R, (-R @ C).reshape(3, 1)])
        A.append(ray[0] * P[2] - P[0])
        A.append(ray[1] * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.array(A))
    X = Vt[-1]
    return X[:3] / X[3]


def _umeyama(src, dst):
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    Sc, Dc = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(Dc.T @ Sc / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / ((Sc ** 2).sum() / len(src))
    t = mu_d - s * R @ mu_s
    return s, R, t


def triangulate_markers(obs, intr, extr, views):
    """-> {marker_id: (X_sfm, n_views)} for markers in >=2 reconstructed views."""
    seen = {}
    for fn, md in obs.items():
        if fn not in views:
            continue
        ip, ipose = views[fn]
        f, cx, cy, k = intr[ip]
        R, C = extr[ipose]
        for mid, (u, v) in md.items():
            seen.setdefault(mid, []).append((_undistort_norm(u, v, f, cx, cy, k), R, C))
    return {mid: (_triangulate(rp), len(rp)) for mid, rp in seen.items() if len(rp) >= 2}


# ----------------------------- RANSAC fit -----------------------------------
def ransac_similarity(ids, src, dst, threshold=0.05, min_sample=3):
    """Robust 7-DOF fit. -> (s,R,t, inlier_ids, residuals{id:m}).
       >=4 ids: rejects outliers. <4: fits all (flag, can't drop)."""
    n = len(ids)
    if n < 3:
        return None
    if n < 4:
        s, R, t = _umeyama(src, dst)
        res = {ids[j]: float(np.linalg.norm(s * (R @ src[j]) + t - dst[j])) for j in range(n)}
        return s, R, t, list(ids), res
    best = None
    for combo in itertools.combinations(range(n), min_sample):
        s, R, t = _umeyama(src[list(combo)], dst[list(combo)])
        d = np.array([np.linalg.norm(s * (R @ src[j]) + t - dst[j]) for j in range(n)])
        inl = np.where(d <= threshold)[0]
        if len(inl) < min_sample:
            continue
        key = (len(inl), -np.sqrt((d[inl] ** 2).mean()))
        if best is None or key > best[0]:
            best = (key, inl)
    inl = best[1] if best else np.arange(n)
    s, R, t = _umeyama(src[inl], dst[inl])
    res = {ids[j]: float(np.linalg.norm(s * (R @ src[j]) + t - dst[j])) for j in range(n)}
    return s, R, t, [ids[j] for j in inl], res


# ----------------------------- offset / presets -----------------------------
def resolve_offset(gcp_coords, preset_offset=None):
    """preset_offset: explicit [x,y,z] shared origin (co-registers models), or None -> auto."""
    if preset_offset is not None:
        return np.asarray(preset_offset, float)
    c = np.mean([np.asarray(v, float) for v in gcp_coords.values()], axis=0)
    return np.floor(c)


# ----------------------------- apply to sfm_data ----------------------------
def apply_to_sfm_data(sfm_json, s, R, t):
    """Approach B: transform poses + sparse structure inside an OpenMVG sfm_data JSON,
       in place, by X' = s*R*X + t (the georef fit). Feed the result to openMVG2openMVS so
       dense/mesh run in the (local metric) georeferenced frame. R = fit rotation.
       Camera centres + structure points transform; rotations compose by R (scale-free)."""
    R = np.asarray(R, float); t = np.asarray(t, float)
    d = json.load(open(sfm_json))
    for e in d.get("extrinsics", []):
        v = e["value"]
        C = np.asarray(v["center"], float)
        v["center"] = (s * (R @ C) + t).tolist()
        v["rotation"] = (np.asarray(v["rotation"], float) @ R.T).tolist()
    for p in d.get("structure", []):
        X = np.asarray(p["value"]["X"], float)
        p["value"]["X"] = (s * (R @ X) + t).tolist()
    json.dump(d, open(sfm_json, "w"))


# ----------------------------- orchestration --------------------------------
def georeference(images_dir, poses_json, gcp_coords,
                 preset_offset=None, threshold=0.05, observations=None):
    """Decoupled georeferencing. gcp_coords {marker_id:(X,Y,Z)} real-world.
       observations: optional precomputed {file:{id:(u,v)}} (e.g. from manual web picks);
                     if None, auto-detect aruco. -> result dict."""
    intr, extr, views = load_poses_openmvg(poses_json)
    obs = observations if observations is not None else detect_observations(images_dir)
    tri = triangulate_markers(obs, intr, extr, views)

    usable = [i for i in sorted(gcp_coords) if i in tri]
    if len(usable) < 3:
        return {"georeferenced": False,
                "reason": f"only {len(usable)} markers triangulated (need >=3)",
                "coverage": {i: tri[i][1] for i in tri}}

    offset = resolve_offset({i: gcp_coords[i] for i in usable}, preset_offset)
    src = np.array([tri[i][0] for i in usable])
    dst = np.array([np.array(gcp_coords[i], float) - offset for i in usable])
    s, R, t, inliers, res = ransac_similarity(usable, src, dst, threshold)
    outliers = [i for i in usable if i not in inliers]
    rms = float(np.sqrt(np.mean([res[i] ** 2 for i in inliers])))

    return {"georeferenced": True, "scale": float(s), "offset": offset.tolist(),
            "inliers": inliers, "outliers": outliers, "residuals_m": res,
            "coverage": {i: tri[i][1] for i in usable},
            "rms_m": rms, "rms_mm": rms * 1000,
            "rotation": R.tolist(), "translation": t.tolist()}


if __name__ == "__main__":
    # Self-test on the tachy dataset — RANSAC should auto-drop marker 2 -> ~1cm.
    GCP = {0: (689261.0612975421, 5891561.924795985, 34.82699999958277),
           1: (689261.4918136494, 5891562.210813309, 34.82899999991059),
           2: (689262.4871601558, 5891561.852933174, 34.83100000023842),
           3: (689261.4956969041, 5891561.376302325, 34.829000000841916)}
    base = os.path.expanduser("~/openmvg-openmvs/test")
    r = georeference(os.path.join(base, "images"), os.path.join(base, "sfm", "poses.json"), GCP)
    print(json.dumps({k: v for k, v in r.items() if k not in ("rotation", "translation")}, indent=2))
