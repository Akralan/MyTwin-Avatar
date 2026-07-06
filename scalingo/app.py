#!/usr/bin/env python3
"""
MyTwin Avatar — Web app (Meshy AI) prête pour Scalingo.

Mobile-first : photo -> modèle 3D (Meshy « Image-to-3D »), affiché avec
<model-viewer> (orbite, pinch-zoom, AR). Le viewer n'apparaît qu'une fois le
modèle reçu.

Sécurité / protections :
  - Clé API Meshy 100 % côté serveur (jamais envoyée au navigateur).
  - Rate-limiting par IP sur /upload (protège tes crédits Meshy).
  - Code d'accès optionnel (ACCESS_CODE) pour réserver l'app à tes clients.
  - Images validées + ré-encodées + redimensionnées (Pillow) ; taille limitée.
  - En-têtes de sécurité (CSP, nosniff, anti-iframe…). HTTPS auto côté Scalingo.

Détourage : le fond des photos est retiré côté serveur (rembg / U²-Net) avant
l'envoi à Meshy -> meilleur mesh (pas de fond qui « bave » sur les bords).
Sortie en PNG transparent. Désactivable via REMOVE_BG=0.

Cache : même image + mêmes options = on réutilise le modèle déjà généré
(pas de nouvel appel Meshy -> pas de crédits consommés). Pratique pour les tests.

Variables d'environnement : voir README.md / scalingo.json.
"""
import os
import io
import time
import uuid
import shutil
import base64
import hashlib
import logging
import threading
from pathlib import Path
from functools import wraps

import requests
from PIL import Image
from flask import (Flask, request, jsonify, render_template, send_from_directory,
                   session, redirect, url_for, abort)
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("mytwin")

# --------------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------------- #
MESHY_API_KEY = os.environ.get("MESHY_API_KEY", "")
MESHY_BASE = "https://api.meshy.ai/openapi/v1"
SECRET_KEY = os.environ.get("SECRET_KEY") or os.urandom(32).hex()
ACCESS_CODE = os.environ.get("ACCESS_CODE", "").strip()

def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


# --- Mode test : court-circuite Meshy et sert un corps local (corps.glb). ---
# Évite les appels payants pendant les tests. Mettre USE_LOCAL_BODY=0 pour
# rebrancher Meshy. LOCAL_BODY_GLB permet de pointer un autre fichier.
USE_LOCAL_BODY = _env_bool("USE_LOCAL_BODY", True)
LOCAL_BODY_GLB = Path(os.environ.get("LOCAL_BODY_GLB",
                                     str(Path(__file__).with_name("corps.glb"))))

# --- Visage local (test) : greffe un visage fourni (visage.glb à la racine) au lieu
# du visage capturé par MediaPipe dans le navigateur. N'a d'effet que si le fichier
# existe. Mettre USE_LOCAL_FACE=0 pour rebrancher la capture ; LOCAL_FACE_GLB pointe
# un autre fichier. ---
USE_LOCAL_FACE = _env_bool("USE_LOCAL_FACE", True)
LOCAL_FACE_GLB = Path(os.environ.get("LOCAL_FACE_GLB",
                                     str(Path(__file__).with_name("visage.glb"))))

AI_MODEL = os.environ.get("MESHY_AI_MODEL", "latest")
POSE_MODE = os.environ.get("MESHY_POSE_MODE", "a-pose")
TARGET_POLYCOUNT = int(os.environ.get("MESHY_TARGET_POLYCOUNT", "30000"))

# Qualité (alignée sur la web app Meshy-6). Voir README.
# should_remesh=False -> maillage haute précision (défaut Meshy-6). Mettre à True
# (+ MESHY_TARGET_POLYCOUNT) pour un mesh allégé (AR / rigging).
SHOULD_REMESH = _env_bool("MESHY_SHOULD_REMESH", False)
ENABLE_PBR = _env_bool("MESHY_ENABLE_PBR", True)      # maps métallique/rugosité/normal
HD_TEXTURE = _env_bool("MESHY_HD_TEXTURE", True)      # texture 4K (Meshy-6 only)
POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL", "5"))
REMOTE_TIMEOUT = float(os.environ.get("REMOTE_TIMEOUT", "1200"))

MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "12"))
MAX_IMAGES = int(os.environ.get("MAX_IMAGES", "4"))  # Meshy multi-image : 1 à 4
MAX_IMAGE_SIDE = int(os.environ.get("MAX_IMAGE_SIDE", "1536"))

# Détourage du fond côté serveur (rembg). REMOVE_BG=0 pour désactiver.
# REMBG_MODEL : "u2net_human_seg" (corps/personnes, défaut) ou "u2net" (général).
REMOVE_BG = os.environ.get("REMOVE_BG", "1").strip().lower() not in ("0", "false", "no", "off")
REMBG_MODEL = os.environ.get("REMBG_MODEL", "u2net_human_seg")
RATE_LIMIT_GENERATE = os.environ.get("RATE_LIMIT_GENERATE", "8 per hour")
JOB_TTL = int(os.environ.get("JOB_TTL", "86400"))   # cache 24 h par défaut

DATA_DIR = Path(os.environ.get("DATA_DIR", "/tmp/mytwin_jobs"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXT = {"jpg", "jpeg", "png", "webp", "bmp", "tiff"}
ALLOWED_FILES = {"model.glb", "avatar.glb"}
_HEX = set("0123456789abcdef")

app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * MAX_IMAGES * 1024 * 1024
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)

limiter = Limiter(get_remote_address, app=app,
                  default_limits=["240 per hour"], storage_uri="memory://")

jobs: dict[str, dict] = {}
results_cache: dict[str, str] = {}   # clé image+options -> job_id dont le corps est prêt
jobs_lock = threading.Lock()


def _new_job() -> dict:
    return {"status": "queued", "stage": "en file d'attente",
            "ply_files": [], "tex_files": [], "person_count": 0,
            "body_url": None, "face_glb": False, "grafting": False,
            "error": None, "warning": None, "created": time.time()}


# --------------------------------------------------------------------------- #
#  Garde d'accès (code partagé optionnel)
# --------------------------------------------------------------------------- #
def require_access(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        if ACCESS_CODE and not session.get("ok"):
            if request.path.startswith(("/upload", "/face", "/status", "/output", "/model_status")):
                return jsonify({"error": "unauthorized"}), 401
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)
    return wrapper


# --------------------------------------------------------------------------- #
#  Helpers Meshy / images
# --------------------------------------------------------------------------- #
def _headers() -> dict:
    return {"Authorization": f"Bearer {MESHY_API_KEY}"}


_rembg_session = None
_rembg_lock = threading.Lock()


def _get_rembg_session():
    """Charge le modèle U²-Net une seule fois (thread-safe). Le modèle (~170 Mo)
    est téléchargé au premier appel puis mis en cache (~/.u2net)."""
    global _rembg_session
    if _rembg_session is None:
        with _rembg_lock:
            if _rembg_session is None:
                from rembg import new_session
                _rembg_session = new_session(REMBG_MODEL)
                log.info("rembg session ready (model=%s)", REMBG_MODEL)
    return _rembg_session


def _remove_bg(img: Image.Image) -> Image.Image:
    """Retire le fond -> image RGBA (fond transparent)."""
    from rembg import remove
    return remove(img, session=_get_rembg_session()).convert("RGBA")


def _prepare_image(file_storage, remove_bg: bool) -> str:
    """Valide, détoure (si remove_bg), redimensionne -> data URI base64 (déterministe).
    Sortie PNG transparent quand le détourage est actif, sinon JPEG."""
    try:
        probe = Image.open(file_storage.stream)
        probe.verify()
    except Exception:
        raise ValueError("Fichier image invalide.")
    file_storage.stream.seek(0)
    img = Image.open(file_storage.stream).convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_IMAGE_SIDE:
        s = MAX_IMAGE_SIDE / max(w, h)
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
    buf = io.BytesIO()
    if remove_bg:
        _remove_bg(img).save(buf, format="PNG", optimize=True)
        mime = "image/png"
    else:
        img.save(buf, format="JPEG", quality=90)
        mime = "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _cache_key(image_uris: list, texture: bool, apose: bool) -> str:
    h = hashlib.sha256("".join(image_uris).encode("ascii")).hexdigest()
    return f"{h}:{int(texture)}:{int(apose)}"


def _create_task(image_uris: list, enable_texture: bool, enable_apose: bool) -> str:
    body = {
        "image_urls": list(image_uris),
        "ai_model": AI_MODEL,
        "should_texture": bool(enable_texture),
        "should_remesh": SHOULD_REMESH,
        "target_formats": ["glb"],
        "pose_mode": POSE_MODE if enable_apose else "",
    }
    # target_polycount n'a d'effet que si l'on remeshe (mesh allégé).
    if SHOULD_REMESH:
        body["target_polycount"] = TARGET_POLYCOUNT
    # Qualité texture : seulement si on texture le modèle.
    if enable_texture:
        body["enable_pbr"] = ENABLE_PBR
        body["hd_texture"] = HD_TEXTURE
    r = requests.post(f"{MESHY_BASE}/multi-image-to-3d", json=body,
                      headers=_headers(), timeout=60)
    r.raise_for_status()
    return r.json()["result"]


def _poll(task_id: str, job_id: str) -> dict:
    deadline = time.time() + REMOTE_TIMEOUT
    while time.time() < deadline:
        r = requests.get(f"{MESHY_BASE}/multi-image-to-3d/{task_id}",
                         headers=_headers(), timeout=30)
        r.raise_for_status()
        d = r.json()
        st = d.get("status")
        with jobs_lock:
            if job_id in jobs:
                jobs[job_id]["stage"] = f"{st} {d.get('progress', 0)}%"
        if st == "SUCCEEDED":
            return d
        if st in ("FAILED", "CANCELED"):
            raise RuntimeError((d.get("task_error") or {}).get("message")
                               or f"Génération {st}")
        time.sleep(POLL_INTERVAL)
    raise TimeoutError("Délai dépassé.")


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=180, stream=True)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)


def _worker(job_id, image_uris, enable_texture, enable_apose, cache_key):
    try:
        with jobs_lock:
            jobs[job_id]["status"] = "processing"
        if USE_LOCAL_BODY:
            # Mode test : on saute Meshy et on réutilise le corps local.
            log.info("USE_LOCAL_BODY : job %s -> %s (pas d'appel Meshy)", job_id, LOCAL_BODY_GLB)
            (DATA_DIR / job_id).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(LOCAL_BODY_GLB, DATA_DIR / job_id / "model.glb")
        else:
            task_id = _create_task(image_uris, enable_texture, enable_apose)
            log.info("meshy task %s (should_texture=%s) for job %s", task_id, enable_texture, job_id)
            data = _poll(task_id, job_id)
            glb = (data.get("model_urls") or {}).get("glb")
            if not glb:
                raise RuntimeError("Aucun modèle renvoyé.")
            _download(glb, DATA_DIR / job_id / "model.glb")
        with jobs_lock:
            results_cache[cache_key] = job_id     # cache du corps (évite un appel Meshy)
        _mark_body_ready(job_id)
    except requests.HTTPError as exc:
        code = getattr(exc.response, "status_code", "?")
        log.warning("job %s HTTP %s: %s", job_id, code,
                    getattr(exc.response, "text", "")[:200])
        with jobs_lock:
            jobs[job_id].update(status="error", ply_files=[],
                                error="Le service de génération a renvoyé une erreur.")
    except Exception as exc:
        log.warning("job %s: %s", job_id, exc)
        with jobs_lock:
            jobs[job_id].update(status="error", ply_files=[],
                                error="La génération a échoué.")


def _mark_body_ready(job_id: str) -> None:
    """Le corps (model.glb) est en place. On greffe le visage s'il est déjà arrivé,
    sinon on passe en attente du visage (l'utilisateur le capture pendant ce temps)."""
    start = False
    with jobs_lock:
        j = jobs.get(job_id)
        if not j:
            return
        url = f"/output/{job_id}/model.glb"
        j.update(body_url=url, ply_files=[url], tex_files=[url], person_count=1)
        if (j.get("face_glb") and (DATA_DIR / job_id / "face.glb").exists()
                and not j.get("grafting")):
            j.update(status="grafting", stage="application du visage", grafting=True)
            start = True
        elif j.get("status") not in ("grafting", "done"):
            j.update(status="awaiting_face", stage="corps prêt")
    if start:
        threading.Thread(target=_graft_worker, args=(job_id,), daemon=True).start()


def _graft_worker(job_id: str) -> None:
    """Greffe le visage scanné (face.glb, topologie MediaPipe canonique) sur le
    corps via le pipeline face-replacement (orchestrator.replace_face), puis écrit
    avatar.glb. Chirurgie de maillage : découpe du visage du corps, alignement
    feature-to-feature du scan, jonction C1, harmonisation ton/grain — pas un simple
    baking de texture."""
    try:
        # Imports lourds (mediapipe/trimesh/scipy/opencv) chargés à la demande.
        from pipeline.orchestrator import replace_face
        body_path = DATA_DIR / job_id / "model.glb"
        face_path = DATA_DIR / job_id / "face.glb"
        out = DATA_DIR / job_id / "avatar.glb"
        _, report = replace_face(str(body_path), str(face_path), str(out))
        url = f"/output/{job_id}/avatar.glb"
        with jobs_lock:
            jobs[job_id].update(status="done", stage="terminé",
                                ply_files=[url], tex_files=[url])
        log.info("graft done job %s: method=%s residual=%s warnings=%s",
                 job_id, report.method, report.landmark_residual, report.warnings)
    except Exception as exc:
        log.warning("graft job %s failed: %s", job_id, exc)
        with jobs_lock:
            j = jobs.get(job_id)
            if not j:
                return
            burl = j.get("body_url")
            if burl:   # repli : on livre au moins le corps
                j.update(status="done", stage="visage non appliqué",
                         ply_files=[burl], tex_files=[burl],
                         warning="Le visage n'a pas pu être appliqué au corps.")
            else:
                j.update(status="error", error="Échec de la greffe du visage.")


def _cleanup():
    now = time.time()
    with jobs_lock:
        stale = [j for j, v in jobs.items() if now - v.get("created", now) > JOB_TTL]
        for j in stale:
            jobs.pop(j, None)
        for k, v in list(results_cache.items()):
            if v in stale:
                results_cache.pop(k, None)
    for j in stale:
        try:
            (DATA_DIR / j / "model.glb").unlink()
            (DATA_DIR / j).rmdir()
        except OSError:
            pass


# --------------------------------------------------------------------------- #
#  Routes
# --------------------------------------------------------------------------- #
@app.route("/login", methods=["GET", "POST"])
def login():
    if not ACCESS_CODE:
        return redirect(url_for("index"))
    if request.method == "POST":
        if request.form.get("code", "") == ACCESS_CODE:
            session["ok"] = True
            session.permanent = True
            nxt = request.args.get("next", "")
            return redirect(nxt if nxt.startswith("/") else url_for("index"))
        return render_template("login.html", error="Code incorrect."), 401
    return render_template("login.html", error=None)


@app.route("/")
@require_access
def index():
    return render_template("index.html")


@app.route("/model_status")
@require_access
def model_status():
    if not MESHY_API_KEY and not USE_LOCAL_BODY:
        return jsonify({"ready": False, "error": "Service non configuré (clé API manquante)."})
    return jsonify({"ready": True, "error": None})


@app.route("/upload", methods=["POST"])
@require_access
@limiter.limit(RATE_LIMIT_GENERATE)
def upload():
    if not MESHY_API_KEY and not USE_LOCAL_BODY:
        return jsonify({"error": "Service non configuré."}), 503
    files = [f for f in request.files.getlist("image") if f and f.filename]
    if not files:
        return jsonify({"error": "Aucune image fournie."}), 400
    if len(files) > MAX_IMAGES:
        return jsonify({"error": f"Maximum {MAX_IMAGES} images."}), 400

    truthy = ("1", "true", "on", "yes", "True")
    enable_texture = request.form.get("texture", "0") in truthy
    enable_apose = request.form.get("apose", "1") in truthy
    # REMOVE_BG (env) = interrupteur maître ; le switch UI affine par requête.
    enable_removebg = REMOVE_BG and request.form.get("removebg", "1") in truthy

    image_uris = []
    for f in files:
        ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
        if ext not in ALLOWED_EXT:
            return jsonify({"error": f"Format non supporté : {ext}"}), 400
        try:
            image_uris.append(_prepare_image(f, enable_removebg))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

    log.info("upload: images=%d texture=%s apose=%s removebg=%s",
             len(image_uris), enable_texture, enable_apose, enable_removebg)

    _cleanup()
    job_id = uuid.uuid4().hex
    with jobs_lock:
        jobs[job_id] = _new_job()

    # --- Cache du corps : mêmes images + options -> on réutilise le model.glb déjà
    # généré (pas d'appel Meshy). Le visage diffère à chaque session, donc on copie
    # le corps dans ce nouveau job et on enchaîne la greffe. ---
    key = _cache_key(image_uris, enable_texture, enable_apose)
    with jobs_lock:
        cached = results_cache.get(key)
        body_cached = bool(cached and cached != job_id
                           and (DATA_DIR / cached / "model.glb").exists())
    if body_cached:
        log.info("cache HIT corps (texture=%s apose=%s) %s -> %s",
                 enable_texture, enable_apose, cached, job_id)
        (DATA_DIR / job_id).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(DATA_DIR / cached / "model.glb", DATA_DIR / job_id / "model.glb")
        _mark_body_ready(job_id)
        return jsonify({"job_id": job_id, "cached": True})

    threading.Thread(target=_worker,
                     args=(job_id, image_uris, enable_texture, enable_apose, key),
                     daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
@require_access
@limiter.exempt   # sondé toutes les 3 s par le front : ne doit pas compter dans la limite globale
def status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job introuvable."}), 404
    return jsonify(job)


@app.route("/face/<job_id>", methods=["POST"])
@require_access
def upload_face(job_id):
    if len(job_id) != 32 or any(c not in _HEX for c in job_id):
        return jsonify({"error": "Job invalide."}), 400
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job introuvable."}), 404

    f = request.files.get("face")
    if not f or not f.filename:
        return jsonify({"error": "Aucun visage fourni."}), 400
    ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
    if ext not in ("glb", "gltf"):
        return jsonify({"error": "Le visage doit être un GLB."}), 400

    dest = DATA_DIR / job_id / "face.glb"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if USE_LOCAL_FACE and LOCAL_FACE_GLB.exists():
        # Mode test : on ignore le visage du navigateur et on greffe le visage local
        # fourni (visage.glb à la racine) — utile pour tester la greffe avec un scan de
        # référence, indépendamment de la capture MediaPipe.
        log.info("USE_LOCAL_FACE : job %s -> %s (visage du navigateur ignoré)",
                 job_id, LOCAL_FACE_GLB)
        shutil.copyfile(LOCAL_FACE_GLB, dest)
    else:
        f.save(dest)
    # Garde-fou : GLB binaire valide (magic "glTF").
    with open(dest, "rb") as fh:
        if fh.read(4) != b"glTF":
            dest.unlink(missing_ok=True)
            return jsonify({"error": "Fichier GLB invalide."}), 400

    start = False
    with jobs_lock:
        j = jobs.get(job_id)
        if not j:
            return jsonify({"error": "Job introuvable."}), 404
        j["face_glb"] = True
        if (j.get("body_url") and (DATA_DIR / job_id / "model.glb").exists()
                and not j.get("grafting")):
            j.update(status="grafting", stage="application du visage", grafting=True)
            start = True
    if start:
        threading.Thread(target=_graft_worker, args=(job_id,), daemon=True).start()

    with jobs_lock:
        return jsonify({"ok": True, "status": jobs[job_id]["status"]})


@app.route("/output/<job_id>/<path:filename>")
@require_access
def serve_output(job_id, filename):
    if len(job_id) != 32 or any(c not in _HEX for c in job_id) or filename not in ALLOWED_FILES:
        abort(404)
    directory = DATA_DIR / job_id
    if not (directory / filename).exists():
        abort(404)
    return send_from_directory(directory, filename,
                               mimetype="model/gltf-binary", max_age=3600)


@app.route("/models/face_landmarker.task")
def face_landmarker_model():
    """Sert le modèle MediaPipe FaceLandmarker au front (même fichier que le pipeline
    backend). Permet au navigateur d'utiliser exactement le modèle d'Android, sans
    dépendre d'un CDN tiers (CSP connect-src 'self')."""
    models_dir = Path(__file__).resolve().parent / "models"
    if not (models_dir / "face_landmarker.task").exists():
        abort(404)
    return send_from_directory(models_dir, "face_landmarker.task",
                               mimetype="application/octet-stream", max_age=86400)


@app.route("/healthz")
def healthz():
    return jsonify({"ok": True, "configured": bool(MESHY_API_KEY)})


# --------------------------------------------------------------------------- #
#  En-têtes de sécurité + erreurs
# --------------------------------------------------------------------------- #
@app.after_request
def security_headers(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "SAMEORIGIN"
    resp.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    resp.headers["Permissions-Policy"] = "camera=(self), geolocation=()"
    resp.headers["Content-Security-Policy"] = (
        "default-src 'self'; base-uri 'self'; form-action 'self'; "
        # 'unsafe-eval' est requis par le runtime WASM de MediaPipe FaceMesh
        # (face_mesh 0.4 utilise new Function()), sinon la détection du visage échoue.
        "script-src 'self' 'unsafe-inline' 'unsafe-eval' 'wasm-unsafe-eval' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com data:; "
        "img-src 'self' data: blob:; "
        "connect-src 'self' blob: data: https://cdn.jsdelivr.net https://www.gstatic.com; "
        "worker-src 'self' blob:; frame-ancestors 'self'"
    )
    return resp


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": f"Image trop lourde (max {MAX_UPLOAD_MB} Mo)."}), 413


@app.errorhandler(429)
def rate_limited(_):
    return jsonify({"error": "Trop de générations. Réessaie plus tard."}), 429


# Pré-charge le modèle de détourage au boot (en arrière-plan) pour éviter le
# téléchargement du modèle lors de la première génération. Exécuté à l'import,
# donc actif aussi sous gunicorn.
if REMOVE_BG:
    threading.Thread(target=_get_rembg_session, daemon=True).start()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")))
