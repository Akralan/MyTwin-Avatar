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

Cache : même image + mêmes options = on réutilise le modèle déjà généré
(pas de nouvel appel Meshy -> pas de crédits consommés). Pratique pour les tests.

Variables d'environnement : voir README.md / scalingo.json.
"""
import os
import io
import time
import uuid
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

AI_MODEL = os.environ.get("MESHY_AI_MODEL", "latest")
POSE_MODE = os.environ.get("MESHY_POSE_MODE", "a-pose")
TARGET_POLYCOUNT = int(os.environ.get("MESHY_TARGET_POLYCOUNT", "30000"))
POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL", "5"))
REMOTE_TIMEOUT = float(os.environ.get("REMOTE_TIMEOUT", "1200"))

MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "12"))
MAX_IMAGES = int(os.environ.get("MAX_IMAGES", "4"))  # Meshy multi-image : 1 à 4
MAX_IMAGE_SIDE = int(os.environ.get("MAX_IMAGE_SIDE", "1536"))
RATE_LIMIT_GENERATE = os.environ.get("RATE_LIMIT_GENERATE", "8 per hour")
JOB_TTL = int(os.environ.get("JOB_TTL", "86400"))   # cache 24 h par défaut

DATA_DIR = Path(os.environ.get("DATA_DIR", "/tmp/mytwin_jobs"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXT = {"jpg", "jpeg", "png", "webp", "bmp", "tiff"}
ALLOWED_FILES = {"model.glb"}
_HEX = set("0123456789abcdef")

app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * MAX_IMAGES * 1024 * 1024
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)

limiter = Limiter(get_remote_address, app=app,
                  default_limits=["240 per hour"], storage_uri="memory://")

jobs: dict[str, dict] = {}
results_cache: dict[str, str] = {}   # clé image+options -> job_id terminé
jobs_lock = threading.Lock()


# --------------------------------------------------------------------------- #
#  Garde d'accès (code partagé optionnel)
# --------------------------------------------------------------------------- #
def require_access(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        if ACCESS_CODE and not session.get("ok"):
            if request.path.startswith(("/upload", "/status", "/output", "/model_status")):
                return jsonify({"error": "unauthorized"}), 401
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)
    return wrapper


# --------------------------------------------------------------------------- #
#  Helpers Meshy / images
# --------------------------------------------------------------------------- #
def _headers() -> dict:
    return {"Authorization": f"Bearer {MESHY_API_KEY}"}


def _prepare_image(file_storage) -> str:
    """Valide, convertit, redimensionne -> data URI JPEG base64 (déterministe)."""
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
    img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _cache_key(image_uris: list, texture: bool, apose: bool) -> str:
    h = hashlib.sha256("".join(image_uris).encode("ascii")).hexdigest()
    return f"{h}:{int(texture)}:{int(apose)}"


def _create_task(image_uris: list, enable_texture: bool, enable_apose: bool) -> str:
    body = {
        "image_urls": list(image_uris),
        "ai_model": AI_MODEL,
        "should_texture": bool(enable_texture),
        "should_remesh": True,
        "target_polycount": TARGET_POLYCOUNT,
        "target_formats": ["glb"],
        "pose_mode": POSE_MODE if enable_apose else "",
    }
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
        task_id = _create_task(image_uris, enable_texture, enable_apose)
        log.info("meshy task %s (should_texture=%s) for job %s", task_id, enable_texture, job_id)
        data = _poll(task_id, job_id)
        glb = (data.get("model_urls") or {}).get("glb")
        if not glb:
            raise RuntimeError("Aucun modèle renvoyé.")
        _download(glb, DATA_DIR / job_id / "model.glb")
        url = f"/output/{job_id}/model.glb"
        with jobs_lock:
            jobs[job_id].update(status="done", stage="terminé",
                                ply_files=[url], glb_files=[], pose_files=[],
                                tex_files=[url], person_count=1)
            results_cache[cache_key] = job_id
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
    if not MESHY_API_KEY:
        return jsonify({"ready": False, "error": "Service non configuré (clé API manquante)."})
    return jsonify({"ready": True, "error": None})


@app.route("/upload", methods=["POST"])
@require_access
@limiter.limit(RATE_LIMIT_GENERATE)
def upload():
    if not MESHY_API_KEY:
        return jsonify({"error": "Service non configuré."}), 503
    files = [f for f in request.files.getlist("image") if f and f.filename]
    if not files:
        return jsonify({"error": "Aucune image fournie."}), 400
    if len(files) > MAX_IMAGES:
        return jsonify({"error": f"Maximum {MAX_IMAGES} images."}), 400

    image_uris = []
    for f in files:
        ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
        if ext not in ALLOWED_EXT:
            return jsonify({"error": f"Format non supporté : {ext}"}), 400
        try:
            image_uris.append(_prepare_image(f))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

    truthy = ("1", "true", "on", "yes", "True")
    enable_texture = request.form.get("texture", "0") in truthy
    enable_apose = request.form.get("apose", "1") in truthy

    log.info("upload: images=%d texture=%s apose=%s", len(image_uris), enable_texture, enable_apose)

    # --- Cache : mêmes images + mêmes options -> on réutilise le modèle existant ---
    key = _cache_key(image_uris, enable_texture, enable_apose)
    with jobs_lock:
        cached = results_cache.get(key)
        if (cached and cached in jobs and jobs[cached].get("status") == "done"
                and (DATA_DIR / cached / "model.glb").exists()):
            log.info("cache HIT (texture=%s apose=%s) -> %s", enable_texture, enable_apose, cached)
            return jsonify({"job_id": cached, "cached": True})

    _cleanup()
    job_id = uuid.uuid4().hex
    with jobs_lock:
        jobs[job_id] = {"status": "queued", "ply_files": [], "glb_files": [],
                        "pose_files": [], "tex_files": [], "person_count": 0,
                        "error": None, "created": time.time()}
    threading.Thread(target=_worker,
                     args=(job_id, image_uris, enable_texture, enable_apose, key),
                     daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
@require_access
def status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job introuvable."}), 404
    return jsonify(job)


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
        "script-src 'self' 'unsafe-inline' 'wasm-unsafe-eval' https://cdn.jsdelivr.net; "
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")))
