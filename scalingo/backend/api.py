#!/usr/bin/env python3
"""
MyTwin Avatar — Backend API (sans état) pour conteneur serverless Scaleway.

Le navigateur orchestre tout et stocke les avatars sur l'appareil (IndexedDB) ;
ce backend n'a AUCUN état (pas de DB, pas de disque persistant, pas de threads
d'arrière-plan). Il expose 3 endpoints :

  POST /body                 images + options -> crée la tâche Meshy -> {task_id}
  GET  /body/status?task_id  proxy court vers Meshy -> {status, progress}
  POST /graft                face.glb + task_id -> pipeline CPU -> renvoie avatar.glb

L'attente longue (génération du corps par Meshy) se passe côté navigateur, qui
poll /body/status pendant que l'utilisateur capture son visage. La greffe est
synchrone (une requête/réponse) et renvoie le GLB dans le corps de la réponse.

Modes test (aucun appel Meshy / crédit), pilotés PAR REQUÊTE via les toggles du
frontend (désactivés par défaut) :
  local_body=1  -> /body renvoie task_id="local", /graft greffe sur corps.glb
  local_face=1  -> /graft ignore le visage reçu et greffe visage.glb

Clé Meshy 100 % serveur. CORS ouvert (démo) via CORS_ORIGIN (défaut *).
Variables : MESHY_API_KEY, MESHY_*, REMOVE_BG, REMBG_MODEL, CORS_ORIGIN.
"""
import os
import io
import time
import base64
import shutil
import logging
import tempfile
import threading
from pathlib import Path

import requests
from PIL import Image
from flask import Flask, request, jsonify, Response
from werkzeug.middleware.proxy_fix import ProxyFix

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("mytwin-api")

HERE = Path(__file__).resolve().parent


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _req_bool(name: str, default: bool = False) -> bool:
    """Flag booléen d'une requête (form d'abord, sinon query string)."""
    raw = request.form.get(name)
    if raw is None:
        raw = request.args.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "on", "yes")


# --------------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------------- #
MESHY_API_KEY = os.environ.get("MESHY_API_KEY", "")
MESHY_BASE = "https://api.meshy.ai/openapi/v1"
CORS_ORIGIN = os.environ.get("CORS_ORIGIN", "*")

# Modes test (voir docstring) : activés par requête via les toggles du frontend
# (local_body / local_face), désactivés par défaut — plus de variable d'env.
LOCAL_BODY_GLB = Path(os.environ.get("LOCAL_BODY_GLB", str(HERE / "corps.glb")))
LOCAL_FACE_GLB = Path(os.environ.get("LOCAL_FACE_GLB", str(HERE / "visage.glb")))

AI_MODEL = os.environ.get("MESHY_AI_MODEL", "latest")
POSE_MODE = os.environ.get("MESHY_POSE_MODE", "a-pose")
TARGET_POLYCOUNT = int(os.environ.get("MESHY_TARGET_POLYCOUNT", "30000"))
SHOULD_REMESH = _env_bool("MESHY_SHOULD_REMESH", False)
ENABLE_PBR = _env_bool("MESHY_ENABLE_PBR", True)
HD_TEXTURE = _env_bool("MESHY_HD_TEXTURE", False)
REMOTE_TIMEOUT = float(os.environ.get("REMOTE_TIMEOUT", "1200"))

MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "12"))
MAX_IMAGES = int(os.environ.get("MAX_IMAGES", "4"))
MAX_IMAGE_SIDE = int(os.environ.get("MAX_IMAGE_SIDE", "1536"))
REMOVE_BG = _env_bool("REMOVE_BG", True)
REMBG_MODEL = os.environ.get("REMBG_MODEL", "u2net_human_seg")

ALLOWED_EXT = {"jpg", "jpeg", "png", "webp", "bmp", "tiff"}
LOCAL_TASK = "local"

app = Flask(__name__)
# Marge : images du corps + face.glb (peut embarquer un JPEG plein cadre).
app.config["MAX_CONTENT_LENGTH"] = (MAX_UPLOAD_MB * MAX_IMAGES + 40) * 1024 * 1024
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)


# --------------------------------------------------------------------------- #
#  Détourage (rembg) + préparation image
# --------------------------------------------------------------------------- #
_rembg_session = None
_rembg_lock = threading.Lock()


def _get_rembg_session():
    """Charge le modèle U²-Net une seule fois (thread-safe)."""
    global _rembg_session
    if _rembg_session is None:
        with _rembg_lock:
            if _rembg_session is None:
                from rembg import new_session
                _rembg_session = new_session(REMBG_MODEL)
                log.info("rembg session ready (model=%s)", REMBG_MODEL)
    return _rembg_session


def _remove_bg(img: Image.Image) -> Image.Image:
    from rembg import remove
    return remove(img, session=_get_rembg_session()).convert("RGBA")


def _prepare_image(file_storage, remove_bg: bool) -> str:
    """Valide, détoure (option), redimensionne -> data URI base64 pour Meshy."""
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


# --------------------------------------------------------------------------- #
#  Meshy (sans état : Meshy détient la tâche)
# --------------------------------------------------------------------------- #
def _headers() -> dict:
    return {"Authorization": f"Bearer {MESHY_API_KEY}"}


def _create_task(image_uris: list, enable_texture: bool, enable_apose: bool) -> str:
    body = {
        "image_urls": list(image_uris),
        "ai_model": AI_MODEL,
        "should_texture": bool(enable_texture),
        "should_remesh": SHOULD_REMESH,
        "target_formats": ["glb"],
        "pose_mode": POSE_MODE if enable_apose else "",
    }
    if SHOULD_REMESH:
        body["target_polycount"] = TARGET_POLYCOUNT
    if enable_texture:
        body["enable_pbr"] = ENABLE_PBR
        body["hd_texture"] = HD_TEXTURE
    r = requests.post(f"{MESHY_BASE}/multi-image-to-3d", json=body,
                      headers=_headers(), timeout=60)
    r.raise_for_status()
    return r.json()["result"]


def _get_task(task_id: str) -> dict:
    r = requests.get(f"{MESHY_BASE}/multi-image-to-3d/{task_id}",
                     headers=_headers(), timeout=30)
    r.raise_for_status()
    return r.json()


def _download(url: str, dest: Path) -> None:
    r = requests.get(url, timeout=180, stream=True)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)


# --------------------------------------------------------------------------- #
#  CORS (démo : origine unique via CORS_ORIGIN, défaut *)
# --------------------------------------------------------------------------- #
@app.after_request
def cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = CORS_ORIGIN
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["X-Content-Type-Options"] = "nosniff"
    return resp


@app.route("/body", methods=["OPTIONS"])
@app.route("/graft", methods=["OPTIONS"])
@app.route("/body/status", methods=["OPTIONS"])
def _preflight():
    return ("", 204)


# --------------------------------------------------------------------------- #
#  Endpoints
# --------------------------------------------------------------------------- #
@app.route("/healthz")
def healthz():
    return jsonify({"ok": True, "configured": bool(MESHY_API_KEY)})


@app.route("/body", methods=["POST"])
def body():
    """Prépare les images (détourage) et crée la tâche Meshy. Renvoie le task_id ;
    le navigateur poll /body/status. En mode local, court-circuite Meshy."""
    if _req_bool("local_body"):
        return jsonify({"task_id": LOCAL_TASK})
    if not MESHY_API_KEY:
        return jsonify({"error": "Service non configuré."}), 503

    files = [f for f in request.files.getlist("image") if f and f.filename]
    if not files:
        return jsonify({"error": "Aucune image fournie."}), 400
    if len(files) > MAX_IMAGES:
        return jsonify({"error": f"Maximum {MAX_IMAGES} images."}), 400

    truthy = ("1", "true", "on", "yes", "True")
    enable_texture = request.form.get("texture", "0") in truthy
    enable_apose = request.form.get("apose", "1") in truthy
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

    try:
        task_id = _create_task(image_uris, enable_texture, enable_apose)
    except requests.HTTPError as exc:
        code = getattr(exc.response, "status_code", "?")
        log.warning("meshy create HTTP %s: %s", code, getattr(exc.response, "text", "")[:200])
        return jsonify({"error": "Le service de génération a renvoyé une erreur."}), 502
    log.info("meshy task %s (texture=%s apose=%s removebg=%s)",
             task_id, enable_texture, enable_apose, enable_removebg)
    return jsonify({"task_id": task_id})


@app.route("/body/status")
def body_status():
    """Proxy court vers Meshy : statut + progression. Sans état."""
    task_id = request.args.get("task_id", "")
    if not task_id:
        return jsonify({"error": "task_id manquant."}), 400
    if task_id == LOCAL_TASK:
        return jsonify({"status": "succeeded", "progress": 100})
    try:
        d = _get_task(task_id)
    except requests.HTTPError as exc:
        code = getattr(exc.response, "status_code", "?")
        return jsonify({"error": f"Meshy {code}"}), 502
    st = (d.get("status") or "").upper()
    mapping = {"PENDING": "queued", "IN_PROGRESS": "processing",
               "SUCCEEDED": "succeeded", "FAILED": "failed", "CANCELED": "failed"}
    out = {"status": mapping.get(st, "processing"), "progress": d.get("progress", 0)}
    if out["status"] == "failed":
        out["error"] = (d.get("task_error") or {}).get("message") or "Génération échouée."
    return jsonify(out)


def _resolve_body(task_id: str, dest: Path) -> None:
    """Écrit le corps (model.glb) dans dest : corps local en mode test, sinon
    télécharge le résultat Meshy de la tâche."""
    if task_id == LOCAL_TASK:
        shutil.copyfile(LOCAL_BODY_GLB, dest)
        return
    d = _get_task(task_id)
    if (d.get("status") or "").upper() != "SUCCEEDED":
        raise RuntimeError("Le corps n'est pas encore prêt.")
    glb = (d.get("model_urls") or {}).get("glb")
    if not glb:
        raise RuntimeError("Aucun modèle renvoyé par Meshy.")
    _download(glb, dest)


@app.route("/graft", methods=["POST"])
def graft():
    """Greffe synchrone : corps (résolu depuis task_id) + visage -> avatar.glb dans
    la réponse. Aucun fichier conservé (tmp éphémère nettoyé)."""
    task_id = request.form.get("task_id", "")
    if not task_id:
        return jsonify({"error": "task_id manquant."}), 400

    tmp = Path(tempfile.mkdtemp(prefix="graft_"))
    try:
        body_path = tmp / "model.glb"
        face_path = tmp / "face.glb"
        out_path = tmp / "avatar.glb"

        # Visage : local (test) ou reçu du navigateur.
        if _req_bool("local_face") and LOCAL_FACE_GLB.exists():
            log.info("local_face : visage du navigateur ignoré (%s)", LOCAL_FACE_GLB)
            shutil.copyfile(LOCAL_FACE_GLB, face_path)
        else:
            f = request.files.get("face")
            if not f or not f.filename:
                return jsonify({"error": "Aucun visage fourni."}), 400
            f.save(face_path)
            with open(face_path, "rb") as fh:
                if fh.read(4) != b"glTF":
                    return jsonify({"error": "Fichier GLB invalide."}), 400

        # Corps : résolu depuis Meshy (ou corps local).
        try:
            _resolve_body(task_id, body_path)
        except requests.HTTPError:
            return jsonify({"error": "Corps introuvable côté Meshy."}), 502
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

        # Greffe (imports lourds à la demande : mediapipe/trimesh/scipy/opencv).
        from pipeline.orchestrator import replace_face
        _, report = replace_face(str(body_path), str(face_path), str(out_path))
        log.info("graft ok: method=%s residual=%s warnings=%s",
                 report.method, report.landmark_residual, report.warnings)
        data = out_path.read_bytes()
        return Response(data, mimetype="model/gltf-binary",
                        headers={"Content-Disposition": "inline; filename=avatar.glb"})
    except Exception as exc:
        log.warning("graft failed: %s", exc)
        return jsonify({"error": "La greffe du visage a échoué."}), 500
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _preload_pipeline():
    """Importe la chaîne de greffe lourde au boot (mediapipe/scipy/trimesh/opencv +
    cache polices matplotlib). Sans ça, la 1re greffe d'une instance paie ~150 s
    d'imports *dans* la requête (voir l'import différé dans /graft) — mesuré sur le
    conteneur Scaleway. Ici c'est absorbé au démarrage, hors du timeout requête."""
    try:
        import pipeline.orchestrator  # noqa: F401  (déclenche tous les imports lourds)
        log.info("pipeline preloaded (mediapipe/scipy/trimesh/opencv ready)")
    except Exception as exc:  # noqa: BLE001
        log.warning("pipeline preload failed: %s", exc)


# Pré-charges au boot (hors requête) : rembg (détourage, mode par défaut) et la
# chaîne de greffe. Avec min-scale>=1, l'instance reste chaude et ne les repaie pas.
if REMOVE_BG:
    threading.Thread(target=_get_rembg_session, daemon=True).start()
threading.Thread(target=_preload_pipeline, daemon=True).start()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5001")))
