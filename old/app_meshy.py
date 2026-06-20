#!/usr/bin/env python3
"""
MyTwin Avatar — variante Meshy AI.
Run: python app_meshy.py

Même interface graphique (templates/index.html) que app.py : mêmes routes et même
forme de réponse JSON. Seul le backend change : au lieu de GeneMAN/UniTEX, on appelle
l'API Meshy AI « Image-to-3D » (dernière version, /openapi/v1).

Mapping avec l'UI existante :
  - case "texture" cochée   -> modèle TEXTURÉ (glb)   -> tex_files (viewer GLTFLoader)
  - case "texture" décochée -> modèle géométrie (obj) -> ply_files (viewer OBJLoader)
  - case "rig"              -> non supportée par Meshy image-to-3d (renvoie vide)

Configuration (variables d'environnement) :
  MESHY_API_KEY           (obligatoire) clé API : https://www.meshy.ai/settings/api
  MESHY_TARGET_POLYCOUNT  (def. 30000)  densité du mesh remeshé
  MESHY_AI_MODEL          (def. "latest")
  MESHY_ENABLE_PBR        (def. "0")    cartes PBR (metallic/roughness/normal)
  POLL_INTERVAL           (def. 5   s)  intervalle de polling
  REMOTE_TIMEOUT          (def. 1200 s) délai max par tâche Meshy
"""

import os
import uuid
import time
import base64
import mimetypes
import threading
from pathlib import Path

import requests
from flask import Flask, request, jsonify, send_from_directory, render_template

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("output")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Configuration Meshy ---
MESHY_API_KEY = os.environ.get("MESHY_API_KEY", "")
MESHY_BASE = "https://api.meshy.ai/openapi/v1"
TARGET_POLYCOUNT = int(os.environ.get("MESHY_TARGET_POLYCOUNT", "30000"))
AI_MODEL = os.environ.get("MESHY_AI_MODEL", "latest")
ENABLE_PBR = os.environ.get("MESHY_ENABLE_PBR", "0") in ("1", "true", "on", "True")
POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL", "5"))
REMOTE_TIMEOUT = float(os.environ.get("REMOTE_TIMEOUT", "1200"))

jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


# --------------------------------------------------------------------------- #
#  Appels API Meshy
# --------------------------------------------------------------------------- #
def _headers() -> dict:
    return {"Authorization": f"Bearer {MESHY_API_KEY}"}


def _image_data_uri(image_path: str) -> str:
    """Encode l'image locale en data URI base64 (accepté par image_url)."""
    mime = mimetypes.guess_type(image_path)[0] or "image/png"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _create_task(image_path: str, enable_texture: bool) -> str:
    """POST /image-to-3d -> renvoie le task_id (champ 'result')."""
    body = {
        "image_url": _image_data_uri(image_path),
        "ai_model": AI_MODEL,
        "should_texture": bool(enable_texture),
        "enable_pbr": ENABLE_PBR if enable_texture else False,
        "should_remesh": True,
        "target_polycount": TARGET_POLYCOUNT,
        "target_formats": ["glb", "obj"],
    }
    r = requests.post(f"{MESHY_BASE}/image-to-3d", json=body,
                      headers=_headers(), timeout=60)
    r.raise_for_status()
    return r.json()["result"]


def _poll_task(task_id: str, job_id: str) -> dict:
    """Interroge GET /image-to-3d/{id} jusqu'à SUCCEEDED. Renvoie le JSON final."""
    deadline = time.time() + REMOTE_TIMEOUT
    while time.time() < deadline:
        r = requests.get(f"{MESHY_BASE}/image-to-3d/{task_id}",
                         headers=_headers(), timeout=30)
        r.raise_for_status()
        data = r.json()
        status = data.get("status")
        with jobs_lock:
            jobs[job_id]["stage"] = f"Meshy : {status} {data.get('progress', 0)}%"
        if status == "SUCCEEDED":
            return data
        if status in ("FAILED", "CANCELED"):
            err = (data.get("task_error") or {}).get("message") or f"Meshy {status}"
            raise RuntimeError(err)
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(f"Meshy : délai dépassé ({REMOTE_TIMEOUT}s)")


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=180, stream=True)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)


# --------------------------------------------------------------------------- #
#  Traitement d'un job (thread)
# --------------------------------------------------------------------------- #
def _process_job(job_id: str, image_path: str,
                 enable_rig: bool = False, enable_texture: bool = False):
    with jobs_lock:
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["stage"] = "Meshy : envoi de l'image…"

    out_dir = OUTPUT_DIR / job_id
    try:
        task_id = _create_task(image_path, enable_texture)
        data = _poll_task(task_id, job_id)
        urls = data.get("model_urls", {}) or {}

        ply_files: list[str] = []
        tex_files: list[str] = []

        if enable_texture and urls.get("glb"):
            # GLB texturé -> viewer GLTFLoader
            _download(urls["glb"], out_dir / "model.glb")
            tex_files = [f"/output/{job_id}/model.glb"]
        elif urls.get("obj"):
            # Géométrie -> viewer OBJLoader (via loadPLY qui aiguille sur .obj)
            _download(urls["obj"], out_dir / "model.obj")
            ply_files = [f"/output/{job_id}/model.obj"]
        elif urls.get("glb"):
            # Repli : pas d'obj fourni -> on sert le glb
            _download(urls["glb"], out_dir / "model.glb")
            tex_files = [f"/output/{job_id}/model.glb"]
        else:
            raise RuntimeError("Meshy : aucune URL de modèle dans la réponse")

        with jobs_lock:
            jobs[job_id].update({
                "status": "done",
                "stage": "terminé",
                "ply_files": ply_files,
                "glb_files": [],        # rig non supporté
                "pose_files": [],       # idem
                "tex_files": tex_files,
                "person_count": 1,
            })

    except requests.HTTPError as exc:
        detail = ""
        try:
            detail = f" — {exc.response.text[:300]}"
        except Exception:
            pass
        with jobs_lock:
            jobs[job_id].update({"status": "error",
                                 "error": f"{exc}{detail}", "ply_files": []})
    except Exception as exc:
        with jobs_lock:
            jobs[job_id].update({"status": "error", "error": str(exc),
                                 "ply_files": []})


# --------------------------------------------------------------------------- #
#  Routes (identiques à app.py)
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    ext = Path(file.filename).suffix.lower()
    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    if ext not in allowed:
        return jsonify({"error": f"Unsupported format: {ext}"}), 400

    job_id = str(uuid.uuid4())
    upload_path = UPLOAD_DIR / f"{job_id}{ext}"
    file.save(str(upload_path))

    _truthy = ("1", "true", "on", "True")
    enable_rig = request.form.get("rig", "0") in _truthy
    enable_texture = request.form.get("texture", "0") in _truthy
    if enable_texture:
        enable_rig = False

    with jobs_lock:
        jobs[job_id] = {"status": "queued", "ply_files": [], "glb_files": [],
                        "pose_files": [], "tex_files": [],
                        "person_count": 0, "error": None}

    thread = threading.Thread(
        target=_process_job,
        args=(job_id, str(upload_path), enable_rig, enable_texture),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/output/<job_id>/<filename>")
def serve_ply(job_id, filename):
    directory = OUTPUT_DIR / job_id
    return send_from_directory(str(directory), filename)


@app.route("/model_status")
def model_status():
    if not MESHY_API_KEY:
        return jsonify({"ready": False, "error": "MESHY_API_KEY non défini"})
    return jsonify({"ready": True, "error": None})


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
