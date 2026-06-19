#!/usr/bin/env python3
"""
MyTwin Avatar — Web app for 3D body reconstruction from image.
Run: python app.py

Version "proxy" : l'app ne charge PLUS de modèle en local. Elle relaie les
requêtes vers les deux API FastAPI distantes (sur l'instance GPU Scaleway) :

  - GeneMAN  (POST /reconstruct)  : image           -> mesh géométrie  (SANS texture)
  - UniTEX   (POST /texture)      : image + mesh     -> mesh texturé    (AVEC texture)

L'interface graphique (templates/index.html) est inchangée : ce fichier conserve
exactement les mêmes routes et la même forme de réponse JSON qu'avant.

Mapping avec l'UI existante :
  - case "texture" cochée   -> GeneMAN puis UniTEX (ply_files + tex_files)
  - case "texture" décochée -> GeneMAN seul        (ply_files)
  - case "rig"              -> non supportée par ce pipeline (renvoie vide)

Configuration (variables d'environnement) :
  GENEMAN_API_URL   (def. http://127.0.0.1:8001)
  UNITEX_API_URL    (def. http://127.0.0.1:8002)
  POLL_INTERVAL     (def. 3   secondes entre deux vérifs de statut distant)
  REMOTE_TIMEOUT    (def. 7200 secondes max par job distant — GeneMAN est lent)
"""

import os
import uuid
import time
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

# --- Endpoints des API distantes ---
GENEMAN_API = os.environ.get("GENEMAN_API_URL", "http://127.0.0.1:8001").rstrip("/")
UNITEX_API  = os.environ.get("UNITEX_API_URL",  "http://127.0.0.1:8002").rstrip("/")
POLL_INTERVAL  = float(os.environ.get("POLL_INTERVAL", "3"))
REMOTE_TIMEOUT = float(os.environ.get("REMOTE_TIMEOUT", "7200"))

jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


# --------------------------------------------------------------------------- #
#  Helpers d'appel aux API distantes
# --------------------------------------------------------------------------- #
def _poll_remote(base: str, job_id: str) -> str:
    """Interroge GET /jobs/{id} jusqu'à done/error. Renvoie le result_url distant."""
    deadline = time.time() + REMOTE_TIMEOUT
    while time.time() < deadline:
        r = requests.get(f"{base}/jobs/{job_id}", timeout=30)
        r.raise_for_status()
        data = r.json()
        if data["status"] == "done":
            return data["result_url"]
        if data["status"] == "error":
            raise RuntimeError(data.get("error") or "job distant en erreur")
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(f"délai dépassé ({REMOTE_TIMEOUT}s) sur {base}")


def _download(base: str, result_url: str, dest_dir: Path, fallback_name: str) -> Path:
    """Télécharge le mesh résultat depuis l'API distante vers dest_dir."""
    r = requests.get(f"{base}{result_url}", timeout=120, stream=True)
    r.raise_for_status()
    # Nom de fichier renvoyé par l'API (Content-Disposition), sinon fallback.
    name = fallback_name
    cd = r.headers.get("content-disposition", "")
    if "filename=" in cd:
        name = cd.split("filename=")[-1].strip('"; ')
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / name
    with open(out, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    return out


def _run_geneman(image_path: str, dest_dir: Path) -> Path:
    """POST image -> GeneMAN, attend, télécharge le mesh géométrie."""
    with open(image_path, "rb") as f:
        r = requests.post(f"{GENEMAN_API}/reconstruct",
                          files={"image": f}, timeout=120)
    r.raise_for_status()
    remote_id = r.json()["job_id"]
    result_url = _poll_remote(GENEMAN_API, remote_id)
    return _download(GENEMAN_API, result_url, dest_dir, "model.obj")


def _run_unitex(image_path: str, mesh_path: Path, dest_dir: Path) -> Path:
    """POST image+mesh -> UniTEX, attend, télécharge le mesh texturé."""
    with open(image_path, "rb") as fi, open(mesh_path, "rb") as fm:
        r = requests.post(f"{UNITEX_API}/texture",
                          files={"image": fi, "mesh": fm}, timeout=120)
    r.raise_for_status()
    remote_id = r.json()["job_id"]
    result_url = _poll_remote(UNITEX_API, remote_id)
    return _download(UNITEX_API, result_url, dest_dir, "textured.glb")


# --------------------------------------------------------------------------- #
#  Traitement d'un job (thread)
# --------------------------------------------------------------------------- #
def _process_job(job_id: str, image_path: str, base_name: str,
                 enable_rig: bool = False, enable_texture: bool = False):
    with jobs_lock:
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["stage"] = "GeneMAN : reconstruction…"

    job_out_dir = OUTPUT_DIR / job_id
    try:
        # 1) GeneMAN -> mesh SANS texture
        mesh = _run_geneman(image_path, job_out_dir)
        ply_files = [f"/output/{job_id}/{mesh.name}"]

        tex_files: list[str] = []
        # 2) UniTEX (optionnel) -> mesh AVEC texture
        if enable_texture:
            with jobs_lock:
                jobs[job_id]["stage"] = "UniTEX : texturing…"
            tex_mesh = _run_unitex(image_path, mesh, job_out_dir)
            tex_files = [f"/output/{job_id}/{tex_mesh.name}"]

        with jobs_lock:
            jobs[job_id].update({
                "status": "done",
                "stage": "terminé",
                "ply_files": ply_files,
                "glb_files": [],        # rig non supporté par ce pipeline
                "pose_files": [],       # idem
                "tex_files": tex_files,
                "person_count": 1,      # GeneMAN = une personne par image
            })

    except Exception as exc:
        with jobs_lock:
            jobs[job_id].update({"status": "error", "error": str(exc),
                                 "ply_files": []})


# --------------------------------------------------------------------------- #
#  Routes (identiques à l'ancienne version)
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

    base_name = Path(file.filename).stem
    _truthy = ("1", "true", "on", "True")
    enable_rig = request.form.get("rig", "0") in _truthy
    enable_texture = request.form.get("texture", "0") in _truthy
    # Mutuellement exclusifs (comme avant) : la texture prime.
    if enable_texture:
        enable_rig = False

    with jobs_lock:
        jobs[job_id] = {"status": "queued", "ply_files": [], "glb_files": [],
                        "pose_files": [], "tex_files": [],
                        "person_count": 0, "error": None}

    thread = threading.Thread(
        target=_process_job,
        args=(job_id, str(upload_path), base_name, enable_rig, enable_texture),
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
    """Vérifie que les deux API distantes répondent (remplace le check du modèle local)."""
    errors = []
    for name, base in (("GeneMAN", GENEMAN_API), ("UniTEX", UNITEX_API)):
        try:
            requests.get(f"{base}/health", timeout=5).raise_for_status()
        except Exception as exc:
            errors.append(f"{name} injoignable ({base}) : {exc}")
    if errors:
        return jsonify({"ready": False, "error": " | ".join(errors)})
    return jsonify({"ready": True, "error": None})


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
