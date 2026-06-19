"""API FastAPI pour GeneMAN : image -> mesh 3D (géométrie).

Asynchrone (la reconstruction prend plusieurs minutes) :
  POST /reconstruct  (multipart: image)  -> {job_id}
  GET  /jobs/{id}                        -> {status, result_url}
  GET  /results/{id}                     -> télécharge le mesh
"""
import os, uuid, shutil, subprocess, threading, queue, traceback
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

WORK = Path("/weights/jobs/geneman")          # persisté sur le volume
WORK.mkdir(parents=True, exist_ok=True)
REPO = "/app/GeneMAN"

app = FastAPI(title="GeneMAN API")
# CORS large : POC, l'app web tourne en local sur un autre origin.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_methods=["*"], allow_headers=["*"],
)

jobs: dict[str, dict] = {}      # job_id -> {status, result, error}
job_q: "queue.Queue" = queue.Queue()


def run_geneman(image_path: str, out_dir: str) -> Path:
    """Lance la reconstruction géométrie GeneMAN et renvoie le mesh produit.

    Par défaut : wrapper run_geneman.sh (préprocess + stages 1-2, géométrie seule).
    Surchargeable sans rebuild via GENEMAN_CMD, où {image} et {out} sont remplacés.
    """
    tmpl = os.environ.get("GENEMAN_CMD", "bash /app/run_geneman.sh {image} {out}")
    cmd = tmpl.format(image=image_path, out=out_dir)
    subprocess.run(cmd, cwd=REPO, shell=True, check=True)

    meshes = (list(Path(out_dir).rglob("*.obj"))
              + list(Path(out_dir).rglob("*.glb"))
              + list(Path(out_dir).rglob("*.ply")))
    # Privilégie model.obj (le mesh final copié par le wrapper)
    meshes.sort(key=lambda p: (p.name != "model.obj", -p.stat().st_mtime))
    if not meshes:
        raise RuntimeError("Aucun mesh trouvé dans la sortie GeneMAN")
    return meshes[0]


def worker():
    while True:
        job_id, image_path, out_dir = job_q.get()
        jobs[job_id]["status"] = "running"
        try:
            mesh = run_geneman(image_path, out_dir)
            jobs[job_id].update(status="done", result=str(mesh))
        except Exception as e:
            jobs[job_id].update(status="error", error=str(e))
            traceback.print_exc()
        finally:
            job_q.task_done()


threading.Thread(target=worker, daemon=True).start()


@app.post("/reconstruct")
async def reconstruct(image: UploadFile = File(...)):
    job_id = uuid.uuid4().hex
    jdir = WORK / job_id
    out = jdir / "out"
    out.mkdir(parents=True, exist_ok=True)
    img_path = jdir / (image.filename or "input.png")
    with open(img_path, "wb") as f:
        shutil.copyfileobj(image.file, f)
    jobs[job_id] = {"status": "queued", "result": None, "error": None}
    job_q.put((job_id, str(img_path), str(out)))
    return {"job_id": job_id, "queue_position": job_q.qsize()}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    j = jobs.get(job_id)
    if not j:
        raise HTTPException(404, "job inconnu")
    return {
        "status": j["status"],
        "error": j["error"],
        "result_url": f"/results/{job_id}" if j["status"] == "done" else None,
    }


@app.get("/results/{job_id}")
def result(job_id: str):
    j = jobs.get(job_id)
    if not j or j["status"] != "done":
        raise HTTPException(404, "résultat indisponible")
    return FileResponse(j["result"], filename=Path(j["result"]).name)


@app.get("/health")
def health():
    return {"ok": True, "queue": job_q.qsize()}
