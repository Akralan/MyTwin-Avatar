"""API FastAPI pour UniTEX : (image + mesh) -> mesh texturé.

Le pipeline UniTEX est chargé UNE SEULE FOIS au démarrage (poids gardés en
mémoire GPU) puis réutilisé pour chaque job -> pas de rechargement par requête.

  POST /texture  (multipart: image, mesh)  -> {job_id}
  GET  /jobs/{id}                          -> {status, result_url}
  GET  /results/{id}                       -> télécharge le mesh texturé
"""
import uuid, shutil, threading, queue, traceback
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# Import du vrai pipeline du repo (api.py vit dans /app/UniTEX)
from pipeline import CustomRGBTextureFullPipeline

WORK = Path("/weights/jobs/unitex")
WORK.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="UniTEX API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_methods=["*"], allow_headers=["*"],
)

jobs: dict[str, dict] = {}
job_q: "queue.Queue" = queue.Queue()
PIPELINE = None          # chargé paresseusement dans le worker (1ʳᵉ requête)


def get_pipeline():
    global PIPELINE
    if PIPELINE is None:
        # Mêmes options que run.py du repo ; télécharge les checkpoints au 1er appel.
        PIPELINE = CustomRGBTextureFullPipeline(
            super_resolutions=False,
            filt_gradient_points=False,
            filt_large_angle_points=True,
            seed=63,
        )
    return PIPELINE


def run_unitex(image_path: str, mesh_path: str, out_dir: str) -> Path:
    pipe = get_pipeline()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # Signature réelle du repo : (save_root, image, mesh, clear_cache)
    pipe(out_dir, image_path, mesh_path, clear_cache=False)
    meshes = (list(Path(out_dir).rglob("*.glb"))
              + list(Path(out_dir).rglob("*.obj")))
    if not meshes:
        raise RuntimeError("Aucun mesh texturé trouvé dans la sortie UniTEX")
    return max(meshes, key=lambda p: p.stat().st_mtime)


def worker():
    while True:
        job_id, image_path, mesh_path, out_dir = job_q.get()
        jobs[job_id]["status"] = "running"
        try:
            mesh = run_unitex(image_path, mesh_path, out_dir)
            jobs[job_id].update(status="done", result=str(mesh))
        except Exception as e:
            jobs[job_id].update(status="error", error=str(e))
            traceback.print_exc()
        finally:
            job_q.task_done()


threading.Thread(target=worker, daemon=True).start()


@app.post("/texture")
async def texture(image: UploadFile = File(...), mesh: UploadFile = File(...)):
    job_id = uuid.uuid4().hex
    jdir = WORK / job_id
    out = jdir / "out"
    out.mkdir(parents=True, exist_ok=True)
    img_path = jdir / (image.filename or "input.png")
    mesh_path = jdir / (mesh.filename or "input.obj")
    for up, dst in ((image, img_path), (mesh, mesh_path)):
        with open(dst, "wb") as f:
            shutil.copyfileobj(up.file, f)
    jobs[job_id] = {"status": "queued", "result": None, "error": None}
    job_q.put((job_id, str(img_path), str(mesh_path), str(out)))
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
