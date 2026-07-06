#!/usr/bin/env python3
"""
MyTwin Avatar — Frontend (Scalingo).

Sert uniquement la page et les assets client. Aucune logique métier, aucun
secret : le navigateur orchestre le parcours et appelle le backend (API_BASE)
directement. Les avatars sont stockés sur l'appareil (IndexedDB), pas ici.

Variables :
  API_BASE   URL du backend Scaleway (ex. https://xxx.functions.fnc.fr-par.scw.cloud).
             Vide = même origine (mode proxy dev, ci-dessous).
  PROXY_API  DEV UNIQUEMENT : si défini, le frontend relaie /body //body/status /graft
             //healthz vers ce backend -> le navigateur ne voit qu'une seule origine
             (un seul certificat en HTTPS local, pas de galère CORS/cert cross-origin).
             En prod, laisser vide et utiliser API_BASE (CORS direct).
"""
import os
from urllib.parse import urlparse
from pathlib import Path

from flask import Flask, render_template, send_from_directory, request, Response
from werkzeug.middleware.proxy_fix import ProxyFix

API_BASE = os.environ.get("API_BASE", "").rstrip("/")
PROXY_API = os.environ.get("PROXY_API", "").rstrip("/")
HERE = Path(__file__).resolve().parent

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 80 * 1024 * 1024   # laisse passer les uploads du proxy
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)


def _forward(subpath: str):
    """Relaie la requête courante vers PROXY_API (dev same-origin). Streame la réponse
    (le /graft renvoie ~60 Mo). Backend local en HTTPS auto-signé -> verify=False."""
    import requests
    files = []
    for key in request.files:
        for f in request.files.getlist(key):
            files.append((key, (f.filename, f.stream, f.mimetype)))
    r = requests.request(request.method, PROXY_API + subpath,
                         params=request.args,
                         data=request.form.to_dict(flat=True) or None,
                         files=files or None, stream=True, verify=False, timeout=600)
    skip = {"content-encoding", "transfer-encoding", "connection", "content-length"}
    headers = [(k, v) for k, v in r.headers.items() if k.lower() not in skip]
    return Response(r.iter_content(1 << 16), status=r.status_code, headers=headers)


if PROXY_API:
    app.add_url_rule("/body", "px_body", lambda: _forward("/body"), methods=["POST"])
    app.add_url_rule("/graft", "px_graft", lambda: _forward("/graft"), methods=["POST"])
    app.add_url_rule("/body/status", "px_status", lambda: _forward("/body/status"))


def _api_origin() -> str:
    """Origine (scheme://host[:port]) du backend, pour l'ajouter à la CSP connect-src."""
    if not API_BASE:
        return ""
    p = urlparse(API_BASE)
    return f"{p.scheme}://{p.netloc}" if p.scheme and p.netloc else ""


@app.route("/")
def index():
    return render_template("index.html", api_base=API_BASE)


@app.route("/models/face_landmarker.task")
def face_landmarker_model():
    """Modèle MediaPipe FaceLandmarker pour la capture visage côté navigateur."""
    return send_from_directory(HERE / "models", "face_landmarker.task",
                               mimetype="application/octet-stream", max_age=86400)


@app.route("/healthz")
def healthz():
    if PROXY_API:
        return _forward("/healthz")
    return {"ok": True, "api_base": API_BASE}


@app.after_request
def security_headers(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "SAMEORIGIN"
    resp.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    resp.headers["Permissions-Policy"] = "camera=(self), geolocation=()"
    api = _api_origin()
    # connect-src doit inclure le backend (fetch API), le CDN MediaPipe (WASM) et
    # blob:/data: (model-viewer + object URLs de la galerie).
    connect = "'self' blob: data: https://cdn.jsdelivr.net https://www.gstatic.com"
    if api:
        connect += f" {api}"
    resp.headers["Content-Security-Policy"] = (
        "default-src 'self'; base-uri 'self'; form-action 'self'; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval' 'wasm-unsafe-eval' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com data:; "
        "img-src 'self' data: blob:; "
        f"connect-src {connect}; "
        "worker-src 'self' blob:; frame-ancestors 'self'"
    )
    return resp


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
