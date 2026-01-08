import os
import sys
import io
import time
import numpy as np
from PIL import Image
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
from app.utils import preprocess_image, CLASS_NAMES, load_model, make_gradcam_heatmap, save_gradcam

try:
    import cv2
    _has_cv2 = True
except Exception:
    cv2 = None
    _has_cv2 = False

# ---------------- Configuration ----------------
SKIP_MODEL = os.environ.get("DISABLE_TF", "1") != "0"
# worker script lives at repo root next to this package
WORKER_PY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "predictor_worker.py"))
WORKER_PYTHON = os.environ.get("WORKER_PYTHON", sys.executable)
"""Simple FastAPI app for CIFAR-10 demo.

This file is written to be easy to read for beginners. It supports two
modes:
- run the model in the web process (if TensorFlow is available), or
- delegate prediction to the worker script (a separate process).

The web UI is a minimal HTML page that posts an image to `/predict`.
"""

import io
import os
import sys
import time
import json
from typing import Tuple

import numpy as np
from PIL import Image
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool

# Path to the worker script (keeps worker next to package)
WORKER_PY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "predictor_worker.py"))
WORKER_PYTHON = os.environ.get("WORKER_PYTHON", sys.executable)

# When DISABLE_TF != 0 the app will use the worker process instead of
# trying to import TensorFlow inside the web server. This is safer when TF
# is not installed in the web environment.
SKIP_MODEL = os.environ.get("DISABLE_TF", "1") != "0"



app = FastAPI(title="CIFAR-10 Classifier")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


# `preprocess_image` and `CLASS_NAMES` are provided by `app.utils`.


def extract_json_from_stdout(stdout: str) -> dict | None:
    """Try to find the last JSON object in a string and parse it.

    Some programs (TensorFlow) print logs before printing JSON. This
    helper looks for the final {...} and parses it.
    """
    s = stdout.strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        # fallback: find last { ... }
        i = s.rfind("{")
        j = s.rfind("}")
        if i != -1 and j != -1 and j > i:
            try:
                return json.loads(s[i : j + 1])
            except Exception:
                return None
    return None


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    """Handle file upload, run prediction, and return the HTML page.

    This function keeps the logic simple: it either runs a small worker
    script as a subprocess (recommended when TF is not installed) or
    calls the local model if available.
    """
    contents = await file.read()

    # Try opening the image; return a friendly error page on failure.
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "error": f"Bad image: {e}"})

    img_array = preprocess_image(img)
    timestamp = int(time.time() * 1000)
    gradcam_filename = f"gradcam_{timestamp}.png"
    gradcam_path = os.path.join("app/static", gradcam_filename)

    prediction = "prediction_failed"
    confidence = 0.0

    if SKIP_MODEL:
        # Run the worker script in a subprocess to avoid loading TF in the web process.
        import tempfile
        import subprocess

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tfp:
            tfp.write(contents)
            tmp_input = tfp.name

        cmd = [WORKER_PYTHON, WORKER_PY, "--input", tmp_input, "--output", gradcam_path]
        try:
            proc = await run_in_threadpool(lambda: subprocess.run(cmd, capture_output=True, text=True, timeout=120))
            # save debug output for inspection by developer
            try:
                with open("app/static/worker_debug.txt", "w") as f:
                    f.write("--- STDOUT ---\n")
                    f.write((proc.stdout or "") + "\n")
                    f.write("--- STDERR ---\n")
                    f.write((proc.stderr or "") + "\n")
            except Exception:
                pass

            if proc.returncode == 0:
                obj = extract_json_from_stdout(proc.stdout or "")
                if obj:
                    prediction = obj.get("prediction", "prediction_failed")
                    confidence = obj.get("confidence", 0.0)
        except Exception:
            prediction = "prediction_failed"
            confidence = 0.0
        finally:
            try:
                os.remove(tmp_input)
            except Exception:
                pass
    else:
        # In-process prediction (requires TF installed in this environment).
        # Keep this simple: load model lazily and run predict.
        import tensorflow as tf

        model = await run_in_threadpool(lambda: load_model("models/cifar10_cnn_model.keras"))
        preds = await run_in_threadpool(lambda: model.predict(img_array))
        prediction = CLASS_NAMES[int(np.argmax(preds))]
        confidence = float(np.max(preds))
        # generate grad-cam image (kept simple here)
        try:
            def make_and_save():
                # use shared helpers from app.utils
                from app.utils import make_gradcam_heatmap, save_gradcam

                heatmap = make_gradcam_heatmap(img_array, model)
                save_gradcam(img, heatmap, gradcam_path)

            await run_in_threadpool(make_and_save)
        except Exception:
            pass

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": prediction,
            "confidence": confidence,
            "gradcam": f"/static/{gradcam_filename}",
        },
    )
