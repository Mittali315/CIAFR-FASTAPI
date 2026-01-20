from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.concurrency import run_in_threadpool
from PIL import Image
import io

from app.utils import make_gradcam_heatmap, save_gradcam
from app.predictor import predict_image, get_model, preprocess_image

app = FastAPI(title="CIFAR-10 Classifier")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    # prediction
    pred_class, confidence = await run_in_threadpool(lambda: predict_image(img))

    # Grad-CAM
    model = await run_in_threadpool(get_model)
    heatmap = await run_in_threadpool(lambda: make_gradcam_heatmap(preprocess_image(img), model))
    await run_in_threadpool(lambda: save_gradcam(img, heatmap))

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": pred_class,
            "confidence": confidence,
            "gradcam": "/static/gradcam.png"
        }
    )
