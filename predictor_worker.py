"""Worker script: load model, predict, and save Grad-CAM image.

This small script is intended to be run as a separate process by the
web server. Keeping the heavy TensorFlow work in a worker prevents the
web process from needing TF installed.
"""

import argparse
import json
import numpy as np
from PIL import Image
from app.utils import preprocess_image, make_gradcam_heatmap, save_gradcam, load_model, CLASS_NAMES


# All helpers (preprocess_image, make_gradcam_heatmap, save_gradcam, load_model, CLASS_NAMES)
# are provided by `app.utils` and imported at the top of this file. This
# keeps the worker small and re-uses shared code.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR-10 prediction worker")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--output", required=True, help="Path to save Grad-CAM image")
    args = parser.parse_args()

    try:
        img = Image.open(args.input).convert("RGB")
        img_array = preprocess_image(img)

        model = load_model("models/cifar10_cnn_model.keras")

        preds = model.predict(img_array)
        predicted_class = CLASS_NAMES[int(np.argmax(preds))]
        confidence = float(np.max(preds))

        heatmap = make_gradcam_heatmap(img_array, model)
        save_gradcam(img, heatmap, args.output)

        print(json.dumps({"prediction": predicted_class, "confidence": confidence}))
    except Exception as e:
        print(json.dumps({"prediction": "prediction_failed", "confidence": 0.0, "error": str(e)}))
