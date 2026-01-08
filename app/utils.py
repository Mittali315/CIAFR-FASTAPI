"""Tiny helpers used by the web app and worker.

This module purposely stays small and clear for newcomers. It avoids
complex typing and tries to work even when heavy dependencies (TensorFlow
or OpenCV) are not available.
"""
import os
import numpy as np
from PIL import Image

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    tf = None
    TF_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    cv2 = None
    CV2_AVAILABLE = False


CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def load_model(path: str):
    """Load a TensorFlow model if available, otherwise return a dummy.

    The dummy model implements a `predict` method that returns a fixed
    one-hot vector so the rest of the code can run without TF.
    """
    if TF_AVAILABLE:
        return tf.keras.models.load_model(path)

    class Dummy:
        def predict(self, x):
            out = np.zeros((1, 10), dtype=float)
            out[0, 0] = 1.0
            return out

    return Dummy()


def preprocess_image(image: Image.Image):
    """Resize to 32x32 and normalize to [0, 1]."""
    image = image.resize((32, 32))
    arr = np.array(image).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


def save_gradcam(img: Image.Image, heatmap, out_path: str):
    """Overlay `heatmap` onto `img` and save to `out_path`.

    - Normalizes heatmap to [0,1] and handles NaN/Inf.
    - Uses OpenCV colormap if available, otherwise PIL fallback.
    - Ensures output directory exists and returns `out_path`.
    """
    # ensure output directory exists
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    # prepare heatmap as float array and normalize
    hm = np.array(heatmap, dtype=float)
    hm = np.nan_to_num(hm, nan=0.0, posinf=0.0, neginf=0.0)
    hmin, hmax = float(np.min(hm)), float(np.max(hm))
    if hmax - hmin < 1e-8:
        norm = np.zeros_like(hm, dtype="float32")
    else:
        norm = (hm - hmin) / (hmax - hmin)

    norm8 = np.uint8(norm * 255)
    img_np = np.array(img.convert("RGB"))

    # try OpenCV path first for better color mapping
    if CV2_AVAILABLE and cv2 is not None:
        try:
            cm = cv2.COLORMAP_JET
            heat_col = cv2.applyColorMap(norm8, cm)
            # OpenCV returns BGR; convert to RGB
            heat_col = cv2.cvtColor(heat_col, cv2.COLOR_BGR2RGB)
            heat_resized = cv2.resize(heat_col, (img_np.shape[1], img_np.shape[0]))
            alpha = 0.4
            out = np.clip((1 - alpha) * img_np.astype("float32") + alpha * heat_resized.astype("float32"), 0, 255).astype(
                "uint8"
            )
            Image.fromarray(out).save(out_path)
            return out_path
        except Exception:
            # fall through to PIL fallback
            pass

    # PIL fallback: resize grayscale heatmap then stack and blend
    heat_img = Image.fromarray(norm8).resize((img_np.shape[1], img_np.shape[0]), Image.BILINEAR)
    heat_np = np.array(heat_img)
    heat_rgb = np.stack([heat_np] * 3, axis=2)
    alpha = 0.4
    out = np.clip((1 - alpha) * img_np.astype("float32") + alpha * heat_rgb.astype("float32"), 0, 255).astype(
        "uint8"
    )
    Image.fromarray(out).save(out_path)
    return out_path


def load_model(path: str):
    """Load TF model if available; otherwise return a dummy with predict()."""
    if TF_AVAILABLE:
        return tf.keras.models.load_model(path)

    class Dummy:
        def predict(self, x):
            out = np.zeros((1, 10), dtype=float)
            out[0, 0] = 1.0
            return out

    return Dummy()


def make_gradcam_heatmap(img_array, model, last_conv_layer_name: str = "conv2d_14"):
    """Compute Grad-CAM heatmap; return zero map if TF or layers not available.

    This helper is defensive: it attempts to call the model first so layer
    outputs exist, tries to find a conv layer, and computes gradients.
    """
    if not TF_AVAILABLE:
        return np.zeros((32, 32), dtype=float)

    try:
        # ensure model built / called
        _ = model(tf.convert_to_tensor(img_array, dtype=tf.float32), training=False)
    except Exception:
        try:
            _ = model.predict(img_array)
        except Exception:
            pass

    last_layer = None
    try:
        last_layer = model.get_layer(last_conv_layer_name)
    except Exception:
        for layer in reversed(getattr(model, "layers", [])):
            if "conv" in layer.__class__.__name__.lower():
                last_layer = layer
                break

    if last_layer is None:
        return np.zeros((32, 32), dtype=float)

    try:
        grad_model = tf.keras.models.Model([model.inputs], [last_layer.output, model.output])
    except Exception:
        return np.zeros((32, 32), dtype=float)

    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


