import os
import numpy as np
from PIL import Image

# Optional dependencies
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
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


# ==================== Model Loader ====================

def load_model(path: str):
    """Load TensorFlow model if available, else dummy model."""
    if TF_AVAILABLE:
        return tf.keras.models.load_model(path)

    class Dummy:
        def predict(self, x):
            out = np.zeros((1, 10))
            out[0, 0] = 1.0
            return out

    return Dummy()


# ==================== Image Preprocessing ====================

def preprocess_image(image: Image.Image):
    """Resize to 32x32 and normalize to [0,1]."""
    image = image.resize((32, 32))
    arr = np.array(image).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


# ==================== Grad-CAM ====================

def make_gradcam_heatmap(img_array, model):
    """Generate Grad-CAM heatmap safely."""
    if not TF_AVAILABLE:
        return np.zeros((32, 32), dtype=np.float32)

    # 🔥 Force model to build
    try:
        _ = model(img_array, training=False)
    except Exception:
        try:
            _ = model.predict(img_array)
        except Exception:
            return np.zeros((32, 32), dtype=np.float32)

    # Find last conv layer automatically
    last_conv_layer = None
    for layer in reversed(model.layers):
        if "conv" in layer.__class__.__name__.lower():
            last_conv_layer = layer
            break

    if last_conv_layer is None:
        return np.zeros((32, 32), dtype=np.float32)

    try:
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[last_conv_layer.output, model.output]
        )
    except Exception:
        return np.zeros((32, 32), dtype=np.float32)

    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    if grads is None:
        return np.zeros((32, 32), dtype=np.float32)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)

    if max_val == 0:
        return np.zeros((32, 32), dtype=np.float32)

    heatmap /= max_val
    return heatmap.numpy()


# ==================== Save Grad-CAM ====================

def save_gradcam(img: Image.Image, heatmap, out_path= "app/static/gradcam.png"):
    """Overlay heatmap on image and save."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    img_np = np.array(img.convert("RGB"))
    heatmap = np.nan_to_num(heatmap)

    # Normalize
    hmin, hmax = heatmap.min(), heatmap.max()
    if hmax - hmin > 1e-8:
        heatmap = (heatmap - hmin) / (hmax - hmin)
    else:
        heatmap = np.zeros_like(heatmap)

    heatmap_uint8 = np.uint8(255 * heatmap)

    # OpenCV path
    if CV2_AVAILABLE and cv2 is not None:
        try:
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
            heatmap_color = cv2.resize(
                heatmap_color, (img_np.shape[1], img_np.shape[0])
            )

            overlay = np.clip(
                img_np * 0.6 + heatmap_color * 0.4, 0, 255
            ).astype("uint8")

            Image.fromarray(overlay).save(out_path)
            return out_path
        except Exception:
            pass

    # PIL fallback
    heatmap_img = Image.fromarray(heatmap_uint8).resize(
        (img_np.shape[1], img_np.shape[0]), Image.BILINEAR
    )
    heatmap_np = np.array(heatmap_img)
    heatmap_rgb = np.stack([heatmap_np] * 3, axis=2)

    overlay = np.clip(
        img_np * 0.6 + heatmap_rgb * 0.4, 0, 255
    ).astype("uint8")

    Image.fromarray(overlay).save(out_path)
    return out_path
