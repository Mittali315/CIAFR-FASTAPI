import os
import numpy as np

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    tf = None
    TF_AVAILABLE = False

from .utils import CLASS_NAMES, preprocess_image

_model = None  # lazy-load global model


def get_model():
    """Load CIFAR-10 TensorFlow model lazily."""
    global _model
    if _model is None:
        if TF_AVAILABLE:
            _model = tf.keras.models.load_model("models/cifar10_cnn_model.keras")
        else:
            class Dummy:
                def predict(self, x):
                    out = np.zeros((1, 10))
                    out[0, 0] = 1.0
                    return out
            _model = Dummy()
    return _model


def predict_image(image) -> tuple[str, float]:
    """Return predicted class name and confidence."""
    model = get_model()
    img_array = preprocess_image(image)
    preds = model.predict(img_array)
    pred_class = CLASS_NAMES[np.argmax(preds)]
    confidence = float(np.max(preds))
    return pred_class, round(confidence, 4)
