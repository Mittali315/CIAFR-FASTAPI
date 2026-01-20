# Building a CIFAR-10 Image Classifier with FastAPI and TensorFlow: A Deep Dive into Model Deployment

## Introduction

Building machine learning models is exciting, but deploying them into production is where the real challenges begin. In this article, I'll walk you through how I built a **CIFAR-10 image classification web application** using FastAPI and TensorFlow, the bugs I encountered, and how I fixed them.

Whether you're a beginner looking to learn ML deployment or an experienced developer curious about production ML pipelines, this article will provide practical insights.

---

## The Problem: Always Predicting "Airplane"

When I first deployed my CIFAR-10 classifier, something was terribly wrong. **Every single image was being classified as "airplane"** — whether I uploaded a cat, dog, ship, or truck.

At first, I thought the model was broken. But after investigation, I discovered the real culprit: **TensorFlow wasn't installed**.

Here's what was happening:

1. The app had a fallback "dummy model" for when TensorFlow isn't available
2. This dummy model always returned class 0 in the one-hot vector
3. Class 0 in CIFAR-10 is "airplane"
4. Result: Everything predicted as airplane 🛩️

This taught me an important lesson: **always verify your dependencies are installed before assuming model issues**.

---

## Project Architecture

Let me explain the architecture of the solution:

```
FastAPI Web Server
    ↓
File Upload Handler
    ↓
Prediction Worker (subprocess)
    ↓
TensorFlow Model
    ↓
Grad-CAM Visualization
    ↓
HTML Response
```

### Key Components:

**1. FastAPI Main App** (`app/main.py`)
- Handles HTTP requests
- Manages file uploads
- Spawns worker processes for predictions

**2. Worker Script** (`predictor_worker.py`)
- Runs predictions in a separate process
- Prevents blocking the web server
- Returns JSON predictions

**3. Utilities** (`app/utils.py`)
- Image preprocessing (resize to 32x32, normalize)
- Model loading (with fallback dummy model)
- Grad-CAM heatmap generation
- Visualization saving

**4. Frontend** (HTML/JS)
- Image upload interface
- Real-time prediction display
- Grad-CAM visualization

---

## The Bugs and Fixes

### Bug #1: TensorFlow Not Installed

**Symptom:** Always predicting "airplane"

**Root Cause:**
```python
def load_model(path: str):
    if TF_AVAILABLE:
        return tf.keras.models.load_model(path)
    
    class Dummy:
        def predict(self, x):
            out = np.zeros((1, 10), dtype=float)
            out[0, 0] = 1.0  # Always returns class 0 (airplane!)
            return out
    
    return Dummy()
```

**Solution:** Install TensorFlow for macOS with M1 chip:
```bash
pip install tensorflow-macos==2.16.2
```

### Bug #2: Dependency Conflicts

**Symptom:** Pip installation failed with conflicting requirements

**Root Cause:** 
- TensorFlow 2.16.2 requires: `numpy < 2.0`
- opencv-python-headless 4.12 requires: `numpy >= 2.0`

**Solution:** Updated `requirements.txt`:
```txt
numpy>=1.23.5,<2.0      # Compatible with TensorFlow
opencv-python==4.8.1.78 # Switched from opencv-python-headless
tensorflow-macos==2.16.2
```

### Bug #3: Missing Model File

**Symptom:** `FileNotFoundError: models/cifar10_cnn_model.keras`

**Solution:** 
```bash
mkdir -p models
cp ciafr-fastapi/models/cifar10_cnn_model.keras models/
```

---

## Technical Implementation

### Image Preprocessing

```python
def preprocess_image(image: Image.Image):
    """Resize to 32x32 and normalize to [0, 1]."""
    image = image.resize((32, 32))
    arr = np.array(image).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)  # Add batch dimension
```

### Model Prediction

```python
model = load_model("models/cifar10_cnn_model.keras")
preds = model.predict(img_array)
predicted_class = CLASS_NAMES[int(np.argmax(preds))]
confidence = float(np.max(preds))
```

### Grad-CAM Visualization

Grad-CAM (Gradient-weighted Class Activation Maps) shows which regions of the image influenced the prediction:

```python
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv2d_14"):
    # Compute gradients with respect to last conv layer
    # Create heatmap overlay
    # Return normalized heatmap
```

---

## Setup and Installation

### Prerequisites
- Python 3.10+
- macOS (or Linux/Windows with appropriate TensorFlow version)

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/Mittali315/CIAFR-FASTAPI.git
cd Mittali_project

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
uvicorn app.main:app --reload
```

### Access the App
Open your browser and go to: `http://localhost:8000`

---

## How It Works: Step by Step

1. **User uploads an image** via the web interface
2. **FastAPI receives the file** and saves it temporarily
3. **Worker process spawns** to run predictions (prevents web server blocking)
4. **Image is preprocessed**: resized to 32×32, normalized
5. **TensorFlow model predicts** the class and confidence
6. **Grad-CAM generates** a heatmap showing important regions
7. **Results returned** as HTML with prediction and visualization

---

## CIFAR-10 Classes

The model can classify images into 10 categories:

```
0: Airplane       5: Dog
1: Automobile     6: Frog
2: Bird           7: Horse
3: Cat            8: Ship
4: Deer           9: Truck
```

---

## Key Learnings

### 1. **Dependency Management is Critical**
When working with ML libraries, version compatibility is crucial. Always test your `requirements.txt` in a fresh environment.

### 2. **Graceful Fallbacks Matter**
The dummy model ensures the app doesn't crash if TensorFlow isn't available, but it should be clearly documented or logged.

### 3. **Separate Heavy Operations**
Running ML inference in a worker process prevents the web server from blocking, improving user experience.

### 4. **Explainability Matters**
Grad-CAM visualizations help users understand why the model made a prediction, building trust in the system.

### 5. **Device-Specific Dependencies**
Mac M1 requires `tensorflow-macos` instead of regular `tensorflow`. Always check for platform-specific packages.

---

## Deployment Considerations

For production deployment:

```bash
# Use Gunicorn instead of development server
pip install gunicorn
gunicorn app.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker

# Containerize with Docker
docker build -t cifar10-classifier .
docker run -p 8000:8000 cifar10-classifier
```

---

## GitHub Repository

Full code available here: **https://github.com/Mittali315/CIAFR-FASTAPI.git**

---

## Future Improvements

- [ ] Add batch prediction endpoint
- [ ] Implement model versioning
- [ ] Add prediction caching
- [ ] Collect user feedback for model improvement
- [ ] Deploy to cloud (AWS/GCP/Azure)
- [ ] Add API authentication
- [ ] Implement prediction logging and monitoring
- [ ] Add confidence threshold alerts

---

## Conclusion

Building a production ML application involves more than just training a model. You need to handle:
- ✅ Dependency management
- ✅ Process architecture  
- ✅ Error handling
- ✅ User-friendly interfaces
- ✅ Explainability

This project demonstrates a complete end-to-end ML deployment pipeline. The bugs I encountered — and more importantly, how I fixed them — are typical challenges you'll face in real-world ML engineering.

If you're getting started with ML deployment, I recommend:
1. Always verify your dependencies are installed
2. Test edge cases (missing files, wrong formats)
3. Add logging for debugging
4. Include visualization to explain model decisions

**Happy deploying! 🚀**

---

## About the Author

I'm a software engineer passionate about machine learning and building scalable applications. Follow me for more articles on ML engineering, FastAPI, and Python development.

**Connect:**
- GitHub: [@Mittali315](https://github.com/Mittali315)
- LinkedIn: [Your LinkedIn]

---

*Have you deployed ML models in production? Share your challenges and solutions in the comments!*
