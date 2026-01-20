# Setup Instructions

## Using Conda Environment (Recommended)

The project is best run with the pre-configured `tf_m1` conda environment:

```bash
# Activate the conda environment
conda activate tf_m1

# Navigate to the project
cd /Users/raviraj/Mittali_project

# Run the application
uvicorn app.main:app --reload
```

Visit: http://localhost:8000

---

## Why Python 3.14 Doesn't Work

The `.venv` environment has Python 3.14, which is too new for:
- TensorFlow 2.16.2 (requires Python ≤ 3.11)
- NumPy 1.26.4 (requires Python < 3.13)

**Solution:** Use the conda environment instead, which has Python 3.10 and all dependencies pre-installed.

---

## If You Must Use Virtual Environment

Create a new venv with Python 3.10:

```bash
# Remove old venv
rm -rf .venv

# Create new venv with Python 3.10
python3.10 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run app
uvicorn app.main:app --reload
```

---

## Verify Installation

```bash
conda activate tf_m1
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
```

Should output: `TensorFlow: 2.16.2`
