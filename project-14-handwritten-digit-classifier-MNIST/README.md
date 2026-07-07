# Project 14 — Handwritten Digit Classifier (MNIST)

A complete digit recognition system: trains a neural network on MNIST, saves the
trained model, loads custom handwritten images, runs inference from a script or
notebook, and ships a Streamlit web app for interactive predictions.

## Project Structure

```
project-14-mnist-digit/
├── app/
│   └── streamlit_app.py        # Interactive web app (draw or upload a digit)
├── data/
│   └── custom_digits/          # Put your own handwritten digit images here
├── models/
│   └── mnist_digit_model.h5    # Saved trained model (created after training)
├── notebooks/
│   └── mnist_digit_classifier.ipynb   # Full walkthrough: train, evaluate, predict
├── reports/
│   ├── training_history.png    # Accuracy/loss curves (created after training)
│   └── metrics.json            # Test accuracy/loss (created after training)
├── src/
│   ├── __init__.py
│   ├── config.py                # Shared paths & hyperparameters
│   ├── model.py                 # Model architecture
│   ├── preprocessing.py         # Shared image preprocessing (used everywhere)
│   ├── train.py                 # Training script
│   └── predict.py               # CLI inference script
├── tests/
│   └── test_preprocessing.py    # Basic sanity tests
├── requirements.txt
├── .gitignore
└── README.md
```

## 1. Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Train the model

```bash
python -m src.train
```

This will:
- Download and normalize MNIST
- Train a fully connected neural network for 5 epochs
- Print test accuracy/loss
- Save the model to `models/mnist_digit_model.h5`
- Save training curves to `reports/training_history.png`
- Save metrics to `reports/metrics.json`

## 3. Predict on custom images

Add your own handwritten digit images (`.png`/`.jpg`) to `data/custom_digits/`:

```
data/custom_digits/
├── digit_3.png
├── digit_7.png
```

Images can be any size or color — preprocessing (grayscale conversion, resize
to 28x28, auto-invert, normalize) is handled automatically and identically
across the CLI script, notebook, and Streamlit app (`src/preprocessing.py`).

Then run:

```bash
python -m src.predict
```

This prints predicted digit + confidence for each image and displays them
with matplotlib.

## 4. Explore interactively in the notebook

```bash
jupyter notebook notebooks/mnist_digit_classifier.ipynb
```

The notebook reuses the exact same `src/` code as the scripts, and walks
through data exploration, training, evaluation, saving, and custom inference
with visualizations at every step.

## 5. Run the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

Features:
- **Draw tab**: draw a digit on a canvas and get an instant prediction
  (requires `pip install streamlit-drawable-canvas`)
- **Upload tab**: upload an image of a handwritten digit
- Shows predicted digit, confidence, and a bar chart of confidence per class

## Model

- Fully connected neural network: `Flatten -> Dense(128, relu) -> Dropout(0.2) -> Dense(64, relu) -> Dense(10, softmax)`
- Trained on 60,000 MNIST images, evaluated on 10,000 held-out test images
- Typically reaches ~97-98% test accuracy after 5 epochs

## Example Predictions

| Input          | Output |
|----------------|--------|
| `digit_3.png`  | 3      |
| `digit_7.png`  | 7      |

## Key Insight

Simple neural networks generalize well when custom-image preprocessing
(grayscale, resize, normalization, and color polarity) matches how the
training data was prepared — most bugs in "why is my own handwriting
misclassified" come from a preprocessing mismatch, not the model itself.

## Running Tests

```bash
pytest tests/
```

## Push to GitHub

```bash
git init
git add .
git commit -m "Project 14: Handwritten Digit Classifier with Custom Inference + Streamlit app"
git branch -M main
git remote add origin <YOUR_REPO_URL>
git push -u origin main
```
