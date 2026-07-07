# Project 12 — Cat vs Dog Classifier (Transfer Learning)

Binary image classification using a pretrained MobileNetV2 backbone: freeze →
train head → unfreeze top layers → fine-tune → evaluate → save model →
training curve plot. Ships with a CLI predictor and a Streamlit web app.

## Project Structure

```
project-12-cat-dog/
├── app/
│   └── streamlit_app.py         # Interactive web app (upload a photo)
├── data/
│   ├── raw/                     # Downloaded dataset lands here (auto-created)
│   └── custom_images/           # Put your own cat/dog photos here
├── models/
│   └── cat_dog_transfer.h5      # Saved trained model (created after training)
├── notebooks/
│   └── cat_dog_classifier.ipynb # Full walkthrough: data, train, fine-tune, predict
├── reports/
│   ├── figures/
│   │   └── training_curve.png   # Mandatory submission plot (created after training)
│   └── metrics.json             # Final val accuracy/loss (created after training)
├── src/
│   ├── __init__.py
│   ├── config.py                 # Shared paths & hyperparameters
│   ├── data.py                   # Dataset download + tf.data loading
│   ├── model.py                  # Model architecture + fine-tuning helper
│   ├── train.py                  # Two-phase training script
│   └── predict.py                # CLI inference script
├── tests/
│   └── test_model.py             # Offline-safe architecture sanity tests
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
- Download the Cats vs Dogs dataset (~68MB, cached under `data/raw/` after
  the first run — safe to re-run, won't re-download or break)
- **Phase 1:** freeze the MobileNetV2 backbone and train the classification
  head for 5 epochs
- **Phase 2:** unfreeze the top 50 backbone layers and fine-tune at a much
  lower learning rate (1e-5) for 5 more epochs
- Evaluate final validation accuracy/loss
- Save the model to `models/cat_dog_transfer.h5`
- Save the training curve to `reports/figures/training_curve.png` ✅ **mandatory for submission**
- Save metrics to `reports/metrics.json`

> **Note on the dataset split:** the raw download already ships separate
> `train/` and `validation/` folders. `src/data.py` points at each directly
> instead of re-splitting `train/` with `validation_split`, which keeps the
> train/validation sets exactly as the dataset authors intended and makes
> re-running the download step idempotent.

## 3. Predict on custom images

Add your own cat/dog photos (`.png`/`.jpg`) to `data/custom_images/`:

```
data/custom_images/
├── my_cat.jpg
├── my_dog.jpg
```

Then run:

```bash
python -m src.predict
```

This prints predicted label + confidence for each image and displays them
with matplotlib. MobileNetV2's `preprocess_input` is baked directly into the
model itself, so the script only resizes images — no manual normalization
step to keep in sync across script, notebook, and app.

## 4. Explore interactively in the notebook

```bash
jupyter notebook notebooks/cat_dog_classifier.ipynb
```

The notebook reuses the exact same `src/` code as the scripts and walks
through data exploration, phase-1 training, phase-2 fine-tuning, evaluation,
saving, and custom inference with visualizations at every step.

## 5. Run the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

Upload a photo of a cat or dog and get an instant prediction with a
confidence breakdown.

## Model

- **Backbone:** MobileNetV2 (ImageNet pretrained), frozen in phase 1
- **Head:** `GlobalAveragePooling2D → Dropout(0.2) → Dense(128, relu) → Dense(1, sigmoid)`
- **Phase 1:** train head only, Adam @ 1e-3, 5 epochs
- **Phase 2:** unfreeze last 50 backbone layers, Adam @ 1e-5, 5 more epochs

## Training Curve

![Training Curve](reports/figures/training_curve.png)

*(Generated after running `python -m src.train` or the notebook — the
dashed line marks where fine-tuning begins.)*

## Key Insight

Transfer learning enables strong performance with limited data and
significantly reduces training time, since most of the useful visual
features (edges, textures, shapes) are already learned from ImageNet.
Freezing the backbone first prevents large random gradients from the
untrained head from destroying those pretrained features; only after the
head has stabilized is it safe to unfreeze the top backbone layers and
fine-tune at a low learning rate for the last bit of accuracy.

## Running Tests

```bash
pytest tests/
```

Tests use `weights=None` so they run fully offline (no ImageNet weight
download required) and only check architecture shapes, output validity,
and freeze/unfreeze behavior.

## Push to GitHub

```bash
git init
git add .
git commit -m "Project 12: Cat vs Dog Classifier using Transfer Learning + Streamlit app"
git branch -M main
git remote add origin <YOUR_REPO_URL>
git push -u origin main
```
