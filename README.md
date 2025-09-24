# 📄 SvaraAI – Reply Classification Project

This repository contains the solution for the SvaraAI Internship Assignment:

## Part A: ML/NLP pipeline (Google Colab)

## Part B: Deployment as a Flask API with optional web UI

---

## 🚀 Part A — Reply Classification Pipeline

The pipeline is implemented in a Google Colab notebook.
For best results, run the cells sequentially from top to bottom.

### 1. Install Dependencies

Run this in the first cell:

```bash
pip install pandas scikit-learn numpy joblib torch transformers datasets
```

(Standard libraries like re, os, warnings, and shutil are already included in Python.)

### 2. Data Setup

Upload the dataset file `reply_classification_dataset.csv` to your Colab environment.
Preprocessing will generate a `cleaned_data.csv` file for model training.

### 3. Preprocessing

Initial cells handle:
- Lowercasing
- Removing URLs/special chars
- Label conversion

### 4. Model Training

Notebook sections:
- Baseline (SGDClassifier + TF-IDF)
- Transformer (DistilBERT fine-tuned)

### 5. Saved Artifacts

**Baseline (SGDClassifier)**  
Saved under `./baseline_model/`  
Files: `calibrated_sgd_model.pkl`, `vectorizer.pkl`

**DistilBERT (fine-tuned)**  
Saved at `/content/distilbert_reply_classifier/`  
Includes both model + tokenizer  
Exported as a `.zip` file

---

## 🤖 Models & Evaluation

**Baseline (SGDClassifier + TF-IDF):**  
Accuracy = 98.83%, Weighted F1 = 0.99

**Transformer (DistilBERT):**  
Validation Accuracy = 99.30%

---

## ✨ Production Recommendation

For production, the Calibrated SGDClassifier is recommended:
- Near-identical performance to DistilBERT
- Faster inference
- Lower compute cost
- Simpler deployment

---

## 🔢 Notebook Cell Guide

- Cell 1: Load & preprocess dataset
- Cell 2: Train baseline SGDClassifier
- Cell 3: Calibrate + save baseline model/vectorizer
- Cell 4: Fine-tune DistilBERT
- Cell 5: Save DistilBERT model as zip
- Cell 6: Demonstrate DistilBERT inference

---

## 🖥️ Part B — Reply Sentiment Classifier API

Wraps the DistilBERT model in a Flask service with a `/predict` endpoint and optional web UI.

### API

**POST** `/predict`  

Input JSON (supports both keys):

```json
{ "text": "Looking forward to the demo!" }
```

or

```json
{ "reply": "Looking forward to the demo!" }
```

Response JSON:

```json
{
  "label": "positive",
  "confidence_top": 0.87,
  "predicted_label": "positive",
  "confidence": { "negative": 0.02, "positive": 0.87, "neutral": 0.11 },
  "history": [ ... ]
}
```

### Local Run

Python 3.10+ recommended

#### Create venv + install dependencies

**Windows PowerShell**

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**macOS/Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### Ensure model path exists

**Windows:**

```bash
set MODEL_PATH=C:\CODES\PY\AINTERNTASK\SVARAAI\models\distilbert_reply_classifier
```

**macOS/Linux:**

```bash
export MODEL_PATH=/abs/path/to/models/distilbert_reply_classifier
```

#### Start server

```bash
python app.py
```

#### Test API via curl

```bash
curl -X POST http://127.0.0.1:5000/predict      -H "Content-Type: application/json"      -d '{"text":"Looking forward to the demo!"}'
```

---

## 🌐 Optional — Web UI

The repository includes a minimal `index.html` interface.

Once the server is running (`python app.py`), open your browser at:  
👉 http://127.0.0.1:5000

Enter text into the UI, hit Submit, and see predictions directly in your browser.
