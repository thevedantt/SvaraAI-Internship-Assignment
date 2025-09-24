from flask import Flask, render_template, request, jsonify
import os
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch.nn.functional as F
import numpy as np

# --- Load model and tokenizer ---
MODEL_PATH = os.getenv("MODEL_PATH", r"C:\CODES\PY\AINTERNTASK\SVARAAI\models\distilbert_reply_classifier")
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# --- Label mapping ---
label_mapping_rev = {0: "negative", 1: "positive", 2: "neutral"}

# --- Flask app ---
app = Flask(__name__)

# In-memory history store
history = []  # list of {text, label, confidence, confidence_per_label}

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True) or {}
        # Accept both keys: "text" (spec) and "reply" (UI)
        raw_text = data.get("text") if data.get("text") is not None else data.get("reply", "")
        user_input = (raw_text or "").strip()
        if not user_input:
            return jsonify({"error": "Empty input."}), 400

        # Tokenize input
        inputs = tokenizer(
            user_input,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )

        # Model inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()

        # Determine predicted label
        pred_idx = int(np.argmax(probs))
        predicted_label = label_mapping_rev[pred_idx]

        # Confidence per label
        confidence = {label_mapping_rev[i]: float(probs[i]) for i in range(len(probs))}

        # Optional: if all probs < 0.5, classify as neutral (adjust threshold if needed)
        # if max(probs) < 0.5:
        #     predicted_label = "neutral"

        # Update history (most recent first), cap to 50 items
        history.insert(0, {
            "text": user_input,
            "label": predicted_label,
            "confidence": float(probs[pred_idx]),
            "confidence_per_label": confidence
        })
        if len(history) > 50:
            del history[50:]

        top_confidence = float(probs[pred_idx])

        return jsonify({
            "predicted_label": predicted_label,
            "confidence": confidence,
            "history": history,
            # Spec-compatible fields
            "label": predicted_label,
            "confidence_top": top_confidence
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
