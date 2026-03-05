import json
import torch
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from sklearn.feature_extraction.text import CountVectorizer

from models import SentimentAPI
from database import get_db, init_db, Prediction
from schemas import PredictRequest, PredictResponse

device = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI()

# create tables if missing
init_db()

# load vocab
with open("models/vocab.json", "r", encoding="utf-8") as f:
  vocab = json.load(f)

vectorizer = CountVectorizer(vocabulary=vocab)
vocab_size = len(vocab)

# load model weights from models folder
model = SentimentAPI(vocab_size=vocab_size).to(device)
model.load_state_dict(torch.load("models/SentimentAPI.pth", map_location=device))
model.eval()

label_map = {0: "negative", 1: "positive"}

@app.get("/health")
def health():
  return {"ok": True}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, db: Session = Depends(get_db)):
  text = request.text.strip()
  if not text:
    raise HTTPException(status_code=400, detail="text must not be empty")

  text_clean = text.lower()

  X = vectorizer.transform([text_clean]).toarray()
  X = torch.tensor(X, dtype=torch.float32).to(device)

  with torch.inference_mode():
    logits = model(X)
    probs = torch.softmax(logits, dim=1)
    pred_idx = probs.argmax(dim=1).item()
    confidence = probs[0, pred_idx].item()

  label = label_map[pred_idx]

  db_entry = Prediction(text=text_clean, label=label, confidence=confidence)
  db.add(db_entry)
  db.commit()

  return PredictResponse(label=label, confidence=confidence)