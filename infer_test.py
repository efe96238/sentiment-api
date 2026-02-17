import torch
import json
from sklearn.feature_extraction.text import CountVectorizer

from models import SentimentAPI

device = "cuda" if torch.cuda.is_available() else "cpu"

with open("models/vocab.json", "r", encoding="utf-8") as f:
  vocab = json.load(f)

vectorizer = CountVectorizer(vocabulary=vocab)
vocab_size = len(vocab)

loaded_model = SentimentAPI(vocab_size=vocab_size).to(device)
loaded_model.load_state_dict(torch.load("models/SentimentAPI.pth", map_location=device))

texts = ["i love this", "this is terrible", "great quality", "worst purchase ever"]

X = vectorizer.transform(texts).toarray()
X = torch.tensor(X, dtype=torch.float32).to(device)

def infere(model, X):
  model.eval()
  with torch.inference_mode():
    logits = model(X)
    probs = torch.softmax(logits, dim=1)
    pred = probs.argmax(dim=1)
    print(pred)
    print(probs)

if __name__ == "__main__":
  infere(loaded_model, X)