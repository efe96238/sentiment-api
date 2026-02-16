import torch
from torch import nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from models import SentimentAPI

device = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_csv("data/sentiment.csv")
df['text'] = df['text'].astype(str).str.strip().str.lower()
df['label'] = df['label'].astype(int)

X_train_text, X_val_text, y_train, y_val = train_test_split(df["text"], df["label"], test_size=0.2, shuffle=True, random_state=42)

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train_text).toarray()
X_val_vec = vectorizer.transform(X_val_text).toarray()

vocab_size = len(vectorizer.vocabulary_)

X_train = torch.tensor(X_train_vec, dtype=torch.float32).to(device)
X_val = torch.tensor(X_val_vec, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train.values, dtype=torch.long).to(device)
y_val = torch.tensor(y_val.values, dtype=torch.long).to(device)

model = SentimentAPI(vocab_size=vocab_size).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train(model, epochs:int):
  torch.manual_seed(42)

  for epoch in range(epochs):
    model.train()
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
      train_pred = model(X_train)
      train_acc = (train_pred.argmax(dim=1) == y_train).float().mean()

      val_pred = model(X_val)
      val_loss = loss_fn(val_pred, y_val)
      val_acc = (val_pred.argmax(dim=1) == y_val).float().mean()
    
    if epoch % 10 == 0:
      print(f"Epoch: {epoch} | Loss: {loss:.4f} | Train Acc: {train_acc:.3f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f}")

if __name__ == "__main__":
  train(model=model, epochs=100)
  torch.save(model.state_dict(), "models/SentimentAPI.pth")