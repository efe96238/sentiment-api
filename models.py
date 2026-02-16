import torch
from torch import nn

class SentimentAPI(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()
    self.vocab_size = vocab_size
    self.stack = nn.Sequential(
      nn.Linear(in_features=vocab_size, out_features=128),
      nn.ReLU(),
      nn.Linear(in_features=128, out_features=2)
    )

  def forward(self, x):
    return self.stack(x)