from datetime import datetime
from pydantic import BaseModel

class PredictRequest(BaseModel):
  text: str

class PredictResponse(BaseModel):
  label: str
  confidence: float

class HistoryItem(BaseModel):
  id: int
  text: str
  label: str
  confidence: float
  created_at: datetime

  class Config:
    from_attributes = True