from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional

class StudentCreate(BaseModel):
    nome: str
    matricula: str
    embedding: List[float]

class ClassSessionCreate(BaseModel):
    nome: str
    starts_at: datetime
    ends_at: datetime
    min_presence_percent: float = 0.75

class RecognitionPayload(BaseModel):
    class_id: int
    camera_id: str
    embedding: List[float]
    timestamp: datetime