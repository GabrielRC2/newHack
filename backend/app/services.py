"""Regras puras e servicos de dominio para matching e frequencia."""
from __future__ import annotations

import math
import os
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import Attendance, FaceEmbedding, PresenceInterval, RecognitionEvent, Student


# O modelo SFace do OpenCV usa similaridade de cosseno; 0.45 e um ponto de partida
# conservador. Este valor deve ser calibrado com dados autorizados da instituicao.
MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.45"))
COOLDOWN_SECONDS = int(os.getenv("RECOGNITION_COOLDOWN_SECONDS", "15"))


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return -1.0
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    if norm_left == 0 or norm_right == 0:
        return -1.0
    return dot / (norm_left * norm_right)


def find_student_by_embedding(db: Session, vector: list[float], threshold: float = MATCH_THRESHOLD):
    best_student: Student | None = None
    best_score = -1.0
    for item in db.scalars(select(FaceEmbedding)).all():
        score = cosine_similarity(vector, item.vector)
        if score > best_score:
            best_score = score
            best_student = item.student
    if best_student is None or best_score < threshold:
        return None, best_score if best_score >= 0 else None
    return best_student, best_score


def get_or_create_attendance(db: Session, class_id: int, student_id: int) -> Attendance:
    record = db.scalar(select(Attendance).where(
        Attendance.class_session_id == class_id,
        Attendance.student_id == student_id,
    ))
    if record is None:
        record = Attendance(class_session_id=class_id, student_id=student_id)
        db.add(record)
        db.flush()
    return record


def clipped_seconds(interval: PresenceInterval, starts_at: datetime, ends_at: datetime, open_until: datetime | None = None) -> int:
    interval_end = interval.exited_at or open_until
    if interval_end is None:
        return 0
    start = max(interval.entered_at, starts_at)
    end = min(interval_end, ends_at)
    return max(0, int((end - start).total_seconds()))


def total_present_seconds(db: Session, class_id: int, student_id: int, starts_at: datetime, ends_at: datetime, open_until: datetime | None = None) -> int:
    intervals = db.scalars(select(PresenceInterval).where(
        PresenceInterval.class_session_id == class_id,
        PresenceInterval.student_id == student_id,
    )).all()
    return sum(clipped_seconds(interval, starts_at, ends_at, open_until) for interval in intervals)


def last_state_change(db: Session, class_id: int, student_id: int) -> RecognitionEvent | None:
    return db.scalar(select(RecognitionEvent).where(
        RecognitionEvent.class_session_id == class_id,
        RecognitionEvent.student_id == student_id,
        RecognitionEvent.status == "identified",
    ).order_by(RecognitionEvent.detected_at.desc()))
