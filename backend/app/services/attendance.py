"""Regra temporal de presença da câmera de saída."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ExitEvaluation:
    """Resultado da avaliação de um aluno identificado na saída."""

    occurred_at: datetime
    halfway_at: datetime
    qualifies_for_attendance: bool


def evaluate_exit_time(
    *, lesson_starts_at: datetime, lesson_ends_at: datetime, occurred_at: datetime
) -> ExitEvaluation:
    """Avalia se uma identificação na saída gera presença.

    "Depois da metade" é uma comparação estrita. Um evento exatamente no
    ponto médio é identificado, mas ainda não recebe presença.
    """

    _ensure_timezone("lesson_starts_at", lesson_starts_at)
    _ensure_timezone("lesson_ends_at", lesson_ends_at)
    _ensure_timezone("occurred_at", occurred_at)

    if lesson_ends_at <= lesson_starts_at:
        raise ValueError("lesson_ends_at deve ser posterior a lesson_starts_at")

    halfway_at = lesson_starts_at + (lesson_ends_at - lesson_starts_at) / 2
    return ExitEvaluation(
        occurred_at=occurred_at,
        halfway_at=halfway_at,
        qualifies_for_attendance=occurred_at > halfway_at,
    )


def _ensure_timezone(field_name: str, value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} deve informar o fuso horário")
