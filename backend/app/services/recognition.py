"""Processamento de eventos de reconhecimento, ainda sem modelo facial real."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from app.schemas import DecisaoReconhecimento, PapelCamera
from app.services.attendance import evaluate_exit_time


@dataclass(frozen=True)
class RecognitionOutcome:
    event_id: int
    decision: DecisaoReconhecimento
    lesson_session_id: Optional[int]
    cutoff_at: Optional[datetime]


def process_recognition_event(
    connection: sqlite3.Connection,
    *,
    camera: sqlite3.Row,
    student: Optional[sqlite3.Row],
    occurred_at: datetime,
    received_at: datetime,
) -> RecognitionOutcome:
    """Persiste um evento e aplica a regra de presença quando for aplicável."""

    if student is None:
        decision = (
            DecisaoReconhecimento.CADASTRO_NECESSARIO
            if camera["papel"] == PapelCamera.ENTRADA.value
            else DecisaoReconhecimento.ROSTO_DESCONHECIDO
        )
        event_id = _insert_event(
            connection,
            camera_id=camera["id"],
            student_id=None,
            lesson_session_id=None,
            occurred_at=occurred_at,
            received_at=received_at,
            decision=decision,
        )
        return RecognitionOutcome(
            event_id=event_id,
            decision=decision,
            lesson_session_id=None,
            cutoff_at=None,
        )

    if camera["papel"] == PapelCamera.ENTRADA.value:
        event_id = _insert_event(
            connection,
            camera_id=camera["id"],
            student_id=student["id"],
            lesson_session_id=None,
            occurred_at=occurred_at,
            received_at=received_at,
            decision=DecisaoReconhecimento.ENTRADA_IDENTIFICADA,
        )
        return RecognitionOutcome(
            event_id=event_id,
            decision=DecisaoReconhecimento.ENTRADA_IDENTIFICADA,
            lesson_session_id=None,
            cutoff_at=None,
        )

    lesson_session = _find_active_lesson_session(connection, camera["id"], occurred_at)
    if lesson_session is None:
        event_id = _insert_event(
            connection,
            camera_id=camera["id"],
            student_id=student["id"],
            lesson_session_id=None,
            occurred_at=occurred_at,
            received_at=received_at,
            decision=DecisaoReconhecimento.IDENTIFICADO_SEM_AULA_ATIVA,
        )
        return RecognitionOutcome(
            event_id=event_id,
            decision=DecisaoReconhecimento.IDENTIFICADO_SEM_AULA_ATIVA,
            lesson_session_id=None,
            cutoff_at=None,
        )

    starts_at = datetime.fromisoformat(lesson_session["inicio_em"])
    ends_at = datetime.fromisoformat(lesson_session["fim_em"])
    evaluation = evaluate_exit_time(
        lesson_starts_at=starts_at,
        lesson_ends_at=ends_at,
        occurred_at=occurred_at,
    )
    cutoff_at = evaluation.halfway_at
    lesson_session_id = int(lesson_session["id"])

    if not evaluation.qualifies_for_attendance:
        event_id = _insert_event(
            connection,
            camera_id=camera["id"],
            student_id=student["id"],
            lesson_session_id=lesson_session_id,
            occurred_at=occurred_at,
            received_at=received_at,
            decision=DecisaoReconhecimento.IDENTIFICADO_ANTES_DO_CORTE,
        )
        return RecognitionOutcome(
            event_id=event_id,
            decision=DecisaoReconhecimento.IDENTIFICADO_ANTES_DO_CORTE,
            lesson_session_id=lesson_session_id,
            cutoff_at=cutoff_at,
        )

    event_id = _insert_event(
        connection,
        camera_id=camera["id"],
        student_id=student["id"],
        lesson_session_id=lesson_session_id,
        occurred_at=occurred_at,
        received_at=received_at,
        decision=DecisaoReconhecimento.PRESENCA_REGISTRADA,
    )
    cursor = connection.execute(
        """
        INSERT OR IGNORE INTO presencas
            (sessao_aula_id, aluno_id, evento_id, registrado_em)
        VALUES (?, ?, ?, ?)
        """,
        # A presença representa o instante observado pela câmera; o horário de
        # recebimento continua registrado no evento para auditoria técnica.
        (lesson_session_id, student["id"], event_id, occurred_at.isoformat()),
    )

    if cursor.rowcount == 1:
        decision = DecisaoReconhecimento.PRESENCA_REGISTRADA
    else:
        decision = DecisaoReconhecimento.PRESENCA_JA_REGISTRADA
        connection.execute(
            "UPDATE eventos_reconhecimento SET decisao = ? WHERE id = ?",
            (decision.value, event_id),
        )

    return RecognitionOutcome(
        event_id=event_id,
        decision=decision,
        lesson_session_id=lesson_session_id,
        cutoff_at=cutoff_at,
    )


def _find_active_lesson_session(
    connection: sqlite3.Connection, camera_id: int, occurred_at: datetime
) -> Optional[sqlite3.Row]:
    """Busca sessão no intervalo [início, fim), evitando ambiguidade às 16:00."""

    return connection.execute(
        """
        SELECT id, inicio_em, fim_em
        FROM sessoes_aula
        WHERE camera_saida_id = ?
          AND inicio_em <= ?
          AND fim_em > ?
        ORDER BY inicio_em DESC
        LIMIT 1
        """,
        (camera_id, occurred_at.isoformat(), occurred_at.isoformat()),
    ).fetchone()


def _insert_event(
    connection: sqlite3.Connection,
    *,
    camera_id: int,
    student_id: Optional[int],
    lesson_session_id: Optional[int],
    occurred_at: datetime,
    received_at: datetime,
    decision: DecisaoReconhecimento,
) -> int:
    cursor = connection.execute(
        """
        INSERT INTO eventos_reconhecimento
            (camera_id, aluno_id, sessao_aula_id, ocorreu_em, recebido_em, decisao)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            camera_id,
            student_id,
            lesson_session_id,
            occurred_at.isoformat(),
            received_at.isoformat(),
            decision.value,
        ),
    )
    return int(cursor.lastrowid)
