"""API HTTP do primeiro marco do MVP de controle de presença."""

from __future__ import annotations

import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator, List

from fastapi import FastAPI, HTTPException, status

from app.database import get_connection, initialize_database
from app.schemas import (
    AlunoCreate,
    AlunoResponse,
    CameraCreate,
    CameraResponse,
    EventoReconhecimentoCreate,
    EventoReconhecimentoResponse,
    PapelCamera,
    PresencaResponse,
    SessaoAulaCreate,
    SessaoAulaResponse,
    as_utc,
)
from app.services.recognition import process_recognition_event


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    initialize_database()
    yield


app = FastAPI(
    title="Presença por Reconhecimento Facial",
    version="0.1.0",
    description=(
        "MVP com duas câmeras: cadastro na entrada e presença definida pelo "
        "horário de identificação na saída."
    ),
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/alunos", response_model=AlunoResponse, status_code=status.HTTP_201_CREATED)
def create_student(payload: AlunoCreate) -> AlunoResponse:
    created_at = datetime.now(timezone.utc)
    try:
        with get_connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO alunos (nome, matricula, criado_em)
                VALUES (?, ?, ?)
                """,
                (payload.nome, payload.matricula, created_at.isoformat()),
            )
            student_id = int(cursor.lastrowid)
    except sqlite3.IntegrityError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Já existe um aluno com esta matrícula.",
        ) from error

    return AlunoResponse(
        id=student_id,
        nome=payload.nome,
        matricula=payload.matricula,
        criado_em=created_at,
    )


@app.post("/cameras", response_model=CameraResponse, status_code=status.HTTP_201_CREATED)
def create_camera(payload: CameraCreate) -> CameraResponse:
    created_at = datetime.now(timezone.utc)
    try:
        with get_connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO cameras (codigo, nome, papel, criado_em)
                VALUES (?, ?, ?, ?)
                """,
                (payload.codigo, payload.nome, payload.papel.value, created_at.isoformat()),
            )
            camera_id = int(cursor.lastrowid)
    except sqlite3.IntegrityError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Já existe uma câmera com este código.",
        ) from error

    return CameraResponse(
        id=camera_id,
        codigo=payload.codigo,
        nome=payload.nome,
        papel=payload.papel,
        criada_em=created_at,
    )


@app.post(
    "/sessoes-aula",
    response_model=SessaoAulaResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_lesson_session(payload: SessaoAulaCreate) -> SessaoAulaResponse:
    starts_at = as_utc(payload.inicio_em)
    ends_at = as_utc(payload.fim_em)
    cutoff_at = starts_at + (ends_at - starts_at) / 2
    created_at = datetime.now(timezone.utc)

    with get_connection() as connection:
        camera = connection.execute(
            "SELECT id, papel FROM cameras WHERE codigo = ?", (payload.camera_saida_codigo,)
        ).fetchone()
        if camera is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Câmera não encontrada.")
        if camera["papel"] != PapelCamera.SAIDA.value:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Uma sessão de aula deve usar uma câmera com papel SAIDA.",
            )

        overlapping_session = connection.execute(
            """
            SELECT id FROM sessoes_aula
            WHERE camera_saida_id = ?
              AND inicio_em < ?
              AND fim_em > ?
            LIMIT 1
            """,
            (camera["id"], ends_at.isoformat(), starts_at.isoformat()),
        ).fetchone()
        if overlapping_session is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Já existe uma sessão de aula sobreposta para esta câmera de saída.",
            )

        cursor = connection.execute(
            """
            INSERT INTO sessoes_aula
                (nome, camera_saida_id, inicio_em, fim_em, corte_em, criado_em)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                payload.nome,
                camera["id"],
                starts_at.isoformat(),
                ends_at.isoformat(),
                cutoff_at.isoformat(),
                created_at.isoformat(),
            ),
        )
        session_id = int(cursor.lastrowid)

    return SessaoAulaResponse(
        id=session_id,
        nome=payload.nome,
        camera_saida_codigo=payload.camera_saida_codigo,
        inicio_em=starts_at,
        fim_em=ends_at,
        corte_em=cutoff_at,
        criado_em=created_at,
    )


@app.post(
    "/eventos-reconhecimento",
    response_model=EventoReconhecimentoResponse,
    status_code=status.HTTP_201_CREATED,
)
def register_recognition_event(
    payload: EventoReconhecimentoCreate,
) -> EventoReconhecimentoResponse:
    """Simula a identificação que futuramente será produzida pelo matcher facial."""

    occurred_at = as_utc(payload.ocorreu_em)
    received_at = datetime.now(timezone.utc)

    with get_connection() as connection:
        camera = connection.execute(
            "SELECT id, codigo, papel FROM cameras WHERE codigo = ?", (payload.camera_codigo,)
        ).fetchone()
        if camera is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Câmera não encontrada.")

        student = None
        if payload.aluno_id is not None:
            student = connection.execute(
                "SELECT id FROM alunos WHERE id = ? AND ativo = 1", (payload.aluno_id,)
            ).fetchone()
            if student is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Aluno não encontrado.")

        outcome = process_recognition_event(
            connection,
            camera=camera,
            student=student,
            occurred_at=occurred_at,
            received_at=received_at,
        )

    return EventoReconhecimentoResponse(
        id=outcome.event_id,
        camera_codigo=payload.camera_codigo,
        aluno_id=payload.aluno_id,
        decisao=outcome.decision,
        sessao_aula_id=outcome.lesson_session_id,
        ocorreu_em=occurred_at,
        corte_em=outcome.cutoff_at,
    )


@app.get("/sessoes-aula/{sessao_aula_id}/presencas", response_model=List[PresencaResponse])
def list_attendance(sessao_aula_id: int) -> List[PresencaResponse]:
    with get_connection() as connection:
        session_exists = connection.execute(
            "SELECT id FROM sessoes_aula WHERE id = ?", (sessao_aula_id,)
        ).fetchone()
        if session_exists is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sessão de aula não encontrada.")

        rows = connection.execute(
            """
            SELECT
                presencas.aluno_id,
                alunos.nome AS aluno_nome,
                alunos.matricula,
                presencas.evento_id,
                presencas.registrado_em
            FROM presencas
            JOIN alunos ON alunos.id = presencas.aluno_id
            WHERE presencas.sessao_aula_id = ?
            ORDER BY presencas.registrado_em ASC
            """,
            (sessao_aula_id,),
        ).fetchall()

    return [
        PresencaResponse(
            aluno_id=row["aluno_id"],
            aluno_nome=row["aluno_nome"],
            matricula=row["matricula"],
            evento_id=row["evento_id"],
            registrado_em=datetime.fromisoformat(row["registrado_em"]),
        )
        for row in rows
    ]
