"""Persistência SQLite para o primeiro marco do MVP."""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Union


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "data" / "attendance.db"


def database_path() -> Path:
    """Retorna o caminho configurado ou o banco SQLite local padrão."""

    configured_path = os.getenv("ATTENDANCE_DB_PATH")
    return Path(configured_path) if configured_path else DEFAULT_DATABASE_PATH


def connect(path: Optional[Union[str, Path]] = None) -> sqlite3.Connection:
    """Abre uma conexão com chaves estrangeiras habilitadas."""

    resolved_path = Path(path) if path is not None else database_path()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(str(resolved_path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database(path: Optional[Union[str, Path]] = None) -> None:
    """Cria as tabelas do marco inicial, sem acoplar a API ao SQLite."""

    connection = connect(path)
    try:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS alunos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nome TEXT NOT NULL,
                matricula TEXT NOT NULL UNIQUE,
                ativo INTEGER NOT NULL DEFAULT 1,
                criado_em TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS cameras (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                codigo TEXT NOT NULL UNIQUE,
                nome TEXT NOT NULL,
                papel TEXT NOT NULL CHECK (papel IN ('ENTRADA', 'SAIDA')),
                ativa INTEGER NOT NULL DEFAULT 1,
                criado_em TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessoes_aula (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nome TEXT NOT NULL,
                camera_saida_id INTEGER NOT NULL,
                inicio_em TEXT NOT NULL,
                fim_em TEXT NOT NULL,
                corte_em TEXT NOT NULL,
                criado_em TEXT NOT NULL,
                FOREIGN KEY (camera_saida_id) REFERENCES cameras(id)
            );

            CREATE TABLE IF NOT EXISTS eventos_reconhecimento (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id INTEGER NOT NULL,
                aluno_id INTEGER,
                sessao_aula_id INTEGER,
                ocorreu_em TEXT NOT NULL,
                recebido_em TEXT NOT NULL,
                decisao TEXT NOT NULL,
                FOREIGN KEY (camera_id) REFERENCES cameras(id),
                FOREIGN KEY (aluno_id) REFERENCES alunos(id),
                FOREIGN KEY (sessao_aula_id) REFERENCES sessoes_aula(id)
            );

            CREATE TABLE IF NOT EXISTS presencas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sessao_aula_id INTEGER NOT NULL,
                aluno_id INTEGER NOT NULL,
                evento_id INTEGER NOT NULL,
                registrado_em TEXT NOT NULL,
                UNIQUE (sessao_aula_id, aluno_id),
                FOREIGN KEY (sessao_aula_id) REFERENCES sessoes_aula(id),
                FOREIGN KEY (aluno_id) REFERENCES alunos(id),
                FOREIGN KEY (evento_id) REFERENCES eventos_reconhecimento(id)
            );

            CREATE INDEX IF NOT EXISTS idx_sessoes_aula_camera_horario
                ON sessoes_aula(camera_saida_id, inicio_em, fim_em);
            CREATE INDEX IF NOT EXISTS idx_eventos_reconhecimento_camera_horario
                ON eventos_reconhecimento(camera_id, ocorreu_em);
            """
        )
        connection.commit()
    finally:
        connection.close()


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    """Fornece uma transação curta por requisição HTTP."""

    connection = connect()
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
