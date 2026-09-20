from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


class Student(Base):
    __tablename__ = "students"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    enrollment_number: Mapped[str] = mapped_column(String(80), unique=True, index=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    # Estado operacional da aula atual. Attendance e PresenceInterval sao a fonte historica.
    # Novo cadastro e tratado como aluno dentro da aula, conforme a regra do projeto.
    # O primeiro reconhecimento ainda cria o intervalo auditavel que sera contabilizado.
    em_aula: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    tempo_em_aula: Mapped[int] = mapped_column(Integer, default=0, nullable=False)  # segundos

    embeddings: Mapped[list["FaceEmbedding"]] = relationship(back_populates="student", cascade="all, delete-orphan")


class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    student_id: Mapped[int] = mapped_column(ForeignKey("students.id"), index=True, nullable=False)
    vector: Mapped[list[float]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    student: Mapped[Student] = relationship(back_populates="embeddings")


class ClassSession(Base):
    __tablename__ = "class_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    label: Mapped[str] = mapped_column(String(160), nullable=False)
    starts_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    ends_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    minimum_percentage: Mapped[float] = mapped_column(Float, default=0.75, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="open", nullable=False)  # open | closed
    closed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class RecognitionEvent(Base):
    __tablename__ = "recognition_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    class_session_id: Mapped[int] = mapped_column(ForeignKey("class_sessions.id"), index=True, nullable=False)
    student_id: Mapped[int | None] = mapped_column(ForeignKey("students.id"), index=True, nullable=True)
    camera_id: Mapped[str] = mapped_column(String(80), nullable=False)
    detected_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    action: Mapped[str | None] = mapped_column(String(32), nullable=True)
    match_score: Mapped[float | None] = mapped_column(Float, nullable=True)


class PresenceInterval(Base):
    __tablename__ = "presence_intervals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    class_session_id: Mapped[int] = mapped_column(ForeignKey("class_sessions.id"), index=True, nullable=False)
    student_id: Mapped[int] = mapped_column(ForeignKey("students.id"), index=True, nullable=False)
    entered_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    exited_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class Attendance(Base):
    __tablename__ = "attendance"
    __table_args__ = (UniqueConstraint("class_session_id", "student_id", name="uq_attendance_session_student"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    class_session_id: Mapped[int] = mapped_column(ForeignKey("class_sessions.id"), index=True, nullable=False)
    student_id: Mapped[int] = mapped_column(ForeignKey("students.id"), index=True, nullable=False)
    total_seconds: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    required_seconds: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="pending", nullable=False)
    finalized_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
