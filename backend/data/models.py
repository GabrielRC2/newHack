from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey, Float
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from data.database import Base


class Student(Base):
    __tablename__ = "students"
    id = Column(Integer, primary_key=True, index=True)
    nome = Column(String, index=True)
    matricula = Column(String, unique=True, index=True)
    data_cadastro = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    embeddings = relationship("FaceEmbedding", back_populates="student")
    attendances = relationship("Attendance", back_populates="student")


class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"
    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id"))
    embedding_json = Column(String)  # Armazenado como string JSON para SQLite

    student = relationship("Student", back_populates="embeddings")


class ClassSession(Base):
    __tablename__ = "class_sessions"
    id = Column(Integer, primary_key=True, index=True)
    nome = Column(String)
    starts_at = Column(DateTime)
    ends_at = Column(DateTime)
    min_presence_percent = Column(Float, default=0.75)
    is_closed = Column(Boolean, default=False)

    attendances = relationship("Attendance", back_populates="class_session")


class Attendance(Base):
    __tablename__ = "attendances"
    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id"))
    class_session_id = Column(Integer, ForeignKey("class_sessions.id"))

    em_aula = Column(Boolean, default=False)  # 1 = Na sala, 0 = Fora
    tempo_em_aula = Column(Integer, default=0)  # Em minutos
    status_final = Column(String, default="pending")  # present, absent, pending
    last_event_at = Column(DateTime)  # Para controle de cooldown

    student = relationship("Student", back_populates="attendances")
    class_session = relationship("ClassSession", back_populates="attendances")
    intervals = relationship("PresenceInterval", back_populates="attendance")


class PresenceInterval(Base):
    __tablename__ = "presence_intervals"
    id = Column(Integer, primary_key=True, index=True)
    attendance_id = Column(Integer, ForeignKey("attendances.id"))
    entered_at = Column(DateTime)
    exited_at = Column(DateTime, nullable=True)

    attendance = relationship("Attendance", back_populates="intervals")