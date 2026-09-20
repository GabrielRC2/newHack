from __future__ import annotations

from datetime import datetime, time
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .database import Base, engine, get_db
from .models import Attendance, ClassSession, FaceEmbedding, PresenceInterval, RecognitionEvent, Student
from .schemas import (
    AttendanceOut, ClassSessionCreate, ClassSessionOut, EmbeddingCreate, FinalizeOut,
    IntervalOut, RecognitionOut, RecognitionRequest, ScheduledSessionCreate, StudentCreate, StudentOut,
    StudentSummaryOut,
)
from .services import COOLDOWN_SECONDS, MATCH_THRESHOLD, find_student_by_embedding, get_or_create_attendance, last_state_change, total_present_seconds

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Frequencia por Visao - MVP", version="1.0.0")

# Horarios oficiais informados. Indices sao usados pelo endpoint /sessions/scheduled.
SCHOOL_SCHEDULE = [
    (time(7, 55), time(8, 50)), (time(8, 50), time(10, 10)),
    (time(10, 10), time(12, 0)), (time(13, 30), time(15, 20)),
    (time(15, 20), time(17, 35)), (time(19, 0), time(21, 50)),
]


def get_session_or_404(db: Session, class_id: int) -> ClassSession:
    session = db.get(ClassSession, class_id)
    if not session:
        raise HTTPException(status_code=404, detail="Aula nao encontrada")
    return session


@app.get("/health")
def health():
    # Mostra qual banco esta em uso: util quando ha mais de um .db no projeto.
    database = engine.url.database
    if database and engine.url.drivername.startswith("sqlite"):
        database = Path(database).resolve().as_posix()
    return {
        "status": "ok",
        "database": database or "memoria",
        "match_threshold": MATCH_THRESHOLD,
        "cooldown_seconds": COOLDOWN_SECONDS,
    }


@app.get("/students", response_model=list[StudentSummaryOut])
def list_students(db: Session = Depends(get_db)):
    rows = db.execute(
        select(Student, func.count(FaceEmbedding.id))
        .outerjoin(FaceEmbedding, FaceEmbedding.student_id == Student.id)
        .group_by(Student.id)
        .order_by(Student.name)
    ).all()
    return [
        StudentSummaryOut(id=s.id, name=s.name, enrollment_number=s.enrollment_number,
                          created_at=s.created_at, embeddings_count=count)
        for s, count in rows
    ]


@app.post("/students", response_model=StudentOut, status_code=status.HTTP_201_CREATED)
def create_student(payload: StudentCreate, db: Session = Depends(get_db)):
    if db.scalar(select(Student).where(Student.enrollment_number == payload.enrollment_number)):
        raise HTTPException(status_code=409, detail="Matricula ja cadastrada")
    student = Student(name=payload.name, enrollment_number=payload.enrollment_number, em_aula=True)
    db.add(student)
    db.flush()
    db.add(FaceEmbedding(student_id=student.id, vector=payload.embedding))
    db.commit()
    db.refresh(student)
    return student


@app.post("/students/{student_id}/embeddings", status_code=status.HTTP_201_CREATED)
def add_embedding(student_id: int, payload: EmbeddingCreate, db: Session = Depends(get_db)):
    if not db.get(Student, student_id):
        raise HTTPException(status_code=404, detail="Aluno nao encontrado")
    embedding = FaceEmbedding(student_id=student_id, vector=payload.vector)
    db.add(embedding)
    db.commit()
    return {"id": embedding.id, "student_id": student_id, "message": "Embedding cadastrado"}


@app.post("/sessions", response_model=ClassSessionOut, status_code=status.HTTP_201_CREATED)
def create_session(payload: ClassSessionCreate, db: Session = Depends(get_db)):
    if payload.ends_at <= payload.starts_at:
        raise HTTPException(status_code=422, detail="ends_at deve ser posterior a starts_at")
    session = ClassSession(**payload.model_dump())
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@app.get("/sessions", response_model=list[ClassSessionOut])
def list_sessions(limit: int = 50, db: Session = Depends(get_db)):
    return db.scalars(
        select(ClassSession).order_by(ClassSession.starts_at.desc(), ClassSession.id.desc()).limit(limit)
    ).all()


@app.get("/sessions/{class_id}", response_model=ClassSessionOut)
def get_session(class_id: int, db: Session = Depends(get_db)):
    return get_session_or_404(db, class_id)


@app.post("/sessions/scheduled", response_model=ClassSessionOut, status_code=status.HTTP_201_CREATED)
def create_scheduled_session(payload: ScheduledSessionCreate, db: Session = Depends(get_db)):
    start, end = SCHOOL_SCHEDULE[payload.schedule_index - 1]
    session = ClassSession(
        label=payload.label or f"Aula {payload.schedule_index}",
        starts_at=datetime.combine(payload.class_date, start),
        ends_at=datetime.combine(payload.class_date, end),
        minimum_percentage=payload.minimum_percentage,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@app.post("/recognitions", response_model=RecognitionOut)
def receive_recognition(payload: RecognitionRequest, db: Session = Depends(get_db)):
    session = get_session_or_404(db, payload.class_id)
    if session.status == "closed":
        event = RecognitionEvent(class_session_id=session.id, camera_id=payload.camera_id,
                                 detected_at=payload.timestamp, status="class_closed")
        db.add(event)
        db.commit()
        return RecognitionOut(event_id=event.id, status="class_closed", message="Aula ja foi encerrada")

    student, score = find_student_by_embedding(db, payload.embedding)
    if student is None:
        event = RecognitionEvent(class_session_id=session.id, camera_id=payload.camera_id,
                                 detected_at=payload.timestamp, status="unregistered", match_score=score)
        db.add(event)
        db.commit()
        return RecognitionOut(event_id=event.id, status="unregistered", match_score=score,
                              message="Rosto nao cadastrado; nenhuma presenca foi criada")

    previous = last_state_change(db, session.id, student.id)
    if previous:
        elapsed = (payload.timestamp - previous.detected_at).total_seconds()
        if elapsed < 0:
            event = RecognitionEvent(class_session_id=session.id, student_id=student.id, camera_id=payload.camera_id,
                                     detected_at=payload.timestamp, status="invalid_timestamp", match_score=score)
            db.add(event)
            db.commit()
            return RecognitionOut(event_id=event.id, status="invalid_timestamp", student_id=student.id,
                                  match_score=score, message="Evento anterior ao ultimo reconhecimento aceito")
        if elapsed < COOLDOWN_SECONDS:
            event = RecognitionEvent(class_session_id=session.id, student_id=student.id, camera_id=payload.camera_id,
                                     detected_at=payload.timestamp, status="cooldown", match_score=score)
            db.add(event)
            db.commit()
            return RecognitionOut(event_id=event.id, status="cooldown", student_id=student.id,
                                  match_score=score, message=f"Ignorado pelo cooldown de {COOLDOWN_SECONDS}s")

    open_interval = db.scalar(select(PresenceInterval).where(
        PresenceInterval.class_session_id == session.id,
        PresenceInterval.student_id == student.id,
        PresenceInterval.exited_at.is_(None),
    ).order_by(PresenceInterval.entered_at.desc()))
    if open_interval:
        open_interval.exited_at = payload.timestamp
        action = "exited"
        student.em_aula = False
    else:
        had_interval = db.scalar(select(PresenceInterval.id).where(
            PresenceInterval.class_session_id == session.id,
            PresenceInterval.student_id == student.id,
        )) is not None
        db.add(PresenceInterval(class_session_id=session.id, student_id=student.id, entered_at=payload.timestamp))
        action = "returned" if had_interval else "entered"
        student.em_aula = True

    attendance = get_or_create_attendance(db, session.id, student.id)
    # Inclui eventual intervalo aberto ate o instante detectado para o painel em tempo real.
    attendance.total_seconds = total_present_seconds(db, session.id, student.id, session.starts_at, session.ends_at, payload.timestamp)
    attendance.required_seconds = int((session.ends_at - session.starts_at).total_seconds() * session.minimum_percentage)
    student.tempo_em_aula = attendance.total_seconds
    event = RecognitionEvent(class_session_id=session.id, student_id=student.id, camera_id=payload.camera_id,
                             detected_at=payload.timestamp, status="identified", action=action, match_score=score)
    db.add(event)
    db.commit()
    return RecognitionOut(event_id=event.id, status="identified", action=action, student_id=student.id,
                          match_score=score, message="Entrada/saida registrada pelo estado atual do aluno")


def attendance_view(db: Session, session: ClassSession, student: Student) -> AttendanceOut:
    record = get_or_create_attendance(db, session.id, student.id)
    now_or_end = session.ends_at if session.status == "closed" else datetime.now()
    total = total_present_seconds(db, session.id, student.id, session.starts_at, session.ends_at, now_or_end)
    required = int((session.ends_at - session.starts_at).total_seconds() * session.minimum_percentage)
    intervals = db.scalars(select(PresenceInterval).where(
        PresenceInterval.class_session_id == session.id, PresenceInterval.student_id == student.id,
    ).order_by(PresenceInterval.entered_at)).all()
    open_now = any(item.exited_at is None for item in intervals)
    return AttendanceOut(student_id=student.id, student_name=student.name, enrollment_number=student.enrollment_number,
                         total_seconds=total, total_minutes=round(total / 60, 2), required_seconds=required,
                         status=record.status, em_aula=open_now, intervals=[IntervalOut(entered_at=i.entered_at, exited_at=i.exited_at) for i in intervals])


@app.get("/sessions/{class_id}/attendance", response_model=list[AttendanceOut])
def list_attendance(class_id: int, db: Session = Depends(get_db)):
    session = get_session_or_404(db, class_id)
    student_ids = db.scalars(select(Attendance.student_id).where(Attendance.class_session_id == session.id)).all()
    students = db.scalars(select(Student).where(Student.id.in_(student_ids))).all() if student_ids else []
    result = [attendance_view(db, session, student) for student in students]
    db.commit()
    return result


@app.get("/sessions/{class_id}/students/{student_id}/attendance", response_model=AttendanceOut)
def student_attendance(class_id: int, student_id: int, db: Session = Depends(get_db)):
    session = get_session_or_404(db, class_id)
    student = db.get(Student, student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Aluno nao encontrado")
    response = attendance_view(db, session, student)
    db.commit()
    return response


@app.post("/sessions/{class_id}/finalize", response_model=FinalizeOut)
def finalize_session(class_id: int, db: Session = Depends(get_db)):
    session = get_session_or_404(db, class_id)
    if session.status == "closed":
        records = db.scalars(select(Attendance).where(Attendance.class_session_id == session.id)).all()
        return FinalizeOut(class_id=session.id, finalized_students=len(records),
                           present=sum(r.status == "present" for r in records), absent=sum(r.status == "absent" for r in records))

    # Como a duracao oficial e o limite contabil, um intervalo aberto fecha em ends_at.
    open_intervals = db.scalars(select(PresenceInterval).where(
        PresenceInterval.class_session_id == session.id, PresenceInterval.exited_at.is_(None)
    )).all()
    for interval in open_intervals:
        interval.exited_at = max(interval.entered_at, session.ends_at)

    required = int((session.ends_at - session.starts_at).total_seconds() * session.minimum_percentage)
    students = db.scalars(select(Student)).all()
    present = absent = 0
    for student in students:
        record = get_or_create_attendance(db, session.id, student.id)
        total = total_present_seconds(db, session.id, student.id, session.starts_at, session.ends_at)
        record.total_seconds, record.required_seconds = total, required
        record.status = "present" if total >= required else "absent"
        record.finalized_at = session.ends_at
        student.tempo_em_aula = total
        if any(i.student_id == student.id for i in open_intervals):
            student.em_aula = False
        if record.status == "present":
            present += 1
        else:
            absent += 1
    session.status, session.closed_at = "closed", session.ends_at
    db.commit()
    return FinalizeOut(class_id=session.id, finalized_students=len(students), present=present, absent=absent)