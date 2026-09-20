import numpy as np
import json
from datetime import timezone, datetime  # <-- Adicionado o datetime aqui
from sqlalchemy.orm import Session
from data.models import Student, FaceEmbedding, ClassSession, Attendance, PresenceInterval

COOLDOWN_SECONDS = 15
SIMILARITY_THRESHOLD = 0.7

# ... resto do código continua igual ...


def cosine_similarity(vec1, vec2):
    v1, v2 = np.array(vec1), np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def match_face(db: Session, target_embedding: list):
    embeddings = db.query(FaceEmbedding).all()
    best_match, highest_sim = None, 0.0

    for emb in embeddings:
        saved_vec = json.loads(emb.embedding_json)
        sim = cosine_similarity(target_embedding, saved_vec)
        if sim > highest_sim and sim >= SIMILARITY_THRESHOLD:
            highest_sim = sim
            best_match = emb.student_id

    return best_match


def process_recognition(db: Session, class_id: int, embedding: list, timestamp: datetime):
    student_id = match_face(db, embedding)
    if not student_id:
        return {"status": "unregistered", "message": "Rosto não reconhecido."}

    session_info = db.query(ClassSession).filter(ClassSession.id == class_id).first()
    if not session_info or session_info.is_closed:
        return {"status": "error", "message": "Aula não encontrada ou já encerrada."}

    # Busca ou cria o registro de frequência (Attendance)
    attendance = db.query(Attendance).filter_by(student_id=student_id, class_session_id=class_id).first()

    if not attendance:
        attendance = Attendance(
            student_id=student_id,
            class_session_id=class_id,
            em_aula=False,
            tempo_em_aula=0
        )
        db.add(attendance)
        db.commit()
        db.refresh(attendance)

    # Cooldown (Debounce)
    if attendance.last_event_at:
        diff = (timestamp - attendance.last_event_at.replace(tzinfo=timezone.utc)).total_seconds()
        if diff < COOLDOWN_SECONDS:
            return {"status": "cooldown", "message": "Reconhecimento ignorado (cooldown)."}

    attendance.last_event_at = timestamp

    if not attendance.em_aula:
        # ALUNO ENTRANDO
        attendance.em_aula = True
        interval = PresenceInterval(attendance_id=attendance.id, entered_at=timestamp)
        db.add(interval)
        msg = "Entrada registrada."
    else:
        # ALUNO SAINDO
        attendance.em_aula = False
        interval = db.query(PresenceInterval).filter_by(
            attendance_id=attendance.id, exited_at=None
        ).order_by(PresenceInterval.entered_at.desc()).first()

        if interval:
            interval.exited_at = timestamp

            # Calcula o tempo efetivo apenas dentro do horário oficial da aula
            calc_start = max(interval.entered_at, session_info.starts_at)
            calc_end = min(interval.exited_at, session_info.ends_at)

            if calc_end > calc_start:
                minutes_added = (calc_end - calc_start).total_seconds() / 60.0
                attendance.tempo_em_aula += int(minutes_added)
        msg = "Saída registrada."

    db.commit()
    return {"status": "success", "message": msg, "em_aula": attendance.em_aula,
            "tempo_em_aula": attendance.tempo_em_aula}


def fechar_aula(db: Session, class_id: int):
    session_info = db.query(ClassSession).filter(ClassSession.id == class_id).first()
    if not session_info: return False

    attendances = db.query(Attendance).filter_by(class_session_id=class_id).all()
    duracao_total = (session_info.ends_at - session_info.starts_at).total_seconds() / 60.0
    minutos_exigidos = duracao_total * session_info.min_presence_percent

    for att in attendances:
        # Se esqueceu de sair (ainda em_aula = True), fecha no horário atual ou fim da aula
        if att.em_aula:
            interval = db.query(PresenceInterval).filter_by(attendance_id=att.id, exited_at=None).first()
            if interval:
                interval.exited_at = session_info.ends_at
                calc_start = max(interval.entered_at, session_info.starts_at)
                if session_info.ends_at > calc_start:
                    att.tempo_em_aula += int((session_info.ends_at - calc_start).total_seconds() / 60.0)
            att.em_aula = False

        # Define status final
        att.status_final = "present" if att.tempo_em_aula >= minutos_exigidos else "absent"

    session_info.is_closed = True
    db.commit()
    return True