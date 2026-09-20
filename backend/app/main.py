import json
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from app import services
from data import models, schemas
from data.database import engine, get_db

models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="MVP Frequência Visão Computacional")


@app.post("/alunos")
def cadastrar_aluno(aluno: schemas.StudentCreate, db: Session = Depends(get_db)):
    db_aluno = models.Student(nome=aluno.nome, matricula=aluno.matricula)
    db.add(db_aluno)
    db.commit()
    db.refresh(db_aluno)

    emb = models.FaceEmbedding(student_id=db_aluno.id, embedding_json=json.dumps(aluno.embedding))
    db.add(emb)
    db.commit()
    return {"id": db_aluno.id, "nome": db_aluno.nome}


@app.post("/aulas")
def criar_aula(aula: schemas.ClassSessionCreate, db: Session = Depends(get_db)):
    db_aula = models.ClassSession(**aula.dict())
    db.add(db_aula)
    db.commit()
    db.refresh(db_aula)
    return db_aula


@app.post("/reconhecimento")
def receber_reconhecimento(payload: schemas.RecognitionPayload, db: Session = Depends(get_db)):
    result = services.process_recognition(db, payload.class_id, payload.embedding, payload.timestamp)
    return result


@app.post("/aulas/{class_id}/fechar")
def fechar_sessao_aula(class_id: int, db: Session = Depends(get_db)):
    sucesso = services.fechar_aula(db, class_id)
    if not sucesso: raise HTTPException(status_code=404, detail="Aula não encontrada")
    return {"status": "Aula encerrada e presenças calculadas."}


@app.get("/aulas/{class_id}/presencas")
def listar_presencas(class_id: int, db: Session = Depends(get_db)):
    res = db.query(models.Attendance).filter(models.Attendance.class_session_id == class_id).all()
    return [{"aluno_id": r.student_id, "em_aula": r.em_aula, "tempo_em_aula": r.tempo_em_aula, "status": r.status_final}
            for r in res]