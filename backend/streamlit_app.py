import os
from datetime import date, datetime

import numpy as np
import requests
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Frequencia por Visao", page_icon="🎓", layout="wide")
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Modelo usado para gerar os embeddings. Precisa ser SEMPRE o mesmo no cadastro e no reconhecimento.
FACE_MODEL = os.getenv("FACE_MODEL", "ArcFace")  # 512 dimensoes
FACE_DETECTOR = os.getenv("FACE_DETECTOR", "retinaface")  # o "opencv" (Haar) falha com facilidade em fotos de webcam


def call(method: str, path: str, **kwargs):
    try:
        response = requests.request(method, f"{API_URL}{path}", timeout=8, **kwargs)
        if response.ok:
            return response.json(), None
        return None, response.json().get("detail", response.text)
    except requests.RequestException as exc:
        return None, f"API indisponivel: {exc}"


@st.cache_resource(show_spinner="Carregando modelo facial (só na primeira vez)...")
def load_face_model():
    """Carrega o modelo uma unica vez e mantem em memoria."""
    from deepface import DeepFace

    DeepFace.build_model(FACE_MODEL)
    return DeepFace


def extract_embeddings(photo) -> list[list[float]]:
    """Recebe a foto do st.camera_input e devolve um embedding por rosto encontrado."""
    deepface = load_face_model()
    rgb = np.array(Image.open(photo).convert("RGB"))
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])  # DeepFace espera BGR quando recebe um array
    try:
        faces = deepface.represent(
            img_path=bgr,
            model_name=FACE_MODEL,
            detector_backend=FACE_DETECTOR,
            enforce_detection=True,
        )
    except ValueError as exc:
        if "could not be detected" in str(exc):  # nenhum rosto detectado
            return []
        raise  # qualquer outro erro aparece na tela em vez de virar "nenhum rosto"
    return [face["embedding"] for face in faces]


st.title("🎓 Controle de Frequência")
st.caption("MVP com uma câmera: cada reconhecimento alterna entrada e saída após o cooldown.")

tab_student, tab_session, tab_event, tab_report = st.tabs(["Aluno", "Aula", "Reconhecimento", "Relatório"])

# ----------------------------------------------------------------------------- ALUNO
with tab_student:
    st.subheader("Cadastrar aluno")
    name = st.text_input("Nome")
    enrollment = st.text_input("Matrícula")
    photo = st.camera_input("Tire uma foto do rosto (de frente, com boa iluminação)", key="cam_register")

    if st.button("Cadastrar aluno", type="primary"):
        if not name or not enrollment:
            st.error("Preencha nome e matrícula.")
        elif photo is None:
            st.error("Tire a foto antes de cadastrar.")
        else:
            with st.spinner("Analisando o rosto..."):
                embeddings = extract_embeddings(photo)
            if len(embeddings) == 0:
                st.error("Nenhum rosto encontrado. Tente com mais luz e olhando para a câmera.")
            elif len(embeddings) > 1:
                st.error("Há mais de um rosto na foto. Tire a foto só do aluno.")
            else:
                result, error = call(
                    "POST", "/students",
                    json={"name": name, "enrollment_number": enrollment, "embedding": embeddings[0]},
                )
                if result:
                    st.success(f"Aluno cadastrado: #{result['id']}")
                else:
                    st.error(error)

    st.divider()
    st.subheader("Adicionar mais uma foto a um aluno")
    st.caption("Várias fotos (ângulos e iluminações diferentes) melhoram a precisão do reconhecimento.")
    student_id = st.number_input("ID do aluno", min_value=1, step=1, key="extra_student")
    extra_photo = st.camera_input("Nova foto", key="cam_extra")

    if st.button("Adicionar foto"):
        if extra_photo is None:
            st.error("Tire a foto primeiro.")
        else:
            with st.spinner("Analisando o rosto..."):
                embeddings = extract_embeddings(extra_photo)
            if len(embeddings) != 1:
                st.error("A foto precisa ter exatamente um rosto.")
            else:
                result, error = call(
                    "POST", f"/students/{int(student_id)}/embeddings",
                    json={"vector": embeddings[0]},
                )
                st.success(result["message"]) if result else st.error(error)

# ----------------------------------------------------------------------------- AULA
with tab_session:
    st.write("Use um dos seis horários oficiais já configurados ou cadastre um horário manual.")
    with st.form("scheduled"):
        class_date = st.date_input("Data da aula", value=date.today())
        slot = st.selectbox("Horário", range(1, 7), format_func=lambda i: ["07:55–08:50", "08:50–10:10", "10:10–12:00", "13:30–15:20", "15:20–17:35", "19:00–21:50"][i-1])
        label = st.text_input("Turma/descrição", "Turma A")
        pct = st.number_input("Presença mínima", min_value=0.01, max_value=1.0, value=0.75, step=0.01)
        create = st.form_submit_button("Criar aula")
    if create:
        result, error = call("POST", "/sessions/scheduled", json={"class_date": class_date.isoformat(), "schedule_index": slot, "label": label, "minimum_percentage": pct})
        st.success(f"Aula criada com ID {result['id']}") if result else st.error(error)

# ----------------------------------------------------------------------------- RECONHECIMENTO
with tab_event:
    class_id = st.number_input("ID da aula", min_value=1, step=1)
    camera = st.text_input("ID da câmera", "camera-entrada")
    detected_photo = st.camera_input("Tire uma foto para reconhecer", key="cam_recognize")

    # Botao explicito evita enviar o mesmo reconhecimento de novo a cada rerun do Streamlit.
    if st.button("Enviar reconhecimento", type="primary"):
        if detected_photo is None:
            st.error("Tire a foto antes de enviar.")
        else:
            with st.spinner("Analisando o rosto..."):
                embeddings = extract_embeddings(detected_photo)
            if not embeddings:
                st.warning("Nenhum rosto detectado.")
            for i, embedding in enumerate(embeddings, start=1):
                result, error = call(
                    "POST", "/recognitions",
                    json={
                        "class_id": int(class_id),
                        "camera_id": camera,
                        "embedding": embedding,
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                    },
                )
                if result:
                    st.success(f"Rosto {i}: {result['message']}")
                    st.json(result)
                else:
                    st.error(f"Rosto {i}: {error}")

# ----------------------------------------------------------------------------- RELATORIO
with tab_report:
    report_id = st.number_input("ID da aula para consultar", min_value=1, step=1, key="report")
    col1, col2 = st.columns(2)
    if col1.button("Atualizar relatório"):
        result, error = call("GET", f"/sessions/{int(report_id)}/attendance")
        if result is not None:
            st.dataframe([{k: row[k] for k in ("student_id", "student_name", "total_minutes", "status", "em_aula")} for row in result], use_container_width=True)
        else: st.error(error)
    if col2.button("Finalizar aula e calcular presença"):
        result, error = call("POST", f"/sessions/{int(report_id)}/finalize")
        st.success(result) if result else st.error(error)