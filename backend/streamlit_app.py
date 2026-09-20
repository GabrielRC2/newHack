import os
from datetime import date, datetime, time

import requests
import streamlit as st

st.set_page_config(page_title="Frequencia por Visao", page_icon="🎓", layout="wide")
API_URL = os.getenv("API_URL", "http://localhost:8000")


def call(method: str, path: str, **kwargs):
    try:
        response = requests.request(method, f"{API_URL}{path}", timeout=8, **kwargs)
        if response.ok:
            return response.json(), None
        return None, response.json().get("detail", response.text)
    except requests.RequestException as exc:
        return None, f"API indisponivel: {exc}"


st.title("🎓 Controle de Frequência")
st.caption("MVP com uma câmera: cada reconhecimento alterna entrada e saída após o cooldown.")

tab_student, tab_session, tab_event, tab_report = st.tabs(["Aluno", "Aula", "Reconhecimento", "Relatório"])

with tab_student:
    with st.form("student"):
        name = st.text_input("Nome")
        enrollment = st.text_input("Matrícula")
        vector = st.text_area("Embedding facial (números separados por vírgula)", "1, 0, 0")
        submitted = st.form_submit_button("Cadastrar aluno")
    if submitted:
        try:
            embedding = [float(item.strip()) for item in vector.split(",")]
            result, error = call("POST", "/students", json={"name": name, "enrollment_number": enrollment, "embedding": embedding})
            st.success(f"Aluno cadastrado: #{result['id']}") if result else st.error(error)
        except ValueError:
            st.error("O embedding deve conter somente números separados por vírgula.")

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

with tab_event:
    with st.form("recognition"):
        class_id = st.number_input("ID da aula", min_value=1, step=1)
        camera = st.text_input("ID da câmera", "camera-entrada")
        vector = st.text_area("Embedding detectado", "1, 0, 0", key="detected")
        moment = st.text_input("Timestamp ISO (vazio = agora)", "")
        send = st.form_submit_button("Enviar reconhecimento")
    if send:
        try:
            timestamp = moment or datetime.now().isoformat(timespec="seconds")
            result, error = call("POST", "/recognitions", json={"class_id": int(class_id), "camera_id": camera, "embedding": [float(v.strip()) for v in vector.split(",")], "timestamp": timestamp})
            st.success(result["message"]) if result else st.error(error)
            if result: st.json(result)
        except ValueError:
            st.error("Confira o embedding e o timestamp.")

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
