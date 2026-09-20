"""Painel Streamlit do sistema de frequencia por reconhecimento facial.

Executar na raiz do projeto (a pasta que contem `vision_client/`):

    streamlit run streamlit_app.py

Nenhuma foto e gravada em disco: a imagem da webcam fica so na memoria da sessao,
e apenas o embedding (vetor numerico) e enviado para a API.
"""
from __future__ import annotations

import html
import os
from datetime import date, datetime, time, timedelta, timezone

import cv2 as cv
import numpy as np
import pandas as pd
import requests
import streamlit as st

from vision_client.api import FrequencyApi
from vision_client.face_engine import FaceEngine, draw_face
from vision_client.run_camera import DEFAULT_DETECTOR, DEFAULT_RECOGNIZER

st.set_page_config(page_title="Frequência por reconhecimento facial", page_icon="🎓", layout="wide")

SCHEDULE = [
    "07:55 às 08:50",
    "08:50 às 10:10",
    "10:10 às 12:00",
    "13:30 às 15:20",
    "15:20 às 17:35",
    "19:00 às 21:50",
]
ACTIONS = {
    "entered": "Entrada registrada",
    "exited": "Saída registrada",
    "returned": "Retorno registrado",
}

CSS = """
<style>
h1, h2, h3 { letter-spacing: -0.01em; }
.block-container { padding-top: 2.2rem; max-width: 1200px; }

.result-card {
    border-radius: 14px; padding: 1.4rem 1.6rem; margin-bottom: .8rem;
    border: 1px solid rgba(30,41,59,.12); border-left-width: 8px;
    background: #fff;
}
.result-card .title { font-size: 1.7rem; font-weight: 700; line-height: 1.2; }
.result-card .detail { margin-top: .35rem; font-size: 1.02rem; color: #475569; }
.result-card.success { border-left-color: #1F8A5B; }
.result-card.info    { border-left-color: #3B4CCA; }
.result-card.warning { border-left-color: #C77800; }
.result-card.error   { border-left-color: #B3261E; }
.result-card.idle    { border-left-color: #94A3B8; }

.app-subtitle { color: #64748B; margin-top: -.6rem; margin-bottom: 1rem; }
</style>
"""


# --------------------------------------------------------------------------- estado

def init_state() -> None:
    defaults = {
        "api_url": os.getenv("FREQUENCY_API_URL", os.getenv("API_URL", "http://127.0.0.1:8000")),
        "api_token": os.getenv("FREQUENCY_API_TOKEN", ""),
        "camera_id": "web-streamlit-01",
        "active_class": None,
        "flash": None,
        "rec_n": 0,
        "rec_result": None,
        "rec_preview": None,
        "rec_log": [],
        "enroll_n": 0,
        "enroll_samples": [],
        "enroll_thumbs": [],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def flash(message: str) -> None:
    """Mensagem que sobrevive ao proximo st.rerun()."""
    st.session_state.flash = message


# --------------------------------------------------------------------------- API

def api_client() -> FrequencyApi:
    return FrequencyApi(st.session_state.api_url, st.session_state.api_token or None)


def request(method: str, path: str, **kwargs):
    """Retorna (dados, erro). Somente um dos dois vem preenchido."""
    api = api_client()
    try:
        response = api.session.request(method, f"{api.base_url}{path}", timeout=8, **kwargs)
    except requests.RequestException as exc:
        return None, f"API indisponível: {exc}"
    try:
        body = response.json()
    except ValueError:
        body = {"detail": response.text}
    if not response.ok:
        detail = body.get("detail", body) if isinstance(body, dict) else body
        return None, f"HTTP {response.status_code}: {detail}"
    return body, None


@st.cache_data(ttl=5, show_spinner=False)
def get_health(url: str) -> dict | None:
    """Resposta de /health (inclui banco em uso e limite de similaridade) ou None se a API nao responde."""
    try:
        response = requests.get(f"{url.rstrip('/')}/health", timeout=3)
        return response.json() if response.ok else None
    except (requests.RequestException, ValueError):
        return None


def match_threshold() -> float | None:
    health = get_health(st.session_state.api_url)
    return health.get("match_threshold") if health else None


def fetch_sessions() -> list[dict] | None:
    """None quando o backend ainda nao tem GET /sessions (veja o patch no final da resposta)."""
    data, _ = request("GET", "/sessions")
    return data if isinstance(data, list) else None


def fetch_students() -> list[dict] | None:
    data, _ = request("GET", "/students")
    return data if isinstance(data, list) else None


def fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours} h {minutes:02d} min"
    if minutes:
        return f"{minutes} min {secs:02d} s"
    return f"{secs} s"


def fmt_dt(value: str | None, with_date: bool = False) -> str:
    if not value:
        return "-"
    return datetime.fromisoformat(value).strftime("%d/%m %H:%M:%S" if with_date else "%H:%M:%S")


def fmt_utc_local(value: str) -> str:
    """created_at e gravado em UTC pelo backend; mostra no fuso deste computador."""
    moment = datetime.fromisoformat(value).replace(tzinfo=timezone.utc).astimezone()
    return moment.strftime("%d/%m/%Y %H:%M")


def counted_seconds(interval: dict, session: dict | None, now: datetime) -> int:
    """Mesma regra do backend: so conta o tempo dentro do horario da aula."""
    entered = datetime.fromisoformat(interval["entered_at"])
    left = datetime.fromisoformat(interval["exited_at"]) if interval["exited_at"] else now
    if session:
        entered = max(entered, datetime.fromisoformat(session["starts_at"]))
        left = min(left, datetime.fromisoformat(session["ends_at"]))
    return max(0, int((left - entered).total_seconds()))


def session_label(item: dict) -> str:
    start = datetime.fromisoformat(item["starts_at"])
    end = datetime.fromisoformat(item["ends_at"])
    state = "aberta" if item["status"] == "open" else "encerrada"
    return f"#{item['id']} {item['label']}, {start:%d/%m %H:%M} às {end:%H:%M} ({state})"


# --------------------------------------------------------------------------- visao

@st.cache_resource(show_spinner="Carregando modelos faciais…")
def load_engine() -> FaceEngine:
    return FaceEngine(DEFAULT_DETECTOR, DEFAULT_RECOGNIZER)


def engine_or_stop() -> FaceEngine | None:
    try:
        return load_engine()
    except (FileNotFoundError, RuntimeError) as exc:
        st.error(f"{exc}")
        return None


def analyse_photo(upload) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """Decodifica a foto, detecta rostos e devolve (frame BGR, rostos, preview RGB anotado)."""
    frame = cv.imdecode(np.frombuffer(upload.getvalue(), np.uint8), cv.IMREAD_COLOR)
    faces = load_engine().detect(frame)
    annotated = frame.copy()
    color = (0, 210, 0) if len(faces) == 1 else (0, 0, 255)
    for face in faces:
        draw_face(annotated, face, color)
    return frame, faces, cv.cvtColor(annotated, cv.COLOR_BGR2RGB)


def face_crop(frame: np.ndarray, face: np.ndarray, margin: float = 0.25) -> np.ndarray:
    x, y, w, h = (int(v) for v in face[:4])
    mx, my = int(w * margin), int(h * margin)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(frame.shape[1], x + w + mx), min(frame.shape[0], y + h + my)
    return cv.cvtColor(frame[y0:y1, x0:x1], cv.COLOR_BGR2RGB)


# --------------------------------------------------------------------------- componentes

def result_card(kind: str, title: str, detail: str = "") -> None:
    st.markdown(
        f'<div class="result-card {kind}"><div class="title">{html.escape(title)}</div>'
        f'<div class="detail">{html.escape(detail)}</div></div>',
        unsafe_allow_html=True,
    )


def sidebar() -> tuple[int | None, list[dict] | None]:
    with st.sidebar:
        st.header("Conexão")
        st.text_input("Endereço da API", key="api_url")
        st.text_input("Token (opcional)", key="api_token", type="password")
        st.text_input("ID desta câmera", key="camera_id")
        health = get_health(st.session_state.api_url)
        if health:
            st.success("API conectada")
            st.caption(f"Banco em uso: {health.get('database', 'desconhecido')}")
        else:
            st.error("API fora do ar. Confira o endereço e se o backend está rodando.")

        st.header("Aula ativa")
        sessions = fetch_sessions()
        active = st.session_state.active_class
        if sessions:
            by_id = {item["id"]: item for item in sessions}
            ids = list(by_id)
            index = ids.index(active) if active in ids else 0
            chosen = st.selectbox("Aula", ids, index=index, format_func=lambda i: session_label(by_id[i]))
        else:
            chosen = int(st.number_input("ID da aula", min_value=1, step=1, value=active or 1))
            if sessions is None:
                st.caption("Este backend ainda não lista aulas. Crie uma na aba Aulas ou digite o ID.")
            else:
                st.caption("Nenhuma aula criada ainda. Use a aba Aulas.")
                chosen = None if active is None else chosen
        st.session_state.active_class = chosen
        return chosen, sessions


def class_status_banner(class_id: int, sessions: list[dict] | None, where: str) -> bool:
    """Avisa quando a aula esta encerrada ou fora do horario. Retorna False se nao aceita reconhecimentos."""
    item = next((s for s in (sessions or []) if s["id"] == class_id), None)
    if item is None:
        return True
    start = datetime.fromisoformat(item["starts_at"])
    end = datetime.fromisoformat(item["ends_at"])
    now = datetime.now()
    if item["status"] == "closed":
        if where == "recognize":
            st.error("Esta aula já foi encerrada e recusa novos reconhecimentos. "
                     "Crie ou selecione outra aula na aba Aulas.")
            return False
        st.info("Esta aula foi encerrada. Os resultados abaixo são finais.")
        return True
    if not start <= now <= end:
        st.warning(f"Esta aula é das {start:%H:%M} às {end:%H:%M} e agora são {now:%H:%M}. "
                   "O reconhecimento é aceito, mas só o tempo dentro do horário da aula conta: "
                   "quem entrar fora dele termina com 0 min e ausente. "
                   "Para testar agora, crie uma aula na aba Aulas.")
    return True


# --------------------------------------------------------------------------- aba: painel

def tab_dashboard(class_id: int | None, sessions: list[dict] | None) -> None:
    if class_id is None:
        st.info("Crie ou selecione uma aula para acompanhar a presença.")
        return

    session, _ = request("GET", f"/sessions/{class_id}")
    class_status_banner(class_id, [session] if session else sessions, "dashboard")
    if session:
        start = datetime.fromisoformat(session["starts_at"])
        end = datetime.fromisoformat(session["ends_at"])
        st.caption(f"Aula #{class_id}: {session['label']}, {start:%d/%m} das {start:%H:%M} às {end:%H:%M}. "
                   f"Meta de presença: {session['minimum_percentage']:.0%} do horário da aula.")

    top = st.columns([3, 1.4, 1])
    auto = top[1].toggle("Atualizar a cada 5 segundos", value=False)
    top[2].button("Atualizar agora")

    def body() -> None:
        attendance, error = request("GET", f"/sessions/{class_id}/attendance")
        if error:
            st.error(error)
            return
        students = fetch_students()
        records = {r["student_id"]: r for r in attendance}
        if students is None:
            st.warning("Este backend ainda não lista os alunos cadastrados, então só aparecem quem já foi reconhecido. "
                       "Atualize `app/main.py` e `app/schemas.py` com os arquivos novos.")
            people = [{"id": r["student_id"], "name": r["student_name"], "enrollment_number": r["enrollment_number"]}
                      for r in attendance]
        else:
            people = students
        if not people:
            st.info("Nenhum aluno cadastrado ainda. Use a aba Cadastrar aluno.")
            return

        now = datetime.now()
        required = (int((datetime.fromisoformat(session["ends_at"]) - datetime.fromisoformat(session["starts_at"])).total_seconds()
                        * session["minimum_percentage"]) if session else 0)
        table, timeline = [], []
        for person in people:
            record = records.get(person["id"])
            intervals = record["intervals"] if record else []
            counted = record["total_seconds"] if record else 0
            required_here = record["required_seconds"] if record and record["required_seconds"] else required

            if not intervals:
                state = "Ainda não reconhecido"
            elif record["em_aula"]:
                state = "Na sala"
            else:
                state = "Saiu"

            status = record["status"] if record else "pending"
            if status == "present":
                result = "Presente"
            elif status == "absent":
                result = "Ausente"
            elif not intervals:
                result = "Sem registro"
            elif required_here and counted >= required_here:
                result = "Meta atingida"
            else:
                result = "Em andamento"

            last = intervals[-1] if intervals else None
            table.append({
                "Aluno": person["name"],
                "Matrícula": person["enrollment_number"],
                "Estado": state,
                "Primeira entrada": fmt_dt(intervals[0]["entered_at"]) if intervals else "-",
                "Última saída": ("Ainda na sala" if last["exited_at"] is None else fmt_dt(last["exited_at"])) if last else "-",
                "Entradas": len(intervals),
                "Tempo em aula": fmt_duration(counted),
                "Meta": fmt_duration(required_here),
                "Progresso": min(100.0, 100 * counted / required_here) if required_here else 0.0,
                "Situação": result,
            })
            for index, interval in enumerate(intervals):
                timeline.append((datetime.fromisoformat(interval["entered_at"]), person["name"], "Entrou" if index == 0 else "Voltou"))
                if interval["exited_at"]:
                    timeline.append((datetime.fromisoformat(interval["exited_at"]), person["name"], "Saiu"))

        cols = st.columns(4)
        cols[0].metric("Alunos cadastrados", len(people))
        cols[1].metric("Na sala agora", sum(r["Estado"] == "Na sala" for r in table))
        cols[2].metric("Já reconhecidos", sum(r["Estado"] != "Ainda não reconhecido" for r in table))
        cols[3].metric("Presentes ou na meta", sum(r["Situação"] in ("Presente", "Meta atingida") for r in table))

        st.dataframe(
            pd.DataFrame(table),
            hide_index=True,
            column_config={"Progresso": st.column_config.ProgressColumn("Progresso", min_value=0, max_value=100, format="%.0f%%")},
        )

        st.subheader("Entradas e saídas")
        if timeline:
            timeline.sort(key=lambda item: item[0], reverse=True)
            st.dataframe(pd.DataFrame([{
                "Horário": moment.strftime("%d/%m %H:%M:%S"), "Aluno": name, "Evento": event,
            } for moment, name, event in timeline]), hide_index=True)
        else:
            st.caption("Ninguém foi reconhecido nesta aula ainda. Use a aba Reconhecer.")

        with_intervals = [p for p in people if records.get(p["id"]) and records[p["id"]]["intervals"]]
        if with_intervals:
            st.subheader("Detalhe por aluno")
            names = {p["id"]: f"{p['name']} ({p['enrollment_number']})" for p in with_intervals}
            picked = st.selectbox("Aluno", list(names), format_func=names.get, key="dash_student")
            record = records[picked]
            rows, raw_total = [], 0
            for interval in record["intervals"]:
                entered = datetime.fromisoformat(interval["entered_at"])
                left = datetime.fromisoformat(interval["exited_at"]) if interval["exited_at"] else now
                raw = max(0, int((left - entered).total_seconds()))
                raw_total += raw
                rows.append({
                    "Entrou": fmt_dt(interval["entered_at"], True),
                    "Saiu": fmt_dt(interval["exited_at"], True) if interval["exited_at"] else "Ainda na sala",
                    "Ficou": fmt_duration(raw),
                    "Contou para a presença": fmt_duration(counted_seconds(interval, session, now)),
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True)
            st.metric("Tempo total contado", fmt_duration(record["total_seconds"]))
            if raw_total > record["total_seconds"]:
                st.caption("Parte do tempo ficou fora do horário da aula e não é contada.")

    if auto:
        st.fragment(run_every=5)(body)()
    else:
        body()


# --------------------------------------------------------------------------- aba: reconhecer

def handle_recognition(photo, class_id: int) -> None:
    frame, faces, preview = analyse_photo(photo)
    st.session_state.rec_preview = preview
    engine = load_engine()

    if len(faces) == 0:
        result = ("warning", "Nenhum rosto detectado", "Aproxime-se da câmera, com o rosto bem iluminado.")
    elif len(faces) > 1:
        result = ("warning", "Mais de um rosto na imagem", "Passe um aluno por vez.")
    else:
        try:
            response = api_client().send_recognition(class_id, st.session_state.camera_id, engine.embedding(frame, faces[0]), datetime.now())
        except Exception as exc:  # a API pode cair; a tela continua funcionando
            result = ("error", "Não foi possível enviar", str(exc))
        else:
            score = response.get("match_score")
            threshold = match_threshold()
            score_text = ""
            if score is not None:
                score_text = f"Similaridade {score:.2f}" + (f" (mínimo {threshold:.2f})" if threshold is not None else "")
            status = response["status"]
            if status == "identified":
                info, _ = request("GET", f"/sessions/{class_id}/students/{response['student_id']}/attendance")
                name = info["student_name"] if info else f"Aluno #{response['student_id']}"
                minutes = f", {info['total_minutes']:.1f} min em aula" if info else ""
                result = ("success", f"{name}: {ACTIONS.get(response['action'], response['action'])}", f"{score_text}{minutes}")
            elif status == "cooldown":
                result = ("info", "Aguarde alguns segundos", f"{response['message']}. {score_text}")
            elif status == "unregistered":
                result = ("warning", "Rosto não cadastrado", f"Nenhuma presença foi criada. {score_text}")
            elif status == "class_closed":
                result = ("error", "Aula já encerrada", "Selecione outra aula na barra lateral.")
            else:
                result = ("warning", response["message"], score_text)

    kind, title, detail = result
    st.session_state.rec_result = result
    st.session_state.rec_log = ([{"Hora": datetime.now().strftime("%H:%M:%S"), "Resultado": title}] + st.session_state.rec_log)[:10]
    st.session_state.rec_n += 1  # troca a chave do widget para liberar a camera para a proxima pessoa
    st.rerun()


def tab_recognize(class_id: int | None, sessions: list[dict] | None) -> None:
    if class_id is None:
        st.info("Crie ou selecione uma aula antes de reconhecer alunos.")
        return
    if not class_status_banner(class_id, sessions, "recognize"):
        return
    if engine_or_stop() is None:
        st.caption("Rode `python -m vision_client.download_models` para baixar os modelos.")
        return

    left, right = st.columns([1.1, 1], gap="large")
    with left:
        photo = st.camera_input("Um aluno por vez, olhando para a câmera", key=f"rec_cam_{st.session_state.rec_n}")
        if photo is not None:
            handle_recognition(photo, class_id)
        health = get_health(st.session_state.api_url) or {}
        st.caption(f"Cada foto alterna entre entrada e saída. Espere {health.get('cooldown_seconds', 15)} s "
                   "antes de fotografar o mesmo aluno de novo.")

    with right:
        result = st.session_state.rec_result
        if result:
            result_card(*result)
            if st.session_state.rec_preview is not None:
                st.image(st.session_state.rec_preview, caption="Última captura")
        else:
            result_card("idle", "Aguardando um aluno", "Tire a foto para registrar entrada ou saída.")

        if st.session_state.rec_log:
            st.caption("Últimos registros")
            st.dataframe(pd.DataFrame(st.session_state.rec_log), hide_index=True)

    with st.expander("Prefere uma câmera contínua na porta da sala?"):
        st.write("O modo contínuo detecta sozinho quem passa, sem precisar clicar. Rode no terminal:")
        st.code(
            f"python -m vision_client.run_camera --class-id {class_id} "
            f"--camera-id {st.session_state.camera_id} --api-url {st.session_state.api_url}",
            language="bash",
        )


# --------------------------------------------------------------------------- aba: cadastro

def tab_enroll() -> None:
    if engine_or_stop() is None:
        st.caption("Rode `python -m vision_client.download_models` para baixar os modelos.")
        return

    left, right = st.columns([1.1, 1], gap="large")
    with left:
        name = st.text_input("Nome completo", key="enroll_name")
        enrollment = st.text_input("Matrícula", key="enroll_enrollment")
        target = st.slider("Quantidade de fotos", 1, 7, 3, help="Mais fotos, com ângulos e luz diferentes, melhoram o reconhecimento.")

    samples: list = st.session_state.enroll_samples
    with right:
        st.write(f"Fotos capturadas: {len(samples)} de {target}")
        st.progress(min(1.0, len(samples) / target))
        if st.session_state.enroll_thumbs:
            thumbs = st.columns(min(len(st.session_state.enroll_thumbs), 4))
            for i, thumb in enumerate(st.session_state.enroll_thumbs):
                thumbs[i % len(thumbs)].image(thumb)

    if len(samples) < target:
        photo = st.camera_input(f"Foto {len(samples) + 1} de {target}", key=f"enroll_cam_{st.session_state.enroll_n}")
        if photo is not None:
            frame, faces, preview = analyse_photo(photo)
            if len(faces) != 1:
                st.warning("Precisa haver exatamente um rosto na foto. Limpe a foto e tente de novo.")
                st.image(preview)
            else:
                samples.append(load_engine().embedding(frame, faces[0]))
                st.session_state.enroll_thumbs.append(face_crop(frame, faces[0]))
                st.session_state.enroll_n += 1
                st.rerun()

    ready = len(samples) >= target and len(name.strip()) >= 2 and enrollment.strip() != ""
    cols = st.columns([1, 1, 3])
    if cols[0].button("Cadastrar aluno", type="primary", disabled=not ready):
        api = api_client()
        try:
            created = api.create_student(name.strip(), enrollment.strip(), samples[0])
            for sample in samples[1:]:
                api.add_embedding(created["id"], sample)
        except Exception as exc:
            st.error(str(exc))
        else:
            st.session_state.enroll_samples = []
            st.session_state.enroll_thumbs = []
            st.session_state.enroll_n += 1
            flash(f"{name.strip()} cadastrado com {len(samples)} fotos (ID {created['id']}).")
            st.rerun()
    if cols[1].button("Descartar fotos", disabled=not samples):
        st.session_state.enroll_samples = []
        st.session_state.enroll_thumbs = []
        st.session_state.enroll_n += 1
        st.rerun()
    if not ready and len(samples) >= target:
        st.caption("Preencha o nome (mínimo 2 letras) e a matrícula para concluir.")

    st.subheader("Alunos cadastrados")
    students = fetch_students()
    if students is None:
        st.warning("Este backend ainda não lista alunos. Atualize `app/main.py` e `app/schemas.py` com os arquivos novos.")
    elif not students:
        st.info("Nenhum aluno cadastrado ainda.")
    else:
        st.dataframe(pd.DataFrame([{
            "Nome": item["name"], "Matrícula": item["enrollment_number"],
            "Fotos": item["embeddings_count"], "Cadastrado em": fmt_utc_local(item["created_at"]),
        } for item in students]), hide_index=True)


# --------------------------------------------------------------------------- aba: aulas

def tab_sessions(class_id: int | None, sessions: list[dict] | None) -> None:
    left, right = st.columns(2, gap="large")

    with left:
        st.subheader("Nova aula no horário oficial")
        with st.form("scheduled"):
            class_date = st.date_input("Data", value=date.today())
            slot = st.selectbox("Horário", range(1, 7), format_func=lambda i: SCHEDULE[i - 1])
            label = st.text_input("Turma", "Turma A")
            pct = st.slider("Presença mínima (%)", 10, 100, 75)
            create = st.form_submit_button("Criar aula", type="primary")
        if create:
            data, error = request("POST", "/sessions/scheduled", json={
                "class_date": class_date.isoformat(), "schedule_index": slot,
                "label": label or None, "minimum_percentage": pct / 100,
            })
            if error:
                st.error(error)
            else:
                st.session_state.active_class = data["id"]
                flash(f"Aula criada com ID {data['id']}. Ela já está selecionada.")
                st.rerun()


        if st.button("Criar aula de teste (agora até daqui a 1 hora)"):
            now = datetime.now().replace(microsecond=0)
            data, error = request("POST", "/sessions", json={
                "label": "Aula de teste", "starts_at": now.isoformat(),
                "ends_at": (now + timedelta(hours=1)).isoformat(), "minimum_percentage": 0.10,
            })
            if error:
                st.error(error)
            else:
                st.session_state.active_class = data["id"]
                flash(f"Aula de teste criada com ID {data['id']} (meta de 10% para facilitar o teste).")
                st.rerun()

        with st.expander("Aula com horário personalizado"):
            with st.form("custom"):
                day = st.date_input("Data", value=date.today(), key="custom_day")
                t1, t2 = st.columns(2)
                start = t1.time_input("Início", value=time(8, 0))
                end = t2.time_input("Fim", value=time(9, 0))
                custom_label = st.text_input("Turma", "Turma A", key="custom_label")
                custom_pct = st.slider("Presença mínima (%)", 10, 100, 75, key="custom_pct")
                custom_create = st.form_submit_button("Criar aula personalizada")
            if custom_create:
                data, error = request("POST", "/sessions", json={
                    "label": custom_label, "starts_at": datetime.combine(day, start).isoformat(),
                    "ends_at": datetime.combine(day, end).isoformat(), "minimum_percentage": custom_pct / 100,
                })
                if error:
                    st.error(error)
                else:
                    st.session_state.active_class = data["id"]
                    flash(f"Aula criada com ID {data['id']}. Ela já está selecionada.")
                    st.rerun()

    with right:
        st.subheader("Encerrar aula")
        if class_id is None:
            st.info("Selecione uma aula na barra lateral.")
        else:
            st.write(f"Aula selecionada: **#{class_id}**")
            st.caption("Ao encerrar, quem está dentro da sala sai automaticamente no horário de fim, "
                       "e cada aluno recebe presente ou ausente. Isso não pode ser desfeito.")
            confirm = st.checkbox("A aula terminou e quero calcular a presença")
            if st.button("Encerrar e calcular presença", type="primary", disabled=not confirm):
                data, error = request("POST", f"/sessions/{class_id}/finalize")
                if error:
                    st.error(error)
                else:
                    m = st.columns(3)
                    m[0].metric("Alunos avaliados", data["finalized_students"])
                    m[1].metric("Presentes", data["present"])
                    m[2].metric("Ausentes", data["absent"])

        if sessions:
            st.subheader("Aulas recentes")
            st.dataframe(pd.DataFrame([{
                "ID": s["id"], "Turma": s["label"],
                "Início": datetime.fromisoformat(s["starts_at"]).strftime("%d/%m %H:%M"),
                "Fim": datetime.fromisoformat(s["ends_at"]).strftime("%H:%M"),
                "Meta": f"{s['minimum_percentage']:.0%}",
                "Situação": "Aberta" if s["status"] == "open" else "Encerrada",
            } for s in sessions]), hide_index=True)


# --------------------------------------------------------------------------- aba: diagnostico

def tab_diagnostics() -> None:
    st.subheader("Arquivos dos modelos")
    for path in (DEFAULT_DETECTOR, DEFAULT_RECOGNIZER):
        if not path.is_file():
            st.error(f"Não encontrado: {path}. Rode `python -m vision_client.download_models`.")
        elif path.stat().st_size < 10_000:
            st.error(f"{path.name} tem só {path.stat().st_size} bytes. Baixe de novo com "
                     "`python -m vision_client.download_models --force`.")
        else:
            st.success(f"{path.name}: {path.stat().st_size / 1_000_000:.1f} MB")
    if engine_or_stop() is None:
        return

    st.subheader("Comparar duas fotos")
    threshold = match_threshold() or 0.45
    st.write(f"Tire duas fotos e veja a similaridade que o modelo calcula. O backend só aceita um reconhecimento "
             f"a partir de {threshold:.2f}. Duas fotos da mesma pessoa devem ficar acima disso, e pessoas "
             "diferentes bem abaixo.")
    col_a, col_b = st.columns(2)
    photo_a = col_a.camera_input("Foto A", key="diag_a")
    photo_b = col_b.camera_input("Foto B", key="diag_b")
    if photo_a is None or photo_b is None:
        return

    frame_a, faces_a, preview_a = analyse_photo(photo_a)
    frame_b, faces_b, preview_b = analyse_photo(photo_b)
    if len(faces_a) != 1 or len(faces_b) != 1:
        st.warning(f"Precisa haver exatamente um rosto em cada foto (A: {len(faces_a)}, B: {len(faces_b)}).")
        return
    engine = load_engine()
    score = float(np.dot(engine.embedding(frame_a, faces_a[0]), engine.embedding(frame_b, faces_b[0])))
    if score >= threshold:
        result_card("success", f"Similaridade {score:.2f}", f"Acima do mínimo de {threshold:.2f}: o modelo entende como a mesma pessoa.")
    else:
        result_card("warning", f"Similaridade {score:.2f}", f"Abaixo do mínimo de {threshold:.2f}: o modelo entende como pessoas diferentes.")
    st.caption("Se duas fotos suas ficarem abaixo do mínimo, melhore a luz e o enquadramento, ou cadastre mais fotos "
               "com ângulos diferentes. Se pessoas diferentes ficarem acima, aumente FACE_MATCH_THRESHOLD no backend.")


# --------------------------------------------------------------------------- principal

def main() -> None:
    init_state()
    st.markdown(CSS, unsafe_allow_html=True)
    class_id, sessions = sidebar()

    st.title("Controle de frequência")
    st.markdown('<div class="app-subtitle">Cada reconhecimento alterna entre entrada e saída. '
                'A presença é a soma do tempo dentro da sala.</div>', unsafe_allow_html=True)

    if st.session_state.flash:
        st.success(st.session_state.flash)
        st.session_state.flash = None

    painel, reconhecer, cadastro, aulas, diagnostico = st.tabs(["Painel", "Reconhecer", "Cadastrar aluno", "Aulas", "Diagnóstico"])
    with painel:
        tab_dashboard(class_id, sessions)
    with reconhecer:
        tab_recognize(class_id, sessions)
    with cadastro:
        tab_enroll()
    with aulas:
        tab_sessions(class_id, sessions)
    with diagnostico:
        tab_diagnostics()


main()