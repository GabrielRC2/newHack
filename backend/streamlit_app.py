"""Painel Streamlit do sistema de frequencia por reconhecimento facial.

Executar na raiz do projeto (a pasta que contem `vision_client/`):

    streamlit run streamlit_app.py

Nenhuma foto e gravada em disco: a imagem da webcam fica so na memoria da sessao,
e apenas o embedding (vetor numerico) e enviado para a API.
"""
from __future__ import annotations

import html
import os
import subprocess
import sys
import threading
import time as clock
from csv import DictWriter
from datetime import date, datetime, time, timedelta, timezone
from io import StringIO
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
import requests
import streamlit as st

try:  # camera continua no navegador: pip install streamlit-webrtc
    import av
    from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

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


# --------------------------------------------------------------------------- relatorio

STATUS_LABELS = {"pending": "Em andamento", "present": "Presente", "absent": "Ausente"}


def parse_dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def display_time(value: datetime | str | None) -> str:
    moment = parse_dt(value) if isinstance(value, str) else value
    return moment.strftime("%H:%M:%S") if moment else "—"


def display_duration(seconds: int) -> str:
    hours, remainder = divmod(max(0, int(seconds)), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}h {minutes:02d}min {secs:02d}s"


def build_report(records: list[dict], session: dict) -> dict:
    """Consolida a frequência de uma aula em resumo, tabela por aluno e lista de intervalos."""
    starts_at, ends_at = parse_dt(session["starts_at"]), parse_dt(session["ends_at"])
    closed = session["status"] == "closed"
    # Ponto ate onde o tempo ja "existe": encerramento real, ou agora (limitado ao fim oficial).
    reference = parse_dt(session["closed_at"]) if closed and session.get("closed_at") else min(datetime.now(), ends_at)
    official_seconds = max(1, int((ends_at - starts_at).total_seconds()))
    required_seconds = int(official_seconds * session["minimum_percentage"])

    students: list[dict] = []
    intervals: list[dict] = []
    for item in sorted(records, key=lambda record: record["student_name"].lower()):
        student_intervals = item["intervals"]
        first_entry = parse_dt(student_intervals[0]["entered_at"]) if student_intervals else None
        last = student_intervals[-1] if student_intervals else None
        last_exit = parse_dt(last["exited_at"]) if last else None
        inside_now = bool(last and last["exited_at"] is None)

        total = int(item["total_seconds"])
        late_seconds = max(0, int((first_entry - starts_at).total_seconds())) if first_entry else None
        if first_entry:
            span_end = min(last_exit or reference, ends_at)
            span = max(0, int((span_end - max(first_entry, starts_at)).total_seconds()))
            away_seconds = max(0, span - total)
        else:
            away_seconds = 0

        students.append({
            "name": item["student_name"], "enrollment": item["enrollment_number"],
            "first_entry": first_entry, "last_exit": last_exit, "inside_now": inside_now,
            "intervals": len(student_intervals), "total_seconds": total,
            "percentage": total / official_seconds * 100,
            "missing_seconds": max(0, required_seconds - total),
            "late_seconds": late_seconds, "away_seconds": away_seconds,
            "status": item["status"] if item["status"] != "pending" or first_entry else "no_record",
        })
        for number, interval in enumerate(student_intervals, start=1):
            entry, exit_ = parse_dt(interval["entered_at"]), parse_dt(interval["exited_at"])
            clipped_start = max(entry, starts_at)
            clipped_end = min(exit_ or reference, ends_at)
            intervals.append({
                "name": item["student_name"], "enrollment": item["enrollment_number"], "number": number,
                "entered_at": entry, "exited_at": exit_, "open": exit_ is None,
                "seconds": max(0, int((clipped_end - clipped_start).total_seconds())),
                "chart_start": clipped_start, "chart_end": max(clipped_start, clipped_end),
            })

    counted = [s for s in students if s["first_entry"]]
    summary = {
        "starts_at": starts_at, "ends_at": ends_at, "closed": closed,
        "closed_at": parse_dt(session.get("closed_at")), "reference": reference,
        "official_seconds": official_seconds, "required_seconds": required_seconds,
        "minimum_percentage": session["minimum_percentage"],
        "total_students": len(students), "attended": len(counted),
        "present": sum(s["status"] == "present" for s in students),
        "absent": sum(s["status"] == "absent" for s in students),
        "inside": sum(s["inside_now"] for s in students),
        "average_seconds": int(sum(s["total_seconds"] for s in counted) / len(counted)) if counted else 0,
        "late_students": sum(1 for s in counted if s["late_seconds"] and s["late_seconds"] >= 60),
    }
    return {"summary": summary, "students": students, "intervals": intervals}


def student_status_label(student: dict) -> str:
    if student["status"] == "no_record":
        return "Sem registro"
    return STATUS_LABELS.get(student["status"], student["status"])


def report_display_rows(report: dict) -> list[dict]:
    return [{
        "Aluno": s["name"], "Matrícula": s["enrollment"],
        "Primeira entrada": display_time(s["first_entry"]),
        "Atraso": "—" if s["late_seconds"] is None else f"{s['late_seconds'] // 60} min",
        "Última saída": "Em sala" if s["inside_now"] else display_time(s["last_exit"]),
        "Intervalos": s["intervals"],
        "Tempo em sala": display_duration(s["total_seconds"]),
        "Tempo ausente*": display_duration(s["away_seconds"]) if s["first_entry"] else "—",
        "% da aula": f"{s['percentage']:.1f}%",
        "Falta p/ mínimo": display_duration(s["missing_seconds"]) if s["status"] != "present" else "—",
        "Situação": student_status_label(s),
    } for s in report["students"]]


def to_csv(rows: list[dict]) -> bytes:
    if not rows:
        return b""
    output = StringIO()
    writer = DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8-sig")  # BOM para o Excel abrir acentos corretamente


def report_csv_rows(report: dict) -> list[dict]:
    return [{
        "Aluno": s["name"], "Matrícula": s["enrollment"],
        "Primeira entrada": display_time(s["first_entry"]),
        "Atraso (min)": "" if s["late_seconds"] is None else round(s["late_seconds"] / 60, 1),
        "Última saída": "Em sala" if s["inside_now"] else display_time(s["last_exit"]),
        "Intervalos": s["intervals"],
        "Tempo em sala (min)": round(s["total_seconds"] / 60, 2),
        "Tempo ausente entre entradas (min)": round(s["away_seconds"] / 60, 2),
        "% da aula": round(s["percentage"], 1),
        "Falta para o mínimo (min)": round(s["missing_seconds"] / 60, 2),
        "Situação": student_status_label(s),
    } for s in report["students"]]


def interval_rows(report: dict) -> list[dict]:
    return [{
        "Aluno": i["name"], "Matrícula": i["enrollment"], "Nº": i["number"],
        "Entrada": i["entered_at"].strftime("%d/%m/%Y %H:%M:%S"),
        "Saída": "Em sala" if i["open"] else i["exited_at"].strftime("%d/%m/%Y %H:%M:%S"),
        "Duração contabilizada": display_duration(i["seconds"]),
    } for i in report["intervals"]]


def report_html(session: dict, class_id: int, report: dict) -> bytes:
    """Relatório autocontido, pronto para abrir no navegador e imprimir/salvar em PDF."""
    s = report["summary"]
    esc = html.escape
    status_text = "Aula encerrada" if s["closed"] else "Relatório parcial (aula em andamento)"
    student_rows = "".join(
        f"<tr><td>{esc(r['Aluno'])}</td><td>{esc(r['Matrícula'])}</td><td>{r['Primeira entrada']}</td>"
        f"<td>{r['Última saída']}</td><td>{r['Intervalos']}</td><td>{r['Tempo em sala']}</td>"
        f"<td>{r['% da aula']}</td><td class='{('ok' if r['Situação'] == 'Presente' else 'bad' if r['Situação'] in ('Ausente', 'Sem registro') else '')}'>"
        f"{r['Situação']}</td></tr>"
        for r in report_display_rows(report)
    )
    interval_html = "".join(
        f"<tr><td>{esc(r['Aluno'])}</td><td>{r['Nº']}</td><td>{r['Entrada']}</td><td>{r['Saída']}</td>"
        f"<td>{r['Duração contabilizada']}</td></tr>"
        for r in interval_rows(report)
    )
    document = f"""<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<title>Relatório da aula #{class_id}</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:32px;color:#0f172a}}
h1{{margin:0 0 4px}} h2{{margin-top:28px;border-bottom:2px solid #0891b2;padding-bottom:4px}}
.meta{{color:#475569}} .cards{{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0}}
.card{{border:1px solid #cbd5e1;border-radius:10px;padding:10px 16px;min-width:130px}}
.card b{{display:block;font-size:1.4rem}} table{{border-collapse:collapse;width:100%;font-size:.9rem}}
th,td{{border:1px solid #cbd5e1;padding:6px 8px;text-align:left}} th{{background:#f1f5f9}}
.ok{{color:#047857;font-weight:600}} .bad{{color:#b91c1c;font-weight:600}}
@media print{{body{{margin:12px}}}}
</style></head><body>
<h1>Relatório de frequência — {esc(session['label'])}</h1>
<p class="meta">Aula #{class_id} · {s['starts_at']:%d/%m/%Y %H:%M} até {s['ends_at']:%H:%M} ·
{status_text} · gerado em {datetime.now():%d/%m/%Y %H:%M}</p>
<div class="cards">
<div class="card"><b>{s['attended']}</b>alunos identificados</div>
<div class="card"><b>{s['present']}</b>presentes</div>
<div class="card"><b>{s['absent']}</b>ausentes</div>
<div class="card"><b>{display_duration(s['average_seconds'])}</b>tempo médio em sala</div>
<div class="card"><b>{s['minimum_percentage'] * 100:.0f}%</b>presença mínima ({display_duration(s['required_seconds'])})</div>
</div>
<h2>Frequência por aluno</h2>
<table><tr><th>Aluno</th><th>Matrícula</th><th>Primeira entrada</th><th>Última saída</th><th>Intervalos</th>
<th>Tempo em sala</th><th>% da aula</th><th>Situação</th></tr>{student_rows}</table>
<h2>Entradas e saídas</h2>
<table><tr><th>Aluno</th><th>Nº</th><th>Entrada</th><th>Saída</th><th>Duração contabilizada</th></tr>{interval_html}</table>
<p class="meta">O tempo é contado desde o início oficial da aula e limitado ao seu término.</p>
</body></html>"""
    return document.encode("utf-8")


def render_gantt(report: dict) -> None:
    """Linha do tempo de presença: uma barra por intervalo de cada aluno."""
    import altair as alt  # vem junto com o Streamlit
    summary = report["summary"]
    frame = pd.DataFrame([{
        "Aluno": i["name"], "Entrada": i["chart_start"], "Saída": i["chart_end"],
        "Duração": display_duration(i["seconds"]),
    } for i in report["intervals"] if i["seconds"] > 0])
    if frame.empty:
        return
    scale = alt.Scale(domain=[summary["starts_at"], summary["ends_at"]])
    chart = alt.Chart(frame).mark_bar(cornerRadius=3, clip=True).encode(
        x=alt.X("Entrada:T", scale=scale, title="Horário da aula"),
        x2="Saída:T",
        y=alt.Y("Aluno:N", title=None),
        tooltip=["Aluno", alt.Tooltip("Entrada:T", format="%H:%M:%S"),
                 alt.Tooltip("Saída:T", format="%H:%M:%S"), "Duração"],
    ).properties(height=max(120, 34 * frame["Aluno"].nunique()))
    st.altair_chart(chart, use_container_width=True)


# --------------------------------------------------------------------------- janela de cadastro (OpenCV)

ENROLL_KEY = "enrollment_camera"
PROJECT_DIR = Path(__file__).resolve().parent


@st.cache_resource
def process_registry() -> dict[str, subprocess.Popen]:
    """Compartilhado entre sessoes: sobrevive ao F5, entao ainda da para fechar a janela da webcam."""
    return {}


def enroll_process() -> subprocess.Popen | None:
    return process_registry().get(ENROLL_KEY)


def is_enroll_running() -> bool:
    process = enroll_process()
    return process is not None and process.poll() is None


def start_enroll_process(name: str, enrollment: str, samples: int) -> tuple[bool, str]:
    """Abre vision_client.enroll numa janela OpenCV, com a mesma virtualenv do Streamlit."""
    if is_enroll_running():
        return False, "A janela de cadastro já está aberta."
    environment = os.environ.copy()
    environment["FREQUENCY_API_URL"] = st.session_state.api_url
    if st.session_state.api_token:
        environment["FREQUENCY_API_TOKEN"] = st.session_state.api_token
    command = [sys.executable, "-m", "vision_client.enroll", "--name", name,
               "--enrollment-number", enrollment, "--samples", str(samples)]
    try:
        process_registry()[ENROLL_KEY] = subprocess.Popen(command, cwd=PROJECT_DIR, env=environment)
    except OSError as exc:
        return False, f"Não foi possível abrir a câmera: {exc}"
    return True, "A janela da câmera foi aberta. Pressione C para capturar cada amostra."


def stop_enroll_process() -> tuple[bool, str]:
    process = enroll_process()
    if process is None or process.poll() is not None:
        return False, "Nenhuma janela de cadastro em execução."
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)
    return True, "Janela de cadastro fechada."


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

def describe_recognition(api: FrequencyApi, response: dict, class_id: int, threshold: float | None) -> tuple[str, str, str]:
    """Transforma a resposta de /recognitions em (tipo, titulo, detalhe) para o result_card.

    Nao usa st.session_state, entao pode rodar tanto na tela quanto na thread de video.
    """
    score = response.get("match_score")
    score_text = ""
    if score is not None:
        score_text = f"Similaridade {score:.2f}" + (f" (mínimo {threshold:.2f})" if threshold is not None else "")
    status = response["status"]
    if status == "identified":
        info = None
        try:
            reply = api.session.get(
                f"{api.base_url}/sessions/{class_id}/students/{response['student_id']}/attendance", timeout=8)
            info = reply.json() if reply.ok else None
        except (requests.RequestException, ValueError):
            pass
        name = info["student_name"] if info else f"Aluno #{response['student_id']}"
        minutes = f", {info['total_minutes']:.1f} min em aula" if info else ""
        return ("success", f"{name}: {ACTIONS.get(response['action'], response['action'])}", f"{score_text}{minutes}")
    if status == "cooldown":
        return ("info", "Aguarde alguns segundos", f"{response['message']}. {score_text}")
    if status == "unregistered":
        return ("warning", "Rosto não cadastrado", f"Nenhuma presença foi criada. {score_text}")
    if status == "class_closed":
        return ("error", "Aula já encerrada", "Selecione outra aula na barra lateral.")
    return ("warning", response["message"], score_text)


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
            result = describe_recognition(api_client(), response, class_id, match_threshold())

    kind, title, detail = result
    st.session_state.rec_result = result
    st.session_state.rec_log = ([{"Hora": datetime.now().strftime("%H:%M:%S"), "Resultado": title}] + st.session_state.rec_log)[:10]
    st.session_state.rec_n += 1  # troca a chave do widget para liberar a camera para a proxima pessoa
    st.rerun()


MIN_FACE_WIDTH = 80      # px: rosto mais estreito que isso esta longe demais para um embedding confiavel
STABLE_FRAMES = 3        # frames seguidos com um unico rosto antes de disparar (evita rosto borrado de quem acabou de entrar)
ABSENCE_SECONDS = 1.0    # tempo sem rosto que rearma uma nova passagem


class RecognitionProcessor(VideoProcessorBase if WEBRTC_AVAILABLE else object):
    """Processa cada frame da camera do navegador e envia uma deteccao por passagem.

    Roda numa thread propria do streamlit-webrtc, entao NAO pode usar st.session_state.
    A tela conversa com ele por set_config() e snapshot(), ambos protegidos por lock.
    Nenhuma imagem e gravada: so o embedding vai para a API.
    """

    IDLE = ("idle", "Aguardando um aluno", "Fique de frente para a câmera. O registro é automático.")

    def __init__(self) -> None:
        # Instancia propria: o detector guarda estado (tamanho de entrada) e nao deve ser compartilhado entre threads.
        self.engine = FaceEngine(DEFAULT_DETECTOR, DEFAULT_RECOGNIZER)
        self._lock = threading.Lock()
        self._config: dict = {}
        self._result: tuple[str, str, str] = self.IDLE
        self._log: list[dict] = []
        self._armed = True
        self._sending = False
        self._stable = 0
        self._last_face_at = 0.0

    def set_config(self, **config) -> None:
        with self._lock:
            self._config = config

    def snapshot(self) -> tuple[tuple[str, str, str], list[dict]]:
        with self._lock:
            return self._result, list(self._log)

    def _publish(self, result: tuple[str, str, str]) -> None:
        with self._lock:
            self._result = result
            self._log = ([{"Hora": datetime.now().strftime("%H:%M:%S"), "Resultado": result[1]}] + self._log)[:10]

    def _send(self, embedding: list[float], config: dict) -> None:
        # Em thread separada para o video nao travar enquanto a API responde.
        try:
            api = FrequencyApi(config["api_url"], config["token"])
            response = api.send_recognition(config["class_id"], config["camera_id"], embedding, datetime.now())
            result = describe_recognition(api, response, config["class_id"], config["threshold"])
        except Exception as exc:  # a API pode cair; a camera continua rodando
            result = ("error", "Não foi possível enviar", str(exc))
        self._publish(result)
        self._sending = False

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        now = clock.monotonic()
        with self._lock:
            config = dict(self._config)
        faces = self.engine.detect(image)
        banner, color = "Aguardando uma pessoa", (255, 255, 255)  # ASCII: fontes do OpenCV nao desenham acentos

        if len(faces) == 0:
            self._stable = 0
            if now - self._last_face_at >= ABSENCE_SECONDS:
                self._armed = True
        elif len(faces) > 1:
            self._last_face_at = now
            self._stable = 0
            banner, color = "Mais de um rosto: um aluno por vez", (0, 165, 255)
            for face in faces:
                draw_face(image, face, color)
        else:
            self._last_face_at = now
            face = faces[0]
            if face[2] < MIN_FACE_WIDTH:
                self._stable = 0
                banner, color = "Aproxime-se da camera", (0, 165, 255)
            else:
                self._stable += 1
                if self._armed:
                    banner, color = "Rosto detectado", (0, 210, 0)
                else:
                    banner, color = "Registrado. Saia do quadro para nova passagem", (200, 200, 200)
                ready = self._armed and self._stable >= STABLE_FRAMES and not self._sending
                if ready and config.get("class_id") is not None:
                    self._armed = False
                    self._sending = True
                    try:
                        # O embedding sai do frame limpo, antes de desenhar qualquer coisa nele.
                        embedding = self.engine.embedding(image, face)
                    except Exception as exc:
                        self._sending = False
                        self._publish(("error", "Falha ao ler o rosto", str(exc)))
                    else:
                        threading.Thread(target=self._send, args=(embedding, config), daemon=True).start()
            draw_face(image, face, color)

        cv.rectangle(image, (0, 0), (image.shape[1], 34), (25, 25, 25), -1)
        cv.putText(image, banner, (10, 24), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return av.VideoFrame.from_ndarray(image, format="bgr24")


def live_recognition(class_id: int, left, right) -> None:
    """Camera sempre ligada: o navegador manda o video e cada passagem vira um reconhecimento."""
    with left:
        ctx = webrtc_streamer(
            key="live-recognition",
            video_processor_factory=RecognitionProcessor,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            async_processing=True,
        )
        if ctx.video_processor is not None:
            # Reenviado a cada rerun: trocar de aula ou de API na barra lateral vale na hora, sem religar a camera.
            ctx.video_processor.set_config(
                class_id=class_id, camera_id=st.session_state.camera_id, api_url=st.session_state.api_url,
                token=st.session_state.api_token or None, threshold=match_threshold(),
            )
        health = get_health(st.session_state.api_url) or {}
        st.caption(f"Clique em START e deixe a câmera ligada. Cada passagem alterna entre entrada e saída; "
                   f"o mesmo aluno só é registrado de novo após {health.get('cooldown_seconds', 15)} s.")

    def panel() -> None:
        processor = ctx.video_processor
        if processor is None:
            result_card("idle", "Câmera desligada", "Clique em START para ligar. Depois disso o registro é automático.")
            return
        result, log = processor.snapshot()
        result_card(*result)
        if log:
            st.caption("Últimos registros")
            st.dataframe(pd.DataFrame(log), hide_index=True)

    with right:
        st.fragment(run_every=1 if ctx.state.playing else None)(panel)()


def tab_recognize(class_id: int | None, sessions: list[dict] | None) -> None:
    if class_id is None:
        st.info("Crie ou selecione uma aula antes de reconhecer alunos.")
        return
    if not class_status_banner(class_id, sessions, "recognize"):
        return
    if engine_or_stop() is None:
        st.caption("Rode `python -m vision_client.download_models` para baixar os modelos.")
        return

    modes = ["Câmera contínua", "Foto"] if WEBRTC_AVAILABLE else ["Foto"]
    mode = st.radio("Modo de reconhecimento", modes, horizontal=True, key="rec_mode")
    if not WEBRTC_AVAILABLE:
        st.info("Para reconhecer com a câmera sempre ligada, rode `pip install streamlit-webrtc` e reinicie o app.")

    left, right = st.columns([1.1, 1], gap="large")
    if mode == "Câmera contínua":
        live_recognition(class_id, left, right)
    else:
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

    with st.expander("Prefere uma câmera USB na porta da sala, fora do navegador?"):
        st.write("Esse modo abre uma janela do OpenCV e detecta sozinho quem passa. Rode no terminal:")
        st.code(
            f"python -m vision_client.run_camera --class-id {class_id} "
            f"--camera-id {st.session_state.camera_id} --api-url {st.session_state.api_url}",
            language="bash",
        )


# --------------------------------------------------------------------------- aba: cadastro

def enroll_with_photos() -> None:
    """Cadastro pelo navegador: uma foto por vez com st.camera_input."""
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


def enroll_with_window() -> None:
    """Cadastro pela janela da webcam (OpenCV), aberta no computador que roda o painel."""
    st.info("Abre a janela da webcam. Deixe só um rosto no enquadramento e pressione **C** para capturar "
            "cada amostra (**Q** cancela). Ao concluir, o aluno é enviado para a API.")
    with st.form("enroll_window"):
        name = st.text_input("Nome completo")
        enrollment = st.text_input("Matrícula")
        samples = st.slider("Quantidade de amostras", 1, 7, 3,
                            help="Mais amostras, com ângulos e luz diferentes, melhoram o reconhecimento.")
        go = st.form_submit_button("Abrir câmera para cadastrar", type="primary")
    if go:
        if len(name.strip()) < 2 or not enrollment.strip():
            st.error("Preencha o nome (mínimo 2 letras) e a matrícula antes de abrir a câmera.")
        else:
            started, message = start_enroll_process(name.strip(), enrollment.strip(), samples)
            (st.success if started else st.warning)(message)

    process = enroll_process()
    if is_enroll_running():
        st.success("Janela de cadastro em execução. Quando terminar, clique em “Atualizar lista” abaixo.")
        if st.button("Cancelar cadastro e fechar a câmera"):
            stopped, message = stop_enroll_process()
            (st.success if stopped else st.info)(message)
    elif process is not None and process.poll() not in (None, 0):
        st.warning(f"A janela de cadastro fechou com erro (código {process.poll()}). "
                   "Veja o terminal do Streamlit e confirme que a webcam não está em uso.")
    st.caption("A webcam só pode ser usada por um programa por vez: se a câmera contínua da aba Reconhecer "
               "estiver ligada, clique em STOP nela antes de abrir esta janela.")


def student_list_section() -> None:
    head, refresh = st.columns([4, 1])
    head.subheader("Alunos cadastrados")
    refresh.button("Atualizar lista", use_container_width=True)  # o clique ja reexecuta a pagina
    students = fetch_students()
    if students is None:
        st.warning("Este backend ainda não lista alunos. Atualize `app/main.py` e `app/schemas.py` com os arquivos novos.")
    elif not students:
        st.info("Nenhum aluno cadastrado ainda.")
    else:
        st.dataframe(pd.DataFrame([{
            "ID": item["id"], "Nome": item["name"], "Matrícula": item["enrollment_number"],
            "Fotos": item["embeddings_count"], "Cadastrado em": fmt_utc_local(item["created_at"]),
        } for item in students]), hide_index=True)

        with st.expander("Remover aluno"):
            labels = {item["id"]: f"#{item['id']} {item['name']} ({item['enrollment_number']})" for item in students}
            picked = st.selectbox("Aluno", list(labels), format_func=labels.get, key="delete_student")
            st.caption("Apaga o cadastro, as fotos (embeddings) e todo o histórico de presença e reconhecimentos "
                       "desse aluno, inclusive em aulas já encerradas. Isso não pode ser desfeito. "
                       "Quem ainda está dentro da sala não pode ser removido: registre a saída ou encerre a aula antes.")
            confirm_delete = st.checkbox("Entendo e quero apagar este aluno", key=f"delete_confirm_{picked}")
            if st.button("Apagar aluno", type="primary", disabled=not confirm_delete):
                data, error = request("DELETE", f"/students/{picked}", params={"confirm": "true"})
                if error:
                    st.error(error)
                else:
                    flash(f"{data['name']} foi removido.")
                    st.rerun()


def tab_enroll() -> None:
    mode = st.radio("Como capturar o rosto", ["Janela da câmera (OpenCV)", "Fotos pelo navegador"],
                    horizontal=True, key="enroll_mode")
    if mode.startswith("Janela"):
        enroll_with_window()
    else:
        enroll_with_photos()
    st.divider()
    student_list_section()


# --------------------------------------------------------------------------- aba: aulas

def tab_sessions(class_id: int | None, sessions: list[dict] | None) -> None:
    left, right = st.columns(2, gap="large")

    with left:
        st.subheader("Nova aula")
        mode = st.radio("Horário", ["Livre (escolho início e fim)", "Oficial da escola"], horizontal=True, key="slot_mode")
        free = mode.startswith("Livre")
        with st.form("new_session"):
            day = st.date_input("Data", value=date.today())
            if free:
                t1, t2 = st.columns(2)
                start = t1.time_input("Início", value=time(8, 0), step=60)
                end = t2.time_input("Fim", value=time(9, 0), step=60)
            else:
                slot = st.selectbox("Horário", range(1, 7), format_func=lambda i: SCHEDULE[i - 1])
            label = st.text_input("Turma", "Turma A")
            pct = st.slider("Presença mínima (%)", 10, 100, 75)
            create = st.form_submit_button("Criar aula", type="primary")
        if create:
            if not label.strip():
                data, error = None, "Informe o nome ou a descrição da turma."
            elif free and datetime.combine(day, end) <= datetime.combine(day, start):
                data, error = None, "O fim precisa ser depois do início."
            elif free:
                data, error = request("POST", "/sessions", json={
                    "label": label.strip(), "starts_at": datetime.combine(day, start).isoformat(),
                    "ends_at": datetime.combine(day, end).isoformat(), "minimum_percentage": pct / 100,
                })
            else:
                data, error = request("POST", "/sessions/scheduled", json={
                    "class_date": day.isoformat(), "schedule_index": slot,
                    "label": label.strip(), "minimum_percentage": pct / 100,
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


# --------------------------------------------------------------------------- aba: relatorio

def tab_report(class_id: int | None) -> None:
    top, reload_col = st.columns([4, 1])
    top.subheader("Relatório da aula")
    reload_col.button("Recarregar", use_container_width=True, key="reload_report")
    if class_id is None:
        st.info("Crie ou selecione uma aula na barra lateral para ver o relatório.")
        return

    session, error = request("GET", f"/sessions/{class_id}")
    if error:
        st.error(error)
        return
    records, error = request("GET", f"/sessions/{class_id}/attendance")
    if error:
        st.error(error)
        return

    report = build_report(records or [], session)
    s = report["summary"]
    st.markdown(f"**{session['label']}** · aula #{class_id} · "
                f"{s['starts_at']:%d/%m/%Y}, {s['starts_at']:%H:%M} às {s['ends_at']:%H:%M}")
    if s["closed"]:
        st.success(f"Relatório final · aula encerrada às {s['closed_at']:%H:%M:%S}." if s["closed_at"] else "Relatório final.")
    else:
        st.info("Relatório parcial: a aula ainda está em andamento e os números mudam a cada passagem pela câmera.")

    c = st.columns(5)
    c[0].metric("Alunos com registro", s["total_students"])
    c[1].metric("Identificados", s["attended"])
    c[2].metric("Presentes", s["present"] if s["closed"] else "—")
    c[3].metric("Ausentes", s["absent"] if s["closed"] else "—")
    c[4].metric("Tempo médio em sala", display_duration(s["average_seconds"]))
    st.caption(f"Duração oficial: {display_duration(s['official_seconds'])} · mínimo de {s['minimum_percentage'] * 100:.0f}% "
               f"= {display_duration(s['required_seconds'])} · {s['late_students']} aluno(s) chegaram com mais de 1 min de atraso.")

    if not report["students"]:
        st.info("Ainda não há alunos com registro nesta aula.")
        return

    st.markdown("#### Frequência por aluno")
    st.dataframe(report_display_rows(report), hide_index=True)
    st.caption("*Tempo ausente = tempo fora da sala entre a primeira entrada e a última saída. "
               "O tempo em sala é contado desde o início oficial e limitado ao término da aula.")

    st.markdown("#### Linha do tempo de presença")
    render_gantt(report)

    st.markdown("#### Todas as entradas e saídas")
    st.dataframe(interval_rows(report), hide_index=True)

    st.markdown("#### Exportar")
    d1, d2, d3 = st.columns(3)
    d1.download_button("Resumo por aluno (CSV)", data=to_csv(report_csv_rows(report)),
                       file_name=f"frequencia_aula_{class_id}.csv", mime="text/csv", use_container_width=True)
    d2.download_button("Entradas e saídas (CSV)", data=to_csv(interval_rows(report)),
                       file_name=f"entradas_saidas_aula_{class_id}.csv", mime="text/csv", use_container_width=True)
    d3.download_button("Relatório imprimível (HTML)", data=report_html(session, class_id, report),
                       file_name=f"relatorio_aula_{class_id}.html", mime="text/html", use_container_width=True)


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

    painel, reconhecer, cadastro, aulas, relatorio, diagnostico = st.tabs(
        ["Painel", "Reconhecer", "Cadastrar aluno", "Aulas", "Relatório", "Diagnóstico"])
    with painel:
        tab_dashboard(class_id, sessions)
    with reconhecer:
        tab_recognize(class_id, sessions)
    with cadastro:
        tab_enroll()
    with aulas:
        tab_sessions(class_id, sessions)
    with relatorio:
        tab_report(class_id)
    with diagnostico:
        tab_diagnostics()


main()