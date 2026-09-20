import os
from datetime import datetime, timedelta

os.environ["DATABASE_URL"] = "sqlite://"
os.environ["RECOGNITION_COOLDOWN_SECONDS"] = "15"

from fastapi.testclient import TestClient

from app.database import Base, engine
from app.main import app

client = TestClient(app)


def setup_function():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


def post_recognition(class_id, at, embedding=None):
    return client.post("/recognitions", json={
        "class_id": class_id, "camera_id": "usb-1", "embedding": embedding or [1, 0, 0], "timestamp": at.isoformat()
    })


def create_student_and_session(minimum=0.75):
    student = client.post("/students", json={"name": "Ana", "enrollment_number": "2026001", "embedding": [1, 0, 0]}).json()
    start = datetime(2026, 9, 20, 15, 0)
    session = client.post("/sessions", json={"label": "Teste", "starts_at": start.isoformat(), "ends_at": (start + timedelta(hours=1)).isoformat(), "minimum_percentage": minimum}).json()
    return student, session, start


def test_entry_exit_and_presence_rule():
    student, session, start = create_student_and_session()
    assert post_recognition(session["id"], start).json()["action"] == "entered"
    assert post_recognition(session["id"], start + timedelta(minutes=50)).json()["action"] == "exited"
    final = client.post(f"/sessions/{session['id']}/finalize").json()
    assert final["present"] == 1
    report = client.get(f"/sessions/{session['id']}/students/{student['id']}/attendance").json()
    assert report["total_seconds"] == 3000
    assert report["status"] == "present"


def test_bathroom_return_sums_intervals():
    student, session, start = create_student_and_session()
    post_recognition(session["id"], start)
    post_recognition(session["id"], start + timedelta(minutes=10))
    assert post_recognition(session["id"], start + timedelta(minutes=20)).json()["action"] == "returned"
    post_recognition(session["id"], start + timedelta(minutes=50))
    client.post(f"/sessions/{session['id']}/finalize")
    report = client.get(f"/sessions/{session['id']}/students/{student['id']}/attendance").json()
    assert report["total_seconds"] == 2400  # 10 + 30 min
    assert len(report["intervals"]) == 2


def test_cooldown_does_not_toggle_state():
    _, session, start = create_student_and_session()
    assert post_recognition(session["id"], start).json()["action"] == "entered"
    response = post_recognition(session["id"], start + timedelta(seconds=5)).json()
    assert response["status"] == "cooldown"
    response = post_recognition(session["id"], start + timedelta(seconds=16)).json()
    assert response["action"] == "exited"


def test_unregistered_and_closed_class_are_safe():
    _, session, start = create_student_and_session()
    assert post_recognition(session["id"], start, [0, 1, 0]).json()["status"] == "unregistered"
    client.post(f"/sessions/{session['id']}/finalize")
    assert post_recognition(session["id"], start + timedelta(minutes=30)).json()["status"] == "class_closed"


def test_entry_before_start_is_clipped_to_official_period():
    student, session, start = create_student_and_session(minimum=0.75)
    post_recognition(session["id"], start - timedelta(minutes=10))
    post_recognition(session["id"], start + timedelta(minutes=45))
    client.post(f"/sessions/{session['id']}/finalize")
    report = client.get(f"/sessions/{session['id']}/students/{student['id']}/attendance").json()
    assert report["total_seconds"] == 2700
    assert report["status"] == "present"
