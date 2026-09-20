import os
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


class RecognitionWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_file = Path(self.temporary_directory.name) / "test.db"
        self.previous_database_path = os.environ.get("ATTENDANCE_DB_PATH")
        os.environ["ATTENDANCE_DB_PATH"] = str(self.database_file)
        self.client_context = TestClient(app)
        self.client = self.client_context.__enter__()

        self.student_id = self.client.post(
            "/alunos", json={"nome": "Ana Silva", "matricula": "2026001"}
        ).json()["id"]
        self.client.post(
            "/cameras",
            json={"codigo": "entrada-01", "nome": "Entrada", "papel": "ENTRADA"},
        )
        self.client.post(
            "/cameras",
            json={"codigo": "saida-01", "nome": "Saída", "papel": "SAIDA"},
        )
        self.session_id = self.client.post(
            "/sessoes-aula",
            json={
                "nome": "Algoritmos",
                "camera_saida_codigo": "saida-01",
                "inicio_em": "2026-09-20T14:00:00-03:00",
                "fim_em": "2026-09-20T16:00:00-03:00",
            },
        ).json()["id"]

    def tearDown(self) -> None:
        self.client_context.__exit__(None, None, None)
        if self.previous_database_path is None:
            os.environ.pop("ATTENDANCE_DB_PATH", None)
        else:
            os.environ["ATTENDANCE_DB_PATH"] = self.previous_database_path
        self.temporary_directory.cleanup()

    def recognize(self, camera_code: str, occurred_at: str, student_id: int | None = None):
        payload = {"camera_codigo": camera_code, "ocorreu_em": occurred_at}
        if student_id is not None:
            payload["aluno_id"] = student_id
        return self.client.post("/eventos-reconhecimento", json=payload)

    def test_exit_after_cutoff_records_one_attendance(self) -> None:
        early_event = self.recognize("saida-01", "2026-09-20T14:50:00-03:00", self.student_id)
        self.assertEqual(early_event.status_code, 201)
        self.assertEqual(early_event.json()["decisao"], "IDENTIFICADO_ANTES_DO_CORTE")

        qualifying_event = self.recognize(
            "saida-01", "2026-09-20T18:01:00Z", self.student_id
        )
        self.assertEqual(qualifying_event.status_code, 201)
        self.assertEqual(qualifying_event.json()["decisao"], "PRESENCA_REGISTRADA")

        duplicate_event = self.recognize(
            "saida-01", "2026-09-20T15:30:00-03:00", self.student_id
        )
        self.assertEqual(duplicate_event.status_code, 201)
        self.assertEqual(duplicate_event.json()["decisao"], "PRESENCA_JA_REGISTRADA")

        attendance = self.client.get(f"/sessoes-aula/{self.session_id}/presencas")
        self.assertEqual(attendance.status_code, 200)
        self.assertEqual(len(attendance.json()), 1)
        self.assertEqual(attendance.json()[0]["aluno_id"], self.student_id)
        self.assertEqual(attendance.json()[0]["registrado_em"], "2026-09-20T18:01:00Z")

    def test_entry_camera_never_records_attendance(self) -> None:
        event = self.recognize("entrada-01", "2026-09-20T15:30:00-03:00", self.student_id)
        self.assertEqual(event.status_code, 201)
        self.assertEqual(event.json()["decisao"], "ENTRADA_IDENTIFICADA")

        attendance = self.client.get(f"/sessoes-aula/{self.session_id}/presencas")
        self.assertEqual(attendance.json(), [])

    def test_exit_exactly_at_cutoff_does_not_record_attendance(self) -> None:
        event = self.recognize("saida-01", "2026-09-20T15:00:00-03:00", self.student_id)
        self.assertEqual(event.status_code, 201)
        self.assertEqual(event.json()["decisao"], "IDENTIFICADO_ANTES_DO_CORTE")

        attendance = self.client.get(f"/sessoes-aula/{self.session_id}/presencas")
        self.assertEqual(attendance.json(), [])

    def test_unknown_face_is_audited_without_attendance(self) -> None:
        event = self.recognize("saida-01", "2026-09-20T15:30:00-03:00")
        self.assertEqual(event.status_code, 201)
        self.assertEqual(event.json()["decisao"], "ROSTO_DESCONHECIDO")

        attendance = self.client.get(f"/sessoes-aula/{self.session_id}/presencas")
        self.assertEqual(attendance.json(), [])

    def test_unknown_face_at_entry_requests_registration(self) -> None:
        event = self.recognize("entrada-01", "2026-09-20T14:10:00-03:00")
        self.assertEqual(event.status_code, 201)
        self.assertEqual(event.json()["decisao"], "CADASTRO_NECESSARIO")

    def test_overlapping_sessions_are_rejected(self) -> None:
        response = self.client.post(
            "/sessoes-aula",
            json={
                "nome": "Estruturas de Dados",
                "camera_saida_codigo": "saida-01",
                "inicio_em": "2026-09-20T15:00:00-03:00",
                "fim_em": "2026-09-20T17:00:00-03:00",
            },
        )
        self.assertEqual(response.status_code, 409)

    def test_event_at_lesson_end_has_no_active_session(self) -> None:
        event = self.recognize("saida-01", "2026-09-20T16:00:00-03:00", self.student_id)
        self.assertEqual(event.status_code, 201)
        self.assertEqual(event.json()["decisao"], "IDENTIFICADO_SEM_AULA_ATIVA")

    def test_entry_camera_cannot_be_used_as_an_exit_session_camera(self) -> None:
        response = self.client.post(
            "/sessoes-aula",
            json={
                "nome": "Banco de Dados",
                "camera_saida_codigo": "entrada-01",
                "inicio_em": "2026-09-20T18:00:00-03:00",
                "fim_em": "2026-09-20T20:00:00-03:00",
            },
        )
        self.assertEqual(response.status_code, 422)

    def test_duplicate_registration_is_rejected(self) -> None:
        response = self.client.post(
            "/alunos", json={"nome": "Outra Ana", "matricula": "2026001"}
        )
        self.assertEqual(response.status_code, 409)

    def test_event_timestamp_requires_an_explicit_timezone(self) -> None:
        response = self.recognize("saida-01", "2026-09-20T15:30:00", self.student_id)
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
