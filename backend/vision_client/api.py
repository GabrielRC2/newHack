"""Contrato HTTP do cliente de visao com o backend FastAPI."""
from __future__ import annotations

from datetime import datetime

import requests


class FrequencyApi:
    def __init__(self, base_url: str, token: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"

    @staticmethod
    def _json_or_error(response: requests.Response) -> dict:
        try:
            body = response.json()
        except ValueError:
            body = {"detail": response.text}
        if not response.ok:
            raise RuntimeError(f"API retornou HTTP {response.status_code}: {body.get('detail', body)}")
        return body

    def send_recognition(self, class_id: int, camera_id: str, embedding: list[float], timestamp: datetime) -> dict:
        response = self.session.post(
            f"{self.base_url}/recognitions",
            json={
                "class_id": class_id,
                "camera_id": camera_id,
                "embedding": embedding,
                # O backend MVP usa hora local sem timezone. Mantenha computador/cadastro de aula no mesmo fuso.
                "timestamp": timestamp.isoformat(timespec="seconds"),
            },
            timeout=8,
        )
        return self._json_or_error(response)

    def create_student(self, name: str, enrollment_number: str, embedding: list[float]) -> dict:
        response = self.session.post(
            f"{self.base_url}/students",
            json={"name": name, "enrollment_number": enrollment_number, "embedding": embedding},
            timeout=10,
        )
        return self._json_or_error(response)

    def add_embedding(self, student_id: int, embedding: list[float]) -> dict:
        response = self.session.post(
            f"{self.base_url}/students/{student_id}/embeddings",
            json={"vector": embedding},
            timeout=10,
        )
        return self._json_or_error(response)
