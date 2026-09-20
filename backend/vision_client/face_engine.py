"""Detector YuNet e extrator SFace, ambos via OpenCV e modelos ONNX locais."""
from __future__ import annotations

from pathlib import Path

import cv2 as cv
import numpy as np


class FaceEngine:
    """Extrai um embedding normalizado para cada rosto detectado no frame."""

    def __init__(
        self,
        detector_model: str | Path,
        recognition_model: str | Path,
        score_threshold: float = 0.9,
        nms_threshold: float = 0.3,
    ) -> None:
        detector_path = Path(detector_model)
        recognition_path = Path(recognition_model)
        missing = [str(path) for path in (detector_path, recognition_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Modelo(s) ONNX nao encontrado(s): " + ", ".join(missing) +
                ". Execute: python -m vision_client.download_models"
            )
        if not hasattr(cv, "FaceDetectorYN") or not hasattr(cv, "FaceRecognizerSF"):
            raise RuntimeError("Sua instalacao do OpenCV nao possui FaceDetectorYN/FaceRecognizerSF. Atualize opencv-python.")

        self.detector = cv.FaceDetectorYN.create(
            str(detector_path), "", (320, 320), score_threshold, nms_threshold, 5000
        )
        self.recognizer = cv.FaceRecognizerSF.create(str(recognition_path), "")

    def detect(self, frame: np.ndarray) -> list[np.ndarray]:
        """Retorna linhas [x, y, w, h, 5 landmarks, score] para os rostos do frame."""
        height, width = frame.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(frame)
        return [] if faces is None else [face for face in faces]

    def embedding(self, frame: np.ndarray, face: np.ndarray) -> list[float]:
        aligned = self.recognizer.alignCrop(frame, face)
        feature = self.recognizer.feature(aligned).flatten().astype(np.float32)
        norm = float(np.linalg.norm(feature))
        if norm == 0:
            raise ValueError("O modelo retornou um embedding nulo")
        return (feature / norm).tolist()


def draw_face(frame: np.ndarray, face: np.ndarray, color: tuple[int, int, int] = (0, 210, 0)) -> None:
    """Desenha somente o retangulo/score; nenhum frame ou foto e salvo."""
    x, y, width, height = (int(value) for value in face[:4])
    score = float(face[-1])
    cv.rectangle(frame, (x, y), (x + width, y + height), color, 2)
    cv.putText(frame, f"face {score:.2f}", (x, max(22, y - 8)), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
