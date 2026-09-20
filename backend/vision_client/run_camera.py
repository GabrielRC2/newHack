"""Executa a webcam, gera embedding e envia somente uma deteccao por passagem."""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2 as cv

from .api import FrequencyApi
from .face_engine import FaceEngine, draw_face


ROOT = Path(__file__).resolve().parent
DEFAULT_DETECTOR = ROOT / "models" / "face_detection_yunet_2023mar.onnx"
DEFAULT_RECOGNIZER = ROOT / "models" / "face_recognition_sface_2021dec.onnx"


def open_camera(index: int) -> cv.VideoCapture:
    # DirectShow reduz a demora de abertura da webcam em Windows; os demais SOs usam o padrao.
    camera = cv.VideoCapture(index, cv.CAP_DSHOW) if sys.platform.startswith("win") else cv.VideoCapture(index)
    if not camera.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir a camera USB de indice {index}")
    return camera


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Camera USB para o backend de frequencia")
    parser.add_argument("--class-id", type=int, default=int(os.getenv("VISION_CLASS_ID", "1")))
    parser.add_argument("--camera-id", default=os.getenv("VISION_CAMERA_ID", "usb-entrada-01"))
    parser.add_argument("--api-url", default=os.getenv("FREQUENCY_API_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--camera-index", type=int, default=int(os.getenv("VISION_CAMERA_INDEX", "0")))
    parser.add_argument("--detector-model", type=Path, default=DEFAULT_DETECTOR)
    parser.add_argument("--recognition-model", type=Path, default=DEFAULT_RECOGNIZER)
    parser.add_argument("--absence-seconds", type=float, default=1.0,
                        help="tempo sem rosto que rearma uma nova passagem")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = FaceEngine(args.detector_model, args.recognition_model)
    api = FrequencyApi(args.api_url, os.getenv("FREQUENCY_API_TOKEN"))
    camera = open_camera(args.camera_index)
    armed = True
    last_face_at = 0.0
    last_message = "Aguardando uma pessoa passar pela camera"
    last_color = (255, 255, 255)

    print("Camera ativa. Pressione Q na janela para encerrar.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Falha ao ler a webcam")
            now = time.monotonic()
            faces = engine.detect(frame)

            if len(faces) == 0:
                if now - last_face_at >= args.absence_seconds:
                    armed = True
                    last_message, last_color = "Aguardando uma pessoa passar pela camera", (255, 255, 255)
            elif len(faces) > 1:
                last_face_at = now
                last_message, last_color = "Mais de um rosto: passe um aluno por vez", (0, 165, 255)
                for face in faces:
                    draw_face(frame, face, last_color)
            else:
                last_face_at = now
                face = faces[0]
                draw_face(frame, face, (0, 210, 0) if armed else last_color)
                if armed:
                    # O disparo unico por passagem evita que frames consecutivos parecam uma saida.
                    armed = False
                    try:
                        response = api.send_recognition(args.class_id, args.camera_id, engine.embedding(frame, face), datetime.now())
                        last_message = response["message"]
                        last_color = (0, 210, 0) if response["status"] == "identified" else (0, 165, 255)
                        print(response)
                    except Exception as exc:  # Mantem a camera viva caso a API caia temporariamente.
                        last_message, last_color = f"Erro ao enviar: {exc}", (0, 0, 255)
                        print(last_message)

            cv.rectangle(frame, (0, 0), (frame.shape[1], 38), (25, 25, 25), -1)
            cv.putText(frame, last_message[:95], (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.58, last_color, 2)
            cv.imshow("Frequencia - camera USB (Q para sair)", frame)
            if cv.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    finally:
        camera.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
