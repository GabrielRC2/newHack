"""Captura embeddings de um aluno usando a mesma camera/modelo do reconhecimento."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2 as cv

from .api import FrequencyApi
from .face_engine import FaceEngine, draw_face
from .run_camera import DEFAULT_DETECTOR, DEFAULT_RECOGNIZER, open_camera


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cadastrar aluno com embeddings capturados pela webcam")
    parser.add_argument("--name", required=True)
    parser.add_argument("--enrollment-number", required=True)
    parser.add_argument("--api-url", default=os.getenv("FREQUENCY_API_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--camera-index", type=int, default=int(os.getenv("VISION_CAMERA_INDEX", "0")))
    parser.add_argument("--samples", type=int, default=3, choices=range(1, 8))
    parser.add_argument("--detector-model", type=Path, default=DEFAULT_DETECTOR)
    parser.add_argument("--recognition-model", type=Path, default=DEFAULT_RECOGNIZER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = FaceEngine(args.detector_model, args.recognition_model)
    api = FrequencyApi(args.api_url, os.getenv("FREQUENCY_API_TOKEN"))
    camera = open_camera(args.camera_index)
    samples: list[list[float]] = []
    print("Olhe para a camera; pressione C para capturar cada amostra e Q para cancelar.")
    try:
        while len(samples) < args.samples:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Falha ao ler a webcam")
            faces = engine.detect(frame)
            valid = len(faces) == 1
            for face in faces:
                draw_face(frame, face, (0, 210, 0) if valid else (0, 0, 255))
            instruction = f"Amostras: {len(samples)}/{args.samples}. C captura; Q cancela"
            if not valid:
                instruction = "Mantenha exatamente um rosto no enquadramento"
            cv.putText(frame, instruction, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
            cv.imshow("Cadastro facial", frame)
            key = cv.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                print("Cadastro cancelado; nenhum embedding enviado.")
                return
            if key in (ord("c"), ord("C")) and valid:
                samples.append(engine.embedding(frame, faces[0]))
                print(f"Amostra {len(samples)}/{args.samples} capturada")
    finally:
        camera.release()
        cv.destroyAllWindows()

    created = api.create_student(args.name, args.enrollment_number, samples[0])
    for sample in samples[1:]:
        api.add_embedding(created["id"], sample)
    print(f"Aluno cadastrado com {len(samples)} embeddings: id={created['id']}")


if __name__ == "__main__":
    main()
