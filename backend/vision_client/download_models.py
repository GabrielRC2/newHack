"""Baixa explicitamente os modelos ONNX oficiais do OpenCV Zoo para uso local."""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path


MODELS = {
    "face_detection_yunet_2023mar.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}
MODELS_DIR = Path(__file__).resolve().parent / "models"


def main() -> None:
    parser = argparse.ArgumentParser(description="Baixa modelos YuNet e SFace do OpenCV Zoo")
    parser.add_argument("--force", action="store_true", help="baixa novamente mesmo se o arquivo existir")
    args = parser.parse_args()
    MODELS_DIR.mkdir(exist_ok=True)
    for filename, url in MODELS.items():
        target = MODELS_DIR / filename
        if target.exists() and not args.force:
            print(f"Ja existe: {target}")
            continue
        temporary = target.with_suffix(".download")
        print(f"Baixando {filename}...")
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(target)
        print(f"Salvo em {target}")


if __name__ == "__main__":
    main()
