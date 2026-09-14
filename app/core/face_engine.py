import logging
import os
from functools import lru_cache
from insightface.app.common import Face
from insightface.model_zoo import model_zoo
from app.core.config import settings
logger = logging.getLogger(__name__)

class FaceEngine:
    """Carga el detector SCRFD y el reconocedor AuraFace directamente por
    ruta de archivo (sin pasar por FaceAnalysis), para no depender jamas
    del sistema de descarga de paquetes de InsightFace (que ofrece por
    defecto modelos de uso no comercial como buffalo_l).
    """
    def __init__(self,detector_path: str,recognizer_path: str,providers: list[str],det_size: tuple[int, int],det_thresh: float,ctx_id: int,):
        for path, label in ((detector_path, "detector SCRFD"), (recognizer_path, "reconocedor AuraFace")):
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"No se encontro el modelo {label} en '{path}'. "
                    "Verifica que el archivo .onnx exista en app/models/."
                )

        self.detector = model_zoo.get_model(detector_path, providers=providers)
        if self.detector is None or self.detector.taskname != "detection":
            raise RuntimeError(
                f"El archivo '{detector_path}' no fue reconocido como un modelo de deteccion (SCRFD)."
            )
        self.detector.prepare(ctx_id, input_size=det_size, det_thresh=det_thresh)
        self.recognizer = model_zoo.get_model(recognizer_path, providers=providers)
        if self.recognizer is None or self.recognizer.taskname != "recognition":
            raise RuntimeError(
                f"El archivo '{recognizer_path}' no fue reconocido como un modelo de reconocimiento (AuraFace)."
            )
        self.recognizer.prepare(ctx_id)

        logger.info(
            "FaceEngine listo | detector=%s | recognizer=%s | providers=%s",
            os.path.basename(detector_path),
            os.path.basename(recognizer_path),
            providers,
        )

    def get_faces(self, img, max_num: int = 0) -> list[Face]:
        """Detecta rostros y calcula el embedding de cada uno."""
        bboxes, kpss = self.detector.detect(img, max_num=max_num, metric="default")
        faces: list[Face] = []
        for i in range(bboxes.shape[0]):
            face = Face(
                bbox=bboxes[i, 0:4],
                kps=kpss[i] if kpss is not None else None,
                det_score=bboxes[i, 4],
            )
            self.recognizer.get(img, face)
            faces.append(face)
        return faces

@lru_cache(maxsize=1)
def get_face_engine() -> FaceEngine:
    """Singleton: el modelo se carga una sola vez por proceso."""
    return FaceEngine(
        detector_path=settings.detector_path,
        recognizer_path=settings.recognizer_path,
        providers=settings.onnx_providers,
        det_size=settings.detection_size,
        det_thresh=settings.detection_threshold,
        ctx_id=settings.ctx_id,
    )
