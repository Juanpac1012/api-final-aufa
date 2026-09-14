import base64
import binascii
import logging
import cv2
import numpy as np
from app.enums import CodigoError, TipoImagen

logger = logging.getLogger(__name__)

def _codigo_no_decodificable(tipo_imagen: TipoImagen) -> CodigoError:
    if tipo_imagen == TipoImagen.ACTUAL:
        return CodigoError.IMAGEN_NO_DECODIFICABLE_ACTUAL
    return CodigoError.IMAGEN_NO_DECODIFICABLE_REFERENCIA

def base64_a_imagen(base64_string: str, tipo_imagen: TipoImagen) -> dict:
    """Convierte un string base64 a una imagen OpenCV (BGR). """
    nombre_imagen = tipo_imagen.value

    if not base64_string or not isinstance(base64_string, str):
        return {
            "valido": False,
            "codigo": CodigoError.IMAGEN_INVALIDA,
            "mensaje": f"No se recibio la imagen {nombre_imagen}."
        }
    try:
        # QUITAR ENCABEZADO DATA URI
        base64_data = base64_string.strip()
        if "," in base64_data:
            encabezado, base64_data = base64_data.split(",", 1)
            if not encabezado.lower().startswith("data:image/"):
                return {
                    "valido": False,
                    "codigo": CodigoError.IMAGEN_INVALIDA,
                    "mensaje": f"El formato de la imagen {nombre_imagen} no es valido."
                }
        # BASE64 -> BYTES
        try:
            image_bytes = base64.b64decode(base64_data, validate=True)
        except (binascii.Error, ValueError):
            return {
                "valido": False,
                "codigo": CodigoError.IMAGEN_INVALIDA,
                "mensaje": f"El contenido Base64 de la imagen {nombre_imagen} no es valido."
            }

        if not image_bytes:
            return {
                "valido": False,
                "codigo": CodigoError.IMAGEN_INVALIDA,
                "mensaje": f"La imagen {nombre_imagen} esta vacia."
            }
        # BYTES -> NUMPY -> OPENCV
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        imagen = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if imagen is None:
            return {
                "valido": False,
                "codigo": _codigo_no_decodificable(tipo_imagen),
                "mensaje": f"No se pudo cargar la imagen {nombre_imagen}."
            }

        return {
            "valido": True,
            "imagen": imagen
        }
    except Exception:
        logger.exception("Error inesperado procesando la imagen %s", nombre_imagen)
        return {
            "valido": False,
            "codigo": CodigoError.ERROR_INTERNO,
            "mensaje": f"No se pudo procesar la imagen {nombre_imagen}."
        }