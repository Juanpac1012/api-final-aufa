import numpy as np
from app.enums import ResultadoVerificacion, CodigoError, TipoImagen
from app.parameters import UMBRAL_SIMILITUD


def _codigo_embedding_invalido(tipo_imagen: TipoImagen) -> CodigoError:
    if tipo_imagen == TipoImagen.ACTUAL:
        return CodigoError.EMBEDDING_INVALIDO_ACTUAL
    return CodigoError.EMBEDDING_INVALIDO_REFERENCIA

def comparar_rostros_1a1(rostro_actual, rostro_referencia):
    embedding_actual = rostro_actual.embedding
    embedding_referencia = rostro_referencia.embedding
    # VALIDAR QUE EXISTAN LOS EMBEDDINGS (por separado, para saber cual fallo)
    if embedding_actual is None:
        return {
            "valido": False,
            "codigo": _codigo_embedding_invalido(TipoImagen.ACTUAL),
            "mensaje": "No fue posible obtener el embedding facial de la imagen actual."
        }
    if embedding_referencia is None:
        return {
            "valido": False,
            "codigo": _codigo_embedding_invalido(TipoImagen.REFERENCIA),
            "mensaje": "No fue posible obtener el embedding facial de la imagen de referencia."
        }

    # COPIA EXPLICITA - nunca mutar el array interno del objeto Face
    embedding_actual = np.array(embedding_actual, dtype=np.float32, copy=True)
    embedding_referencia = np.array(embedding_referencia, dtype=np.float32, copy=True)

    # VALIDAR DIMENSIONES
    if embedding_actual.ndim != 1:
        return {
            "valido": False,
            "codigo": _codigo_embedding_invalido(TipoImagen.ACTUAL),
            "mensaje": "El embedding facial de la imagen actual tiene un formato invalido."
        }
    if embedding_referencia.ndim != 1:
        return {
            "valido": False,
            "codigo": _codigo_embedding_invalido(TipoImagen.REFERENCIA),
            "mensaje": "El embedding facial de la imagen de referencia tiene un formato invalido."
        }

    if embedding_actual.shape != embedding_referencia.shape:
        return {
            "valido": False,
            "codigo": CodigoError.ERROR_INTERNO,
            "mensaje": "Los embeddings faciales no tienen la misma dimension."
        }

    # CALCULAR NORMAS
    norma_actual = np.linalg.norm(embedding_actual)
    norma_referencia = np.linalg.norm(embedding_referencia)

    if norma_actual == 0:
        return {
            "valido": False,
            "codigo": _codigo_embedding_invalido(TipoImagen.ACTUAL),
            "mensaje": "No fue posible normalizar el embedding facial de la imagen actual."
        }
    if norma_referencia == 0:
        return {
            "valido": False,
            "codigo": _codigo_embedding_invalido(TipoImagen.REFERENCIA),
            "mensaje": "No fue posible normalizar el embedding facial de la imagen de referencia."
        }

    # NORMALIZAR (ya son copias, seguro mutar)
    embedding_actual /= norma_actual
    embedding_referencia /= norma_referencia

    similitud = float(np.dot(embedding_actual, embedding_referencia))
    similitud = max(-1.0, min(1.0, similitud))

    if similitud >= UMBRAL_SIMILITUD:
        resultado = ResultadoVerificacion.COINCIDE
    else:
        resultado = ResultadoVerificacion.NO_COINCIDE

    return {
        "valido": True,
        "resultado": resultado,
        "similitud": round(similitud, 4)
    }